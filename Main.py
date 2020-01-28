# Initial framework taken from https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py

import numpy as np

import gym

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras import backend as K
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import MobileNetV2

import numba as nb
from tensorboardX import SummaryWriter

ENV = 'LunarLander-v2'
CONTINUOUS = False
import gfootball.env as football_env

EPISODES = 100000

LOSS_CLIPPING = 0.2 # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 1.0 # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 255
BATCH_SIZE = 256
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly



@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss

class Agent:
    def __init__(self):

        self.env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
        self.state_dims = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.actor = self.build_actor(self.state_dims, self.n_actions, self.n_actions)
        self.critic = self.build_critic(self.state_dims)

        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.name = self.get_name()
        self.writer = SummaryWriter(self.name)
        self.gradient_steps = 0

        self.dummy_action, self.dummy_value = np.zeros((1, self.n_actions)), np.zeros((1, 1))

    def get_name(self):
        name = 'AllRuns/'
        if CONTINUOUS is True:
            name += 'continous/'
        else:
            name += 'discrete/'
        name += ENV
        return name

    def build_actor(self, input_dims, output_dims, n_actions):
        state_input = Input(shape=input_dims)
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(output_dims,))

        feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

        for layer in feature_extractor.layers:
            layer.trainable = False

        # Classification block
        x = Flatten(name='flatten')(feature_extractor(state_input))
        x = Dense(1024, activation='relu', name='fc1')(x)
        out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

        model = Model(inputs=[state_input, old_prediction, advantage],
                      outputs=[out_actions])
        model.compile(optimizer=Adam(lr=1e-4), loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()
        return model


    def build_critic(self, input_dims):

        state_input = Input(shape=input_dims)

        feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

        for layer in feature_extractor.layers:
            layer.trainable = False

        # Classification block
        x = Flatten(name='flatten')(feature_extractor(state_input))
        x = Dense(1024, activation='relu', name='fc1')(x)
        out_actions = Dense(1, activation='tanh')(x)

        model = Model(inputs=[state_input], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        model.summary()
        return model

    def reset_env(self):
        self.episode += 1
        if self.episode % 100 == 0:
            self.val = True
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):

        p = self.actor.predict([np.expand_dims(self.observation, 0),self.dummy_action, self.dummy_value])
        print(p)
        if self.val is False:

            action = np.random.choice(self.n_actions, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(self.n_actions)
        action_matrix[action] = 1
        return action, action_matrix, p

    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, predicted_action = self.get_action()

            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values

            actor_loss = self.actor.fit([obs, old_prediction, advantage], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=True)
            critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=True)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

            self.gradient_steps += 1


if __name__ == '__main__':
    ag = Agent()
    ag.run()
