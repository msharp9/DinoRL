#!/usr/bin/python3

from PIL import Image, ImageOps
import pyautogui
import time
import numpy as np
import random

import sys
from mss import mss
import mss.tools as tools

import asyncio
import imageio

from collections import deque
import keras
import tensorflow as tf

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

class DDQN_brain():
    memory = deque(maxlen=30000)
    global_step = 0
    state_size = (100, 140, 4)
    action_size = 4

    def build_model(state_size, action_size):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), activation='relu',
                         input_shape=state_size))
        model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(action_size)) # action_size
        model.summary()
        return model

    # init global models
    model = build_model(state_size, action_size)
    target_model = build_model(state_size, action_size)
    last_conv2d_layername = 'conv2d_3'

    sess = tf.InteractiveSession(config=config)
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())

    def __init__(self, model_path="model/ddqn.h5", record="records/record.txt",
        learning_rate=0.01, reward_decay=0.99, epsilon=0.05, explore=False,):
        self.record = record
        self.model_path = model_path

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon
        self.epsilon_end = 0.05
        self.exploration_steps = 1000000.
        if explore:
            self.epsilon_decay_step = (self.epsilon - self.epsilon_end) \
                                  / self.exploration_steps
        else:
            self.epsilon_decay_step = 0

        # parameters about training
        self.batch_size = 128
        self.train_start = 10000
        self.update_target_rate = 10000
        self.no_op_steps = 30

        self.update_target_model()
        self.optimizer = self.optimizer()
        if self.model_path and os.path.isfile(self.model_path):
            self.model.load_weights(self.model_path)

        self.history = None
        self.action = None
        self.reward = 0
        self.tot_reward = 0
        self.avg_q_max, self.avg_loss = 0, 0

    # Hue loss:
    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None, ), dtype='int32')
        y = K.placeholder(shape=(None, ), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)
        return train

    # Double DQN - double part
    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # save sample <s,a,r,s'> to the replay memory
    def replay_memory(self, s, a, r, s_, end):
        self.memory.append((s, a, r, s_, end))

    def grad_cam_heatmap(self, action, history):
        model_output = self.model.output[:,action]
        last_conv_layer = self.model.get_layer(self.last_conv2d_layername)

        grads = K.gradients(model_output,last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0,1,2))
        iterate = K.function([self.model.input, K.learning_phase()],
            [pooled_grads, last_conv_layer.output[0]])

        pooled_grads_value, conv_layer_output_value = iterate([history, 0])

        for i, pgv in enumerate(pooled_grads_value):
            conv_layer_output_value[:,:,i] *= pgv

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        # heatmap = np.absolute(heatmap)
        heatmap = np.maximum(heatmap,0)
        heatmap /= np.max(heatmap) + 0.00000001

        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

    def merge_heatmap(self, img, heatmap):
        heatmap = cv2.resize(heatmap, (img.shape[1],img.shape[0]))
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        superimposed_img = heatmap*0.4 + img
        return superimposed_img


    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_end:
            self.epsilon += self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size, ))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        value = self.model.predict(next_history)
        target_value = self.target_model.predict(next_history)

        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                target[i] = reward[i] + self.gamma * target_value[i][np.argmax(value[i])]

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]


class Bot(DDQN_brain):
    """Bot for playing Chrome dino run game"""
    def __init__(self, action_space=None, gif=False, grad_cam=False,
            **kwargs):
        self.area = pyautogui.locateOnScreen('dino_start.png', confidence=0.9,
            grayscale=True)
        # self.area = [631, 265, 729, 225]
        if not self.area:
            print("Game area could not be found. Please load up a new game at http://www.trex-game.skipser.com/.")
            sys.exit()
        self.restart_coords = pyautogui.center(self.area)
        self.observation_area = {"top": self.area.top + 110,
            "left": self.area.left + 130, "width": 140, "height": 100}
        self.gameover_area = {"top": self.area.top + 60,
            "left": self.area.left + 250, "width": 250, "height": 40}
        self.mss = mss()
        self.start = time.time()
        self.time = 0

        self.gif = gif
        self.gifimages = []
        self.grad_cam = grad_cam

        # ADDED THE CHOICES #
        self.choices = {0: self.duck,
                        1: self.jump,
                        2: self.short_jump,
                        3: self.walk,
                        }
        self.choicestext = {0: "duck",
                        1: "jump",
                        2: "short_jump",
                        3: "walk",
                        }
        if action_space:
            self.actions = action_space
        else:
            self.actions = [str(c) for c in self.choices]
        self.action_size = len(self.actions)

    def restart(self):
        pyautogui.click(self.restart_coords)
        time.sleep(0.05)
        pyautogui.press('space')

    async def duck(self):
        pyautogui.keyDown('down')
        await asyncio.sleep(0.01)

    async def jump(self):
        pyautogui.keyUp('down')
        pyautogui.keyDown('space')
        await asyncio.sleep(0.05)
        pyautogui.keyUp('space')

    async def short_jump(self):
        pyautogui.keyUp('down')
        pyautogui.keyDown('space')
        pyautogui.keyUp('space')

    async def walk(self):
        pyautogui.keyUp('down')
        await asyncio.sleep(0.01)

    def detection_area(self):
        """
        Area right in front of dino.
        """
        sct = self.mss.grab(self.observation_area)
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        gray_img = ImageOps.grayscale(img)
        arr = np.array(gray_img)
        print(arr.shape)
        return arr, img

    def mean_pixel(self, arr):
        return arr.mean()

    def check_dead(self):
        sct = self.mss.grab(self.gameover_area)
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        gray_img = ImageOps.grayscale(img)
        arr = np.array(gray_img)
        mpixel = self.mean_pixel(arr)
        return mpixel > 230 and mpixel < 239

    def random_agent(self):
        self.restart()
        while True:
            r = random.random()
            if r > 0.75:
                self.jump()
            elif r > 0.5:
                self.short_jump()
            elif r > 0.25:
                self.duck()
            else:
                self.walk()

    async def basic_jump(self, thresh):
        while True:
            arr, img = self.detection_area()
            mpixel = self.mean_pixel(arr)
            if mpixel > 150 and mpixel < thresh:
                await self.jump()
                # print("mpixel: {}".format(mpixel))
                # img.show()
            await asyncio.sleep(0.01)

    async def sync_is_dead(self):
        dead = None
        while not dead:
            dead = self.check_dead()
            await asyncio.sleep(0.05)

    def basic_agent(self):
        BA_THRESHOLD = 241
        # start = time.time()
        self.restart()

        loop = asyncio.get_event_loop()
        done, pending = loop.run_until_complete(
            asyncio.wait([asyncio.ensure_future(self.sync_is_dead()),
                asyncio.ensure_future(self.basic_jump(BA_THRESHOLD))],
                return_when=asyncio.FIRST_COMPLETED))

        for task in pending:
            task.cancel()
        # print(time.time() - start)

    async def rl_agent(self, history):


        try:
            await self.choices[int(self.action)]
        except Exception as e:
            print(str(e))
            pass

    def choose_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() >= self.epsilon:
            q_value = self.model.predict(history)
            action = np.argmax(q_value[0])
        else:
            action = random.randrange(0,4)
        return action

    async def on_step(self):
        _observation, _img = self.detection_area()
        _time = time.time()
        if self.action is not None:
            _observation = np.reshape([_observation], (1, 100, 140, 1))
            _history = np.append(_observation, self.history[:,:,:,:3], axis=3)
            self.replay_memory(self.history, self.action, self.reward, _history, False)
            self.train_replay()
            self.reward = _time - self.time
            print(self.reward)
            self.tot_reward += self.reward
            self.global_step += 1
            if self.global_step % self.update_target_rate == 0:
                self.update_target_model()
        else:
            _history = np.stack((_observation, _observation, _observation, _observation), axis=2)
            _history = np.reshape([_history], (1, 100, 140, 4))
        self._history = self.history
        self.history = _history
        self.time = _time
        self.action = self.choose_action(_history)

    def on_end(self):
        print('--- on_end called ---')

        self.replay_memory(self._history,self.action,-10,self.history,True)
        self.train_replay()
        self.global_step += 1

        with open(self.record,"a") as f:
            #Model, TimeStamp, Reward, Steps, avg_q_max, avg_loss
            f.write("{}, {}, {}, {}, {}, {}\n".format(self.model,
                int(time.time()), self.tot_reward, time.time()-self.start,
                self.avg_q_max/self.time, self.avg_loss/self.time))

        if self.model_path:
            self.model.save_weights(self.model_path)
        else:
            self.model.save_weights('model/ddqn.h5')

        if self.gif:
            imageio.mimsave(self.gif,
              [np.array(img) for i, img in enumerate(self.gifimages) if i%2 == 0],
              fps=30)


if __name__ == "__main__":
    bot = Bot()
    # bot.random_agent()
    # bot.basic_agent()
    # bot.rl_agent()
    bot.detection_area()
