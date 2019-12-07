#!/usr/bin/python3

from PIL import Image, ImageOps
import pyautogui
import time
import numpy as np
import random

import sys
import os
from mss import mss
import mss.tools as tools

import asyncio
import imageio
import cv2

from collections import deque
import keras
import tensorflow as tf

from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6

# HEADLESS = False

class DDQN_brain():
    memory = deque(maxlen=30000)
    global_step = 0
    height = 100
    width = 300
    state_size = (height, width, 4)
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
        self.exploration_steps = 10000.
        if explore:
            self.epsilon_decay_step = (self.epsilon - self.epsilon_end) \
                                  / self.exploration_steps
        else:
            self.epsilon_decay_step = 0

        # parameters about training
        self.batch_size = 128
        self.train_start = 1000
        self.update_target_rate = 5000
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
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
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
            try:
                history[i] = np.float32(mini_batch[i][0] / 255.)
                next_history[i] = np.float32(mini_batch[i][3] / 255.)
                action.append(mini_batch[i][1])
                reward.append(mini_batch[i][2])
                dead.append(mini_batch[i][4])
            except Exception as e:
                print("Training Failed!")
                print(i, mini_batch[i])
                print(i-1, mini_batch[i-1])
                print(str(e))
                sys.exit()

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
                target[i] = reward[i] + self.gamma * \
                    target_value[i][np.argmax(value[i])]

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]


class Bot(DDQN_brain):
    """Bot for playing Chrome dino run game"""
    def __init__(self, action_space=None, **kwargs):
        super().__init__(**kwargs)

        self.area = pyautogui.locateOnScreen('dino_start.png', confidence=0.9,
            grayscale=True)
        # self.area = [631, 265, 729, 225]
        if not self.area:
            print("Game area could not be found. Please load up a new game at \
http://www.trex-game.skipser.com/.")
            sys.exit()
        self.restart_coords = pyautogui.center(self.area)
        self.area_display = {"top": self.area.top, "left": self.area.left,
            "width": self.area.width, "height": self.area.height}
        self.observation_area = {"top": self.area.top + 80,
            "left": self.area.left + 80,
            "width": self.width, "height": self.height}
        self.gameover_area = {"top": self.area.top + 60,
            "left": self.area.left + 250, "width": 250, "height": 40}
        self.mss = mss()

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
        await asyncio.sleep(0.02)

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
        await asyncio.sleep(0.02)

    def detection_area(self):
        """
        Area right in front of dino.
        """
        sct = self.mss.grab(self.observation_area)
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        gray_img = ImageOps.grayscale(img)
        arr = np.array(gray_img)
        return arr, img

    def play_area(self):
        sct = self.mss.grab(self.area_display)
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        return img

    def save_gif(self, img, store):
        gifpic = img.copy()
        gifpic = cv2.cvtColor(np.array(gifpic), cv2.COLOR_RGB2BGR)
        if self.action is not None:
            cv2.putText(gifpic, self.choicestext[self.action], (0, 20),
                cv2.FONT_HERSHEY_PLAIN, 0.8, (0,255,0), 1, cv2.LINE_AA)
        store.append(gifpic)

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
                # await self.duck()
                # await self.short_jump()
                # await self.walk()
                # print("mpixel: {}".format(mpixel))
                # img.show()
            await asyncio.sleep(0.01)

    async def sync_is_dead(self):
        dead = None
        while not dead:
            dead = self.check_dead()
            await asyncio.sleep(0.01)

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

    def rl_agent(self, gif=False, grad_cam=False, replay=False, headless=False):
        self.start = time.time()
        self.time = self.start
        self.steps = 0

        self.history = None
        self.action = None
        self.reward = 0
        self.tot_reward = 0
        self.avg_q_max, self.avg_loss = 0, 0

        self.gif = gif
        self.gifimages = []
        self.grad_cam = grad_cam
        self.gc_images = []

        self.replay = replay
        self.replay_data = []
        self.headless = headless

        self.restart()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        done, pending = loop.run_until_complete(
            asyncio.wait([asyncio.ensure_future(self.sync_is_dead()),
                asyncio.ensure_future(self.take_steps())],
                return_when=asyncio.FIRST_COMPLETED))
        for task in pending:
            task.cancel()
        loop.stop()
        self.on_end()


    async def take_steps(self):
        while True:
            try:
                await self.on_step()
                await asyncio.sleep(0.001)
            except Exception as e:
                print("Exception raised. Failed to run step.")
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

    async def display(self, area):
        sct = self.mss.grab(area)
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if self.action is not None:
            cv2.putText(img, self.choicestext[self.action], (0, 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2, cv2.LINE_AA)
            # if self.grad_cam:
            #     heatmap = self.grad_cam_heatmap(self.action, self.history)
            #     resized = self.merge_heatmap(resized, heatmap)
        cv2.imshow("Dino Bot", img)
        cv2.waitKey(1)
        # await asyncio.sleep(0.01)

    async def on_step(self):
        _observation, _img = self.detection_area()
        if self.gif == "AI":
            self.save_gif(_img, self.gifimages)
        elif self.gif:
            self.save_gif(self.play_area(), self.gifimages)

        if self.grad_cam:
            self.save_gif(_img, self.gc_images)

        _time = time.time()
        if self.action is not None:
            _observation = np.reshape([_observation],
                (1, self.height, self.width, 1))
            _history = np.append(_observation, self.history[:,:,:,:3], axis=3)
            # self.reward = _time - self.time
            self.replay_memory(self.history, self.action,
                self.reward, _history, False)
            # self.train_replay()
            # print(self.reward)
            self.tot_reward += self.reward
            self.global_step += 1
            if self.global_step % self.update_target_rate == 0:
                self.update_target_model()
        else:
            _history = np.stack((_observation, _observation,
                _observation, _observation), axis=2)
            _history = np.reshape([_history], (1, self.height, self.width, 4))
        self.time = _time
        self._history = self.history
        self.history = _history
        self.action = self.choose_action(_history)

        if self.replay or self.grad_cam:
            # y = np.zeros(4)
            # y[self.action] = 1
            self.replay_data.append([self.action, _history])

        self.avg_q_max += np.amax(self.model.predict(
            np.float32(self.history/255.))[0])
        self.steps += 1

        try:
            # print(self.action)
            await self.choices[self.action]()
            if self.headless == "AI":
                await self.display(self.observation_area)
            elif self.headless:
                await self.display(self.area_display)
        except Exception as e:
            print("Exception raised. Failed to run choices.")
            print(str(e))
            pass

    def on_end(self):
        print('--- on_end called ---')
        pyautogui.keyUp('down') #unpress keys
        pyautogui.keyUp('space')

        if self._history is not None:
            self.replay_memory(self._history,self.action,-10,self.history,True)
            self.train_replay()
            for _ in range(self.steps):
                self.train_replay()
            self.global_step += 1

        with open(self.record,"a") as f:
            #Model, TimeStamp, Reward, Game_time, Steps, avg_q_max, avg_loss
            f.write("{}, {}, {}, {}, {}, {}, {}\n".format(self.model,
                int(time.time()), self.tot_reward, time.time()-self.start,
                self.steps, self.avg_q_max/self.steps, self.avg_loss/self.steps))

        if self.model_path:
            self.model.save_weights(self.model_path)
        else:
            self.model.save_weights('model/ddqn.h5')

        if self.gif:
            imageio.mimsave("gifs/{}.gif".format(int(time.time())),
              # [np.array(img) for i, img in enumerate(self.gifimages) if i%2==0],
              [np.array(img) for img in self.gifimages], fps=20)

        if self.replay:
            np.save("replays/{}.npy".format(int(time.time())),
                np.array(self.replay_data))

        if self.grad_cam:
            gcam_pics = []
            for [action, history], gif in zip(self.replay_data, self.gc_images):
                heatmap = self.grad_cam_heatmap(action, history)
                gcam_pic = self.merge_heatmap(gif, heatmap)
                gcam_pics.append(gcam_pic)
            imageio.mimsave("gifs/gcam_{}.gif".format(int(time.time())),
                [np.array(img) for img in gcam_pics], fps=20)


if __name__ == "__main__":
    # bot = Bot()
    # bot.random_agent()
    # bot.basic_agent()

    # bot = Bot(explore=True)
    bot= Bot()
    bot.rl_agent(gif=True,grad_cam=True)
    for episode in range(1000):
        print('Episode: '+str(episode))
        bot.rl_agent(replay=True)
        # bot.rl_agent(gif=True,grad_cam=True,replay=True)
        ga = bot.gameover_area
        ga = [ga["left"], ga["top"], ga["width"], ga["height"]]
        while not pyautogui.locateOnScreen('gameover.png', confidence=0.9,
            region=tuple(ga), grayscale=True, step=2):
            time.sleep(1)
