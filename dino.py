from PIL import Image, ImageOps
import pyautogui
import time
import numpy as np
import random

import sys
from mss import mss
import mss.tools as tools

import asyncio

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

class Bot:
    """Bot for playing Chrome dino run game"""
    def __init__(self):
        self.area = pyautogui.locateOnScreen('dino_start.png', confidence=0.9,
            grayscale=True)
        # self.area = [631, 265, 729, 225]
        if not self.area:
            print("Game area could not be found. Please load up a new game at http://www.trex-game.skipser.com/.")
            sys.exit()
        self.restart_coords = pyautogui.center(self.area)
        self.observation_area = {"top": self.area.top + 110,
            "left": self.area.left +130, "width": 140, "height": 100}
        self.gameover_area = np.sum([list(self.area), [250,60,0,0]], axis=0)
        self.gameover_area[2] = 250 #width
        self.gameover_area[3] = 40 #height
        self.mss = mss()

    def restart(self):
        pyautogui.click(self.restart_coords)
        time.sleep(0.05)
        pyautogui.press('space')

    def duck(self):
        pyautogui.keyDown('down')
        time.sleep(0.01)

    async def jump(self):
        pyautogui.keyUp('down')
        pyautogui.keyDown('space')
        # print('press down')
        await asyncio.sleep(0.05)
        # print('press up')
        pyautogui.keyUp('space')

    def short_jump(self):
        pyautogui.keyUp('down')
        pyautogui.keyDown('space')
        pyautogui.keyUp('space')

    def walk(self):
        pyautogui.keyUp('down')
        time.sleep(0.01)

    def detection_area(self):
        """
        Area right in front of dino.
        """
        sct = self.mss.grab(self.observation_area)
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        gray_img = ImageOps.grayscale(img)
        arr = np.array(gray_img)
        return arr, img

    def mean_pixel(self, arr):
        return arr.mean()

    def check_dead(self):
        return pyautogui.locateOnScreen('gameover.png', confidence=0.9,
            region=tuple(self.gameover_area), grayscale=True, step=2)

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

    async def sync_jump(self, thresh):
        while True:
            # print('test1')
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
            await asyncio.sleep(0.5)

    def basic_agent(self):
        BA_THRESHOLD = 241
        # start = time.time()
        self.restart()

        loop = asyncio.get_event_loop()
        done, pending = loop.run_until_complete(
            asyncio.wait([asyncio.ensure_future(self.sync_is_dead()),
                asyncio.ensure_future(self.sync_jump(BA_THRESHOLD))],
                return_when=asyncio.FIRST_COMPLETED))

        for task in pending:
            # print(task)
            task.cancel()
        # print(time.time() - start)

if __name__ == "__main__":
    bot = Bot()
    # bot.random_agent()
    bot.basic_agent()
