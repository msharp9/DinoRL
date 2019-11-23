from PIL import Image, ImageOps
import pyautogui
import time
import numpy as np
import random

import sys
from mss import mss
import mss.tools as tools

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

class Bot:
    """Bot for playing Chrome dino run game"""
    def __init__(self):
        self.area = pyautogui.locateOnScreen('dino_start.png', confidence=0.9, grayscale=True)
        if not self.area:
            print("Game area could not be found. Please load up a new game at http://www.trex-game.skipser.com/.")
            sys.exit()
        self.restart_coords = pyautogui.center(self.area)
        self.observation_area = {"top": self.area.top + 110,
            "left": self.area.left +130, "width": 140, "height": 100}
        self.mss = mss()

    def restart(self):
        pyautogui.click(self.restart_coords)
        time.sleep(0.05)
        pyautogui.press('space')

    def duck(self):
        pyautogui.keyDown('down')
        time.sleep(0.01)

    def jump(self):
        pyautogui.keyUp('down')
        pyautogui.press('space')
        # press is clean but is affected by the PAUSE attribute
        # pyautogui.keyDown('space')
        # time.sleep(0.05)
        # pyautogui.keyUp('space')

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
        # img = pyautogui.screenshot(region=self.observation_area)
        sct = self.mss.grab(self.observation_area)
        # tools.to_png(sct.rgb, sct.size, output='test.png')
        img = Image.frombytes("RGB", sct.size, sct.bgra, "raw", "BGRX")
        gray_img = ImageOps.grayscale(img)
        arr = np.array(gray_img)
        return arr

    def mean_pixel(self, arr):
        return arr.mean()

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

    def basic_agent(self):
        BA_THRESHOLD = 241
        # start = time.time()
        self.restart()
        while True:
            # print(time.time() - start)
            if self.mean_pixel(self.detection_area()) < BA_THRESHOLD:
                self.jump()
                # self.short_jump()

if __name__ == "__main__":
    bot = Bot()
    # bot.random_agent()
    bot.basic_agent()
