from PIL import ImageOps
import pyautogui
import time
import numpy as np
import random

class Bot:
    """Bot for playing Chrome dino run game"""
    def __init__(self):
        self.area = pyautogui.locateOnScreen('dino_start.png', confidence=0.9)
        self.restart_coords = pyautogui.center(self.area)

    def restart(self):
        pyautogui.click(self.restart_coords)

    def duck(self):
        pyautogui.keyDown('down')
        time.sleep(0.01)

    def jump(self):
        pyautogui.keyUp('down')
        pyautogui.keyDown('space')
        time.sleep(0.05)
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
        Checks the area to have obstacles
        :return: float
        """
        image = pyautogui.screenshot(region=self.area)
        # print(self.area)
        # image.show()
        gray_img = ImageOps.grayscale(image)
        arr = np.array(gray_img.getcolors())
        print(arr.mean())
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

    def

if __name__ == "__main__":
    bot = Bot()
    bot.random_agent()
