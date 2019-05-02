# Self Driving Car Map

# Importing the Libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceLineProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our ai.py

# Adding this line if we don't want the right click to put a red line
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory and draw the sandbox
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AS, which we call brain, and that contains our NN
brain = Dqn(5, 3, 0.9)
action2rotation = [0, 20, -20]
last_reward = 0
scores = []


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeroes((longueur, largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False


# Initializing the last distance
last_distance = 0
