from math import e
import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
from occupy import Kinect

class harderEnv():
    
    def __init__(self, filename, max_steps=1000, obstacles=5):
        self.kinect = Kinect()
        frame = self.kinect.loadFrame("occupancypipe/frames/calibration_frame.npy", type='npy', view=False)
        self.kinect.calibrate(frame)
        video = self.kinect.loadVideo("occupancypipe/videos/video_9489441.npy")
        frames, extent = self.kinect.createVideo(video)
        # self.kinect.videoPlayback(frames, extent=extent)
        
        self.start_grid = frames[0]
        self.size = self.start_grid.shape
        self.grid = self.start_grid.copy()
        self.max_steps = min(max_steps, 1000)
        self.current_step = 0
        self.extent = extent
        self.observation_space = spaces.Box(low=0, high=1, shape=self.start_grid.shape, dtype=np.uint8)
        
        self.action_space = spaces.Discrete(4)  #
        
        self.start = (0, self.grid.shape[0]//2)  # starting point
        self.agent_pos = self.start
        self.goal = (self.grid.shape[0]-1, self.start_grid.shape[0]//2) # goal point
        self.grid[self.start] = 2 
        self.grid[self.goal] = 2 
        self.action_arr_size = 400
        
    def step(self, actions):
        reward = 0
        self.current_step += 1
        # [3 ,3 ,3 2, 1, 0, 0, 4, 4, 4, 4,...]
        # action is a fixed  array of size 400 with an int of the following:
        # 0: up, 1: down, 2: left, 3: right, 4: end
        # first find the end and slice the action array
        # episode only ends when 4 is hit or max steps reached
        # dont accumulate penalties for hitting walls/obstacles
        # if the goal is hit but the episode isnt ended, dont reward success
        # 

        for i, act in enumerate(actions):
            if act == 4:
                actions = actions[:i]
                break
  
        truncated = False
        terminated = False
        # REWARD STRUCTURE
        timepen = -0.5
        wallpen = -5
        obstaclepen = -10
        goalrew = 100
        failedrew = -50
        
        goalhit = False
        wallhit = False
        obstaclehit = False
        
        for action in action:
            if action == 0: # up
                # new_pos = (self.agent_pos[0]-1, self.agent_pos[1])
                #up
                if self.agent_pos[0] <= 0:
                    # above is wall
                    wallhit = True
                elif self.grid[self.agent_pos[0]-1, self.agent_pos[1]] == 1:
                    # above is obstacle
                    obstaclehit = True
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0]-1, self.agent_pos[1])
                    if self.agent_pos == self.goal:
                        # if this move lands in the goal
                        goalhit = True
                        # terminated = True

            elif action == 1:
                if self.agent_pos[0]>= self.grid.shape[0]-1:
                    # below is wall
                    wallhit = True
                elif self.grid[self.agent_pos[0]+1, self.agent_pos[1]] == 1:
                    # below is obstacle
                    obstaclehit = True
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0]+1, self.agent_pos[1])
                    self.grid[self.agent_pos] = 3  # path
                    if self.agent_pos == self.goal:
                        # if this move lands in the goal
                        goalhit = True
                        
            elif action == 2:
                #left
                if self.agent_pos[1]<= 0:
                    # left is wall
                    wallhit = True
                elif self.grid[self.agent_pos[0], self.agent_pos[1]-1] == 1:
                    # left is obstacle
                    obstaclehit = True
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1]-1)
                    self.grid[self.agent_pos] = 3  # path
                    if self.agent_pos == self.goal:
                        # if this move lands in the goal
                        goalhit = True
                        
            elif action == 3:
                #right
                if self.agent_pos[1]>= self.grid.shape[1]-1:
                    # right is wall
                    wallhit = True
                elif self.grid[self.agent_pos[0], self.agent_pos[1]+1] == 1:
                    # right is obstacle
                    obstaclehit = True
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1]+1)
                    self.grid[self.agent_pos] = 3  # path
                    if self.agent_pos == self.goal:
                        # if this move lands in the goal
                        goalhit = True
            elif action == 4:
                # goal is hit
                pass
            
        if self.current_step >= self.max_steps:
            truncated = True
        # grid needs to be nexts frame from kinect
        # 
        return self.grid, reward, terminated, truncated, {}
        
    def reset(self):
        self.grid = self.start_grid.copy()
        self.current_step = 0
        self.agent_pos = self.start
        self.grid[self.start] = 2 
        self.grid[self.goal] = 2 
        return self.grid, 0,  False, False, {}

    def render(self):
        self.kinect.printgrid(self.grid, self.extent, "GridWorldEnv Render")
    
    



    def step_env(self):
        # ensure number of obstacles are currently on the grid
        # if less, add more at y locations
        # step obstacles positions
        pass
    
if __name__ == "__main__":
    env = harderEnv("person.npy", max_steps=500, obstacles=10)