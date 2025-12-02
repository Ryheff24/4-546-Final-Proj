import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from occupy import Kinect
import torch

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(f"env Using device: {device}")
device = torch.device("cpu")

kinect = Kinect()
calibrateframe = kinect.loadFrame("occupancypipe/frames/calibration_frame.npy", type='npy', view=False)
kinect.calibrate(calibrateframe)
video = kinect.loadVideo("occupancypipe/videos/video_9489441.npy")
frames, extent = kinect.createVideo(video, z_min_threshold=-2.1, z_max_threshold=-1)
frames = torch.from_numpy(np.stack(frames)).to(device=device, dtype=torch.float32)




class harderEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, torchMode=True):

        super().__init__()
        self.action_arr_size = 12 # actions per step.
        
        # self.kinect.videoPlayback(frames, extent=extent)
        self.start_grid = frames
        self.grid = self.start_grid[0].clone()
        self.end_grid = torch.zeros_like(self.start_grid).to(device)
        self.size = self.grid.shape
        self.steps = 0
        self.extent = extent
        self.observation_space = spaces.Box(low=0, high=1, shape=tuple(self.grid.shape), dtype=np.float32)
        self.truncated = False
        self.terminated = False
        self.action_space = spaces.MultiDiscrete([4] * self.action_arr_size)  # actions per step 
        self.goalhit = False
        self.start = (0, self.grid.shape[0]//2)  # starting point
        self.agent_pos = self.start
        self.goal = (self.grid.shape[0]-1, self.grid.shape[0]//2) # goal point
        self.grid[self.start] = 2 
        self.grid[self.goal] = 2 
        # keep a sequence of visited positions for rendering the path
        self.path_positions = [self.start]
        self.gridhistory = []
        self.torchMode = torchMode
        self.max_steps = frames.shape[0] - 1
        
    def step(self, actions):

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(device=device, dtype=torch.long)
        else:
            actions = actions.long()
        # Accept either a single action (int) or a sequence of actions
        
        reward = 0.0
        # We'll count each internal move toward the step limit
        steps_taken = 0

        # action is a fixed  array of size 400 with an int of the following:
        # 0: up, 1: down, 2: left, 3: right
        # first find the end and slice the action array
        # dont accumulate penalties for hitting walls/obstacles
        # if the goal is hit but the episode isnt ended, dont reward success
        
        # REWARD STRUCTURE
        timepen = -0.1
        wallpen = -5
        obstaclepen = -10
        goalrew = 150
        failedrew = -100
        
        wallhit = 0
        obstaclehit = 0
        # print(f"Processing {len(actions)} actions at step {self.steps}")
        for action in actions:
            
            steps_taken += 1
            
            action = action.item()
            if action == 0:  
                # up
                # new_pos = (self.agent_pos[0]-1, self.agent_pos[1])
                #up
                if self.agent_pos[0] <= 0:
                    # above is wall
                    wallhit += 1
                    continue
                elif self.grid[self.agent_pos[0]-1, self.agent_pos[1]] == 1:
                    # above is obstacle
                    obstaclehit += 1
                    continue
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0]-1, self.agent_pos[1])
                    self.grid[self.agent_pos] = 3
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    if self.agent_pos == self.goal:
                        self.goalhit = True
                        break

            elif action == 1:
                # down
                if self.agent_pos[0] >= self.grid.shape[0]-1:
                    # below is wall
                    wallhit += 1
                    continue
                elif self.grid[self.agent_pos[0]+1, self.agent_pos[1]] == 1:
                    # below is obstacle
                    obstaclehit += 1
                    continue
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0]+1, self.agent_pos[1])
                    
                    self.grid[self.agent_pos] = 3  # path
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    if self.agent_pos == self.goal:
                        self.goalhit = True
                        break
                           
            elif action == 2:
                # left                
                if self.agent_pos[1] <= 0:
                    # left is wall
                    wallhit += 1
                    continue
                elif self.grid[self.agent_pos[0], self.agent_pos[1]-1] == 1:
                    # left is obstacle
                    obstaclehit += 1
                    continue
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1]-1)
                    self.grid[self.agent_pos] = 3  # path
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    if self.agent_pos == self.goal:
                        self.goalhit = True
                        break
   
            elif action == 3:
                # right
                if self.agent_pos[1] >= self.grid.shape[1]-1:
                    # right is wall
                    wallhit += 1
                    continue
                elif self.grid[self.agent_pos[0], self.agent_pos[1]+1] == 1:
                    # right is obstacle
                    obstaclehit += 1
                    continue
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1]+1)
                    self.grid[self.agent_pos] = 3  # path
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    if self.agent_pos == self.goal:
                        self.goalhit = True
                        break


        # Ensure start and goal markers remain
        self.grid[self.start] = 2
        self.grid[self.goal] = 2
        # self.gridhistory.append(self.grid.clone())
        
        # print(f"len start gridhistory: {len(self.gridhistory)} current step: {self.steps}")
        self.end_grid[self.steps].copy_(self.grid)
        self.grid = self.start_grid[self.steps].clone()
        # print(f"shape grid: {self.grid.shape} at step {self.steps}")
        self.steps += 1 
        
        if self.goalhit:
            reward += goalrew
            print(f"Goal hit at step {self.steps}!")
            self.terminated = True

        # print(f"steps_taken: {steps_taken}, wallhit: {wallhit}, obstaclehit: {obstaclehit}")
        reward = reward + (steps_taken * timepen + wallhit * wallpen + obstaclehit * obstaclepen)
        
        if not self.terminated and self.steps >= self.max_steps:
            self.truncated = True
            # print(f"Max steps reached: {self.steps} >= {self.max_steps}")
            if not self.goalhit:
                reward += failedrew
                
        # print(f"steps_taken: {steps_taken}, total steps: {self.steps}, reward: {reward}, goalhit: {self.goalhit}, wallhit: {wallhit}, obstaclehit: {obstaclehit}, truncated: {self.truncated}, terminated: {self.terminated}")
        return self.grid.clone() if self.torchMode else self.grid.clone().cpu().numpy(), float(reward), self.terminated, self.truncated, {'goal_hit': self.goalhit, 'wall_hits': wallhit, 'obstacle_hits': obstaclehit, 'steps_taken': steps_taken}

    def reset(self, *, seed=None, options=None):
        self.grid = self.start_grid[0].clone()
        self.end_grid.zero_()
        self.steps = 0
        self.agent_pos = self.start
        self.path_positions = [self.start]
        # self.grid[self.start] = 2 
        # self.grid[self.goal] = 2 
        self.truncated = False
        self.terminated = False
        self.goalhit = False
        return self.grid.clone() if self.torchMode else self.grid.clone().cpu().numpy(), {}
    def render(self, video=False, save_path=None, block=True):
        if video:
            print(self.steps)
            kinect.videoPlayback(self.end_grid.detach().cpu().numpy(), extent=self.extent, steps=self.steps)
        else:
            kinect.printgrid(self.end_grid[1].detach().cpu().numpy(), self.extent, "random_agent_initial")



    def step_env(self):
        # ensure number of obstacles are currently on the grid
        # if less, add more at y locations
        # step obstacles positions
        pass
    
if __name__ == "__main__":
    env = harderEnv()
    # action = torch.tensor([random.randint(0, 3) for _ in range(0, 499)], device=device, dtype=torch.long)  # random actions + explicit end
    action = torch.tensor([ 1 for _ in range(0, 499)], device=device, dtype=torch.long)
    for ep in range(1):
        obs, _ = env.reset()
        
        total_reward = 0.0
        terminated = False
        truncated = False
        x = 0
        while True:

            actions = action[x: x+env.action_arr_size]
            # action = torch.tensor([ 1 for _ in range(0, 192)] + [4], device=device, dtype=torch.long)
            # print(len(action))
            obs, reward, terminated, truncated, info = env.step(actions)
            x += env.action_arr_size    
            
            total_reward += reward

            if truncated or terminated:
                print(f"Reward: {reward}, Total Reward: {total_reward}, obstacle hits: {info['obstacle_hits']}, wall hits: {info['wall_hits']}, goal hit: {info['goal_hit']}, steps taken: {info['steps_taken']}, terminated: {terminated}, truncated: {truncated}")
                
                break
        # kinect.printgrid(obs.detach().cpu().numpy(), env.extent, "hardenv_final")
    env.render(video=True)
