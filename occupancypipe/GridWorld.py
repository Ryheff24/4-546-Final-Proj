from math import e
import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from occupy import loadSavedFrame, printgrid, denoise
class SimpleEnv(gym.Env):

    def __init__(self, filename, max_steps=1000):
        self.start_grid, self.extent = denoise(loadSavedFrame(filename))
        self.size = self.start_grid.shape
        self.grid = self.start_grid.copy()
        self.max_steps = min(max_steps, 1000)
        self.current_step = 0
        self.extent = self.extent
        self.observation_space = spaces.Box(low=0, high=1, shape=self.start_grid.shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  #
        self.start = (0, self.grid.shape[0]//2)  # starting point
        self.grid[self.start] = 2 
        self.agent_pos = self.start
        self.goal = (self.grid.shape[0]-1, self.start_grid.shape[0]//2) # goal point
        self.grid[self.goal] = 2 
        
        # self.path = np.zeros_like(self.grid)
        # printgrid(self.grid, self.extent, "GridWorldEnv Initialized")
        # print(self.grid.shape)
        
    def step(self, action):
        reward = 0
        self.current_step += 1
  # small negative reward for each step to encourage shorter paths
        truncated = False
        terminated = False
        timepen = -1
        wallpen = -5
        obstaclepen = -10
        goalrew = 100
        reward += timepen
        # validate action
        # 0: up, 1: down, 2: left, 3: right

        if action == 0:
            # new_pos = (self.agent_pos[0]-1, self.agent_pos[1])
            #up
            if self.agent_pos[0] <= 0:
                # above is wall
                reward += wallpen
            elif self.grid[self.agent_pos[0]-1, self.agent_pos[1]] == 1:
                # above is obstacle
                reward += obstaclepen
            else:
                # valid move
                self.agent_pos = (self.agent_pos[0]-1, self.agent_pos[1])
                self.grid[self.agent_pos] = 3  # path
                if self.agent_pos == self.goal:
                    # if this move lands in the goal
                    reward += goalrew
                    terminated = True
        elif action == 1:
            #down 
            if self.agent_pos[0]>= self.grid.shape[0]-1:
                # below is wall
                reward += wallpen
            elif self.grid[self.agent_pos[0]+1, self.agent_pos[1]] == 1:
                # below is obstacle
                reward += obstaclepen
            else:
                # valid move
                self.agent_pos = (self.agent_pos[0]+1, self.agent_pos[1])
                self.grid[self.agent_pos] = 3  # path
                if self.agent_pos == self.goal:
                    # if this move lands in the goal
                    reward += goalrew
                    terminated = True
                    
        elif action == 2:
            #left
            if self.agent_pos[1]<= 0:
                # left is wall
                reward += wallpen
            elif self.grid[self.agent_pos[0], self.agent_pos[1]-1] == 1:
                # left is obstacle
                reward += obstaclepen
            else:
                # valid move
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1]-1)
                self.grid[self.agent_pos] = 3  # path
                if self.agent_pos == self.goal:
                    # if this move lands in the goal
                    reward += goalrew
                    terminated = True
                    
        elif action == 3:
            #right
            if self.agent_pos[1]>= self.grid.shape[1]-1:
                # right is wall
                reward += wallpen
            elif self.grid[self.agent_pos[0], self.agent_pos[1]+1] == 1:
                # right is obstacle
                reward += obstaclepen
            else:
                # valid move
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1]+1)
                self.grid[self.agent_pos] = 3  # path
                if self.agent_pos == self.goal:
                    # if this move lands in the goal
                    reward += goalrew
                    terminated = True
            
        
        if self.current_step >= self.max_steps:
            truncated = True
        return self.grid, reward, terminated, truncated, {}
        
    def reset(self):
        self.grid = self.start_grid.copy()
        self.current_step = 0
        self.agent_pos = self.start
        self.grid[self.start] = 2 
        self.grid[self.goal] = 2 
        return self.grid, 0,  False, False, {}

    def render(self):
        printgrid(self.grid, self.extent, "GridWorldEnv Render")
    
    
    
class harderEnv():
    
    def __init__(self, filename, max_steps=1000):
        self.start_grid, self.extent = denoise(loadSavedFrame(filename))
        self.size = self.start_grid.shape
        self.grid = self.start_grid.copy()
        self.max_steps = min(max_steps, 1000)
        self.current_step = 0
        self.extent = self.extent
        self.observation_space = spaces.Box(low=0, high=1, shape=self.start_grid.shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  #
        self.start = (0, self.grid.shape[0]//2)  # starting point
        self.grid[self.start] = 2 
        self.agent_pos = self.start
        self.goal = (self.grid.shape[0]-1, self.start_grid.shape[0]//2) # goal point
        self.grid[self.goal] = 2 
        
        # self.path = np.zeros_like(self.grid)
        # printgrid(self.grid, self.extent, "GridWorldEnv Initialized")
        # print(self.grid.shape)
        
    def step(self, action):
        reward = 0
        self.current_step += 1
  # small negative reward for each step to encourage shorter paths
        truncated = False
        terminated = False
        timepen = -1
        wallpen = -5
        obstaclepen = -10
        goalrew = 100
        reward += timepen
        # validate action
        # 0: up, 1: down, 2: left, 3: right

        if action == 0:
            # new_pos = (self.agent_pos[0]-1, self.agent_pos[1])
            #up
            if self.agent_pos[0] <= 0:
                # above is wall
                reward += wallpen
            elif self.grid[self.agent_pos[0]-1, self.agent_pos[1]] == 1:
                # above is obstacle
                reward += obstaclepen
            else:
                # valid move
                self.agent_pos = (self.agent_pos[0]-1, self.agent_pos[1])
                self.grid[self.agent_pos] = 3  # path
                if self.agent_pos == self.goal:
                    # if this move lands in the goal
                    reward += goalrew
                    terminated = True
        elif action == 1:
            #down 
            if self.agent_pos[0]>= self.grid.shape[0]-1:
                # below is wall
                reward += wallpen
            elif self.grid[self.agent_pos[0]+1, self.agent_pos[1]] == 1:
                # below is obstacle
                reward += obstaclepen
            else:
                # valid move
                self.agent_pos = (self.agent_pos[0]+1, self.agent_pos[1])
                self.grid[self.agent_pos] = 3  # path
                if self.agent_pos == self.goal:
                    # if this move lands in the goal
                    reward += goalrew
                    terminated = True
                    
        elif action == 2:
            #left
            if self.agent_pos[1]<= 0:
                # left is wall
                reward += wallpen
            elif self.grid[self.agent_pos[0], self.agent_pos[1]-1] == 1:
                # left is obstacle
                reward += obstaclepen
            else:
                # valid move
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1]-1)
                self.grid[self.agent_pos] = 3  # path
                if self.agent_pos == self.goal:
                    # if this move lands in the goal
                    reward += goalrew
                    terminated = True
                    
        elif action == 3:
            #right
            if self.agent_pos[1]>= self.grid.shape[1]-1:
                # right is wall
                reward += wallpen
            elif self.grid[self.agent_pos[0], self.agent_pos[1]+1] == 1:
                # right is obstacle
                reward += obstaclepen
            else:
                # valid move
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1]+1)
                self.grid[self.agent_pos] = 3  # path
                if self.agent_pos == self.goal:
                    # if this move lands in the goal
                    reward += goalrew
                    terminated = True
            
        
        if self.current_step >= self.max_steps:
            truncated = True
        return self.grid, reward, terminated, truncated, {}
        
    def reset(self):
        self.grid = self.start_grid.copy()
        self.current_step = 0
        self.agent_pos = self.start
        self.grid[self.start] = 2 
        self.grid[self.goal] = 2 
        return self.grid, 0,  False, False, {}

    def render(self):
        printgrid(self.grid, self.extent, "GridWorldEnv Render")
    
    
# if __name__ == "__main__":
#     env = GridWorldEnv("person.txt")
#     reward = 0
        
#     for _ in range(0, env.size[0]-1):
#         obs, rew, _, _, _ = env.step(3)
#         reward += rew
        
#     print(f"Total reward: {reward}")
#     env.render()