from math import e
import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random

def printgrid(occupancy_grid, extent, name):
    plt.figure(figsize=(10, 10))
    plt.imshow(occupancy_grid, cmap='binary', origin='upper', extent=extent)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('2D Occupancy Grid(Black is Occupied)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.show()


def denoise(occupancy_grid):
    # this needs to be more efficient if its live
    # for every point, if none of its neighbors are occupied set it to unoccupied
    for x in range(1, occupancy_grid.shape[0]-1):
        for y in range(1, occupancy_grid.shape[1]-1):
            if occupancy_grid[x, y] == 1:
                z = np.sum(occupancy_grid[x-1:x+2, y-1:y+2])
                if z <= 1:
                    occupancy_grid[x, y] = 0
    return occupancy_grid

class harderEnv():
    
    def __init__(self, filename, max_steps=1000, obstacles=5):
        self.start_grid = np.zeros((200, 200), dtype=np.uint8)
        self.size = self.start_grid.shape
        self.grid = self.start_grid.copy()
        self.max_steps = min(max_steps, 1000)
        self.current_step = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=self.start_grid.shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)  #
        self.start = (0, self.grid.shape[0]//2)  # starting point
        self.grid[self.start] = 2 
        self.agent_pos = self.start
        self.goal = (self.grid.shape[0]-1, self.start_grid.shape[0]//2) # goal point
        self.grid[self.goal] = 2 
        # (x,y, speed,(x_size, y_size))
        for i in range(obstacles):
            size = (random.randint(2, 20), random.randint(2, 20))
            x, y = (10 , 5)
            speed = random.randint(1, 5)
            
        self.obstacles = [(random.randint(0, self.grid.shape[0]), self.start_grid.shape[1]+i ,random.randint(1, 5), (random.randint(2, 20), random.randint(2, 20))) for i in range(obstacles)]
        self.spawninterval = 10
        print(self.obstacles)
        
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
    
    



    def step_env(self):
        # ensure number of obstacles are currently on the grid
        # if less, add more at y locations
        # step obstacles positions
        pass
    
if __name__ == "__main__":
    env = harderEnv("person.npy", max_steps=500, obstacles=10)