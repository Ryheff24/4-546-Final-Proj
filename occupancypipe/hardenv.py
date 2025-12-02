from math import e
import re
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import os
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
        # keep a sequence of visited positions for rendering the path
        self.path_positions = [self.start]
        
    def step(self, actions):
        # Accept either a single action (int) or a sequence of actions
        if isinstance(actions, (int, np.integer)):
            actions = [int(actions)]
        else:
            actions = list(map(int, actions))

        reward = 0.0
        # We'll count each internal move toward the step limit
        steps_taken = 0
        # [3 ,3 ,3 2, 1, 0, 0, 4, 4, 4, 4,...]
        # action is a fixed  array of size 400 with an int of the following:
        # 0: up, 1: down, 2: left, 3: right, 4: end
        # first find the end and slice the action array
        # episode only ends when 4 is hit or max steps reached
        # dont accumulate penalties for hitting walls/obstacles
        # if the goal is hit but the episode isnt ended, dont reward success
        

        # Trim at first explicit end token (4)
        for i, act in enumerate(actions):
            if act == 4:
                actions = actions[:i+1]
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

        for action in actions:
            if action == 0:  # up
                # new_pos = (self.agent_pos[0]-1, self.agent_pos[1])
                #up
                if self.agent_pos[0] <= 0:
                    # above is wall
                    wallhit = True
                    reward += wallpen
                    steps_taken += 1
                    continue
                elif self.grid[self.agent_pos[0]-1, self.agent_pos[1]] == 1:
                    # above is obstacle
                    obstaclehit = True
                    reward += obstaclepen
                    steps_taken += 1
                    continue
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0]-1, self.agent_pos[1])
                    self.grid[self.agent_pos] = 3
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    reward += timepen
                    steps_taken += 1
                    if self.agent_pos == self.goal:
                        # landed on goal (mark but don't grant goal reward until explicit end)
                        goalhit = True

            elif action == 1:
                if self.agent_pos[0] >= self.grid.shape[0]-1:
                    # below is wall
                    wallhit = True
                    reward += wallpen
                    steps_taken += 1
                    continue
                elif self.grid[self.agent_pos[0]+1, self.agent_pos[1]] == 1:
                    # below is obstacle
                    obstaclehit = True
                    reward += obstaclepen
                    steps_taken += 1
                    continue
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0]+1, self.agent_pos[1])
                    self.grid[self.agent_pos] = 3  # path
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    reward += timepen
                    steps_taken += 1
                    if self.agent_pos == self.goal:
                        goalhit = True
                        
            elif action == 2:
                # left
                if self.agent_pos[1] <= 0:
                    # left is wall
                    wallhit = True
                    reward += wallpen
                    steps_taken += 1
                    continue
                elif self.grid[self.agent_pos[0], self.agent_pos[1]-1] == 1:
                    # left is obstacle
                    obstaclehit = True
                    reward += obstaclepen
                    steps_taken += 1
                    continue
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1]-1)
                    self.grid[self.agent_pos] = 3  # path
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    reward += timepen
                    steps_taken += 1
                    if self.agent_pos == self.goal:
                        goalhit = True
                        
            elif action == 3:
                # right
                if self.agent_pos[1] >= self.grid.shape[1]-1:
                    # right is wall
                    wallhit = True
                    reward += wallpen
                    steps_taken += 1
                    continue
                elif self.grid[self.agent_pos[0], self.agent_pos[1]+1] == 1:
                    # right is obstacle
                    obstaclehit = True
                    reward += obstaclepen
                    steps_taken += 1
                    continue
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1]+1)
                    self.grid[self.agent_pos] = 3  # path
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    reward += timepen
                    steps_taken += 1
                    if self.agent_pos == self.goal:
                        goalhit = True
            elif action == 4:
                reward += timepen
                steps_taken += 1
                break
            else:
                # unknown action: ignore but count it
                steps_taken += 1
                reward += timepen
                continue
            
        self.current_step += steps_taken
        if self.current_step >= self.max_steps:
            truncated = True
            # if max steps reached and not terminated, apply failure penalty
            if not terminated:
                reward += failedrew
        # grid needs to be nexts frame from kinect
        # 
        # Ensure start and goal markers remain
        self.grid[self.start] = 2
        self.grid[self.goal] = 2

        return self.grid.copy(), float(reward), bool(terminated), bool(truncated), {}
        
    def reset(self):
        self.grid = self.start_grid.copy()
        self.current_step = 0
        self.agent_pos = self.start
        self.path_positions = [self.start]
        self.grid[self.start] = 2 
        self.grid[self.goal] = 2 
        return self.grid, 0,  False, False, {}

    def render(self, save_path=None, block=True):
        print(f"[hardenv.render] save_path={save_path!r}, block={block}")
        # Draw the grid and overlay the agent path as a line from start to current position
        try:
            x_min, x_max, y_min, y_max = self.extent
        except Exception:
            # fallback to simple imshow if extent is unavailable
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(self.grid, cmap='binary', origin='upper')
            plt.title('GridWorldEnv Render')
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close(fig)
                return
            else:
                plt.show(block=block)
                return

        fig, ax = plt.subplots(figsize=(8, 8))
        img = ax.imshow(self.grid, cmap='binary', origin='upper', extent=self.extent)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('2D Occupancy Grid with Agent Path')

        # If there is a recorded path, convert grid indices to world coordinates and plot a line
        if len(self.path_positions) >= 2:
            grid_h, grid_w = self.grid.shape
            x_min, x_max, y_min, y_max = self.extent
            # cell width/height
            cell_w = (x_max - x_min) / grid_w
            cell_h = (y_max - y_min) / grid_h
            xs = []
            ys = []
            for (r, c) in self.path_positions:
                # map center of cell to world coords
                x = x_min + (c + 0.5) * cell_w
                y = y_min + (r + 0.5) * cell_h
                xs.append(x)
                ys.append(y)
            ax.plot(xs, ys, color='red', linewidth=2, marker='o', markersize=4)

        # Draw start and goal markers
        sr, sc = self.start
        gr, gc = self.goal
        cell_w = (x_max - x_min) / self.grid.shape[1]
        cell_h = (y_max - y_min) / self.grid.shape[0]
        sx = x_min + (sc + 0.5) * cell_w
        sy = y_min + (sr + 0.5) * cell_h
        gx = x_min + (gc + 0.5) * cell_w
        gy = y_min + (gr + 0.5) * cell_h
        ax.scatter([sx], [sy], c='green', s=80, label='start')
        ax.scatter([gx], [gy], c='gold', s=80, label='goal')
        ax.legend()
        plt.tight_layout()

        if save_path:
            import os as _os
            _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)
            plt.close(fig)
            return
        else:
            plt.show(block=block)
            return
    
    



    def step_env(self):
        # ensure number of obstacles are currently on the grid
        # if less, add more at y locations
        # step obstacles positions
        pass
    
if __name__ == "__main__":
    env = harderEnv("person.npy", max_steps=500, obstacles=10)