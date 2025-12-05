import json
import multiprocessing
from flask.cli import F
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import torch
from collections import deque
from multiprocessing import Pool

def compute_single_frame_distances(args):
    i, frame_np, goal, height, width = args
    frame = torch.from_numpy(frame_np)
    visited = torch.full(frame.shape, -1, dtype=torch.float32)
    queue = deque([goal])
    visited[goal] = 0
    while queue:
        cur_x, cur_y = queue.popleft()
        for next_x, next_y in [(cur_x-1, cur_y), (cur_x+1, cur_y), (cur_x, cur_y-1), (cur_x, cur_y+1)]:
            if next_x < 0 or next_x >= height:
                continue
            if next_y < 0 or next_y >= width:
                continue
            if visited[next_x, next_y] != -1:
                continue
            if frame[next_x, next_y] == 1:
                continue
            visited[next_x, next_y] = visited[cur_x, cur_y] + 1
            queue.append((next_x, next_y))
    return i, visited



def load_frames(device, default=True, duration=5, fps=5, count=4):
    if default:
        frames = np.load("occupancypipe/frames/processed_frames.npy")
        extent = np.load("occupancypipe/frames/extent.npy")
    else:
        frames = np.load(f"occupancypipe/frames/processed_frames_{duration}sec{fps}fps{count}.npy")
        extent = np.load(f"occupancypipe/frames/extent_{duration}sec{fps}fps{count}.npy")
    # frames = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
    return frames.astype(np.float32), extent
# THE DEFAULT FRAME IS BROKEN DO NOT USE.
class harderEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, torchMode=True, cnn=False, default=True, duration=5, fps=5, count=4, 
                  cores=multiprocessing.cpu_count()-1):
        if default:
            distFile="occupancypipe/hardenv_distance_map.pt"
        else:
            distFile= f"occupancypipe/hardenv_distance_map_{duration}s_{fps}fps_ct{count}.pt"
            
        self.cnn = True
        self.stack = 4
        self.frame_stack = deque(maxlen=self.stack)
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        # elif torch.cuda.is_available():
        #     device = torch.device("cuda")
        # else:
        device = torch.device("cpu")
        self.device = device
        super().__init__()
        self.frames, self.extent = load_frames(self.device, default=default, duration=duration, fps=fps, count=count)
        self.action_arr_size = 12 # actions per step.
        
        # self.kinect.videoPlayback(frames, extent=extent)
        self.start_grid = self.frames
        self.grid = self.start_grid[0].copy()
        self.end_grid = np.zeros_like(self.start_grid)
        self.size = self.grid.shape
        self.steps = 0
        self.extent = self.extent
        self.truncated = False
        self.terminated = False
        self.action_space = spaces.MultiDiscrete([4] * self.action_arr_size)  # actions per step 
        self.goalhit = False
        self.start = (0, self.grid.shape[1]//2)  # starting point - top center (row 0, middle column)
        self.agent_pos = self.start
        self.goal = (self.grid.shape[0]-1, self.grid.shape[1]//2)  # goal point - bottom center (last row, middle column)
        self.grid[self.start] = 3
        self.grid[self.goal] = 2 
        if self.cnn:
            self.observation_space = spaces.Box(low=0, high=1, 
                shape=(self.stack, self.grid.shape[0], self.grid.shape[1]), 
                dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=tuple(self.grid.shape), dtype=np.float32)
            
        
        self.path_positions = [self.start]
        self.gridhistory = []
        self.torchMode = torchMode
        self.max_steps = self.frames.shape[0] - 1
        self.distances = self.precompute_distances(cores=cores, filename=distFile)
        self.prev_distance = self.distances[0, self.start[0], self.start[1]]
    
    def precompute_distances(self, cores, filename, load=True):
        """compute distance, this is multhreaded based on number of cores"""
        if load:
            if os.path.exists(filename):
                # print(f"Loading precomputed distance map from {filename}")
                file = torch.load(filename, map_location=self.device)
                if file.shape != self.frames.shape:
                    print(f"Distance map shape {file.shape} does not match frames shape {self.frames.shape}. Recomputing distances.")
                else:
                    file = file.numpy()
                    return file
            else:
                print(f"Distance map file {filename} not found. Computing distances.")
        
        frame_count = self.frames.shape[0]
        height = self.frames.shape[1]
        width = self.frames.shape[2]
        
        print(f"Computing distances for {frame_count} frames with shape ({height}, {width})")
        dist_map = torch.zeros((frame_count, height, width), dtype=torch.float32)
        
        if cores > 1:
            frames_np = self.frames.cpu().numpy()
            args = [(i, frames_np[i], self.goal, height, width) for i in range(frame_count)]
            with Pool(processes=cores) as pool:
                results = pool.map(compute_single_frame_distances, args)
            for i, visited in results:
                dist_map[i] = visited
            print(f"Finished computing distances for {frame_count} frames")
        else:
            for i, frame in enumerate(self.frames):
                grid = frame.copy()
                visited = torch.full(grid.shape, -1, dtype=torch.float32)
                queue = deque([self.goal])
                visited[self.goal] = 0
                while queue:
                    cur_x, cur_y = queue.popleft()
                    for next_x, next_y in [(cur_x-1, cur_y), (cur_x+1, cur_y), (cur_x, cur_y-1), (cur_x, cur_y+1)]:
                        if next_x < 0 or next_x >= height:
                            continue
                        if next_y < 0 or next_y >= width:
                            continue
                        if visited[next_x, next_y] != -1:
                            continue
                        if frame[next_x, next_y] == 1:
                            continue
                    visited[next_x, next_y] = visited[cur_x, cur_y] + 1
                    queue.append((next_x, next_y))
                dist_map[i] = visited
        
        print(f"Distance map shape: {dist_map.shape}, Frames shape: {self.frames.shape}")
        
        torch.save(dist_map, filename)
        print(f"Saved precomputed distance map to {filename}")
        
        return dist_map.to(device=self.device)


    def step(self, actions):

        # if isinstance(actions, np.ndarray):
        #     actions = torch.from_numpy(actions).to(device=self.device, dtype=torch.long)
        # else:
        #     actions = actions.long()
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
        # timepen = -0.05
        # goalrew = 200
        # failedrew = -50
        # distancepen = 3
        # deathpen = -25
        
        #NEW
        timepen = -0.02
        goalrew = 150                
        failedrew = -40              
        distancepen = 2.0
        deathpen = -50
        
        wallhit = 0
        obstaclehit = 0
        dead = False
        start_step_distance = self.distances[self.steps, self.agent_pos[0], self.agent_pos[1]]
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
                    dead = True
                    break
                elif self.grid[self.agent_pos[0]-1, self.agent_pos[1]] == 1:
                    # above is obstacle
                    obstaclehit += 1
                    dead = True
                    break
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
                    dead = True
                    break
                elif self.grid[self.agent_pos[0]+1, self.agent_pos[1]] == 1:
                    # below is obstacle
                    obstaclehit += 1
                    dead = True
                    break
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
                    # wallhit += 1
                    dead = True
                    break
                elif self.grid[self.agent_pos[0], self.agent_pos[1]-1] == 1:
                    # left is obstacle
                    # obstaclehit += 1
                    dead = True
                    break
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
                    # wallhit += 1
                    dead = True
                    break
                elif self.grid[self.agent_pos[0], self.agent_pos[1]+1] == 1:
                    # right is obstacle
                    # obstaclehit += 1
                    dead = True
                    break
                else:
                    # valid move
                    self.agent_pos = (self.agent_pos[0], self.agent_pos[1]+1)
                    self.grid[self.agent_pos] = 3  # path
                    # record the move in the path
                    self.path_positions.append(self.agent_pos)
                    if self.agent_pos == self.goal:
                        self.goalhit = True
                        break

        end_step_distance = self.distances[self.steps, self.agent_pos[0], self.agent_pos[1]]
        distance_reward = start_step_distance - end_step_distance

        # Ensure start and goal markers remain
        self.grid[self.start] = 2
        self.grid[self.goal] = 2
        # self.gridhistory.append(self.grid.clone())
        
        # print(f"len start gridhistory: {len(self.gridhistory)} current step: {self.steps}")
        self.end_grid[self.steps] = self.grid
        self.grid = self.start_grid[self.steps].copy()
        # print(f"shape grid: {self.grid.shape} at step {self.steps}")
        self.steps += 1 
        if dead:
            reward += deathpen
            reward += distance_reward * distancepen  
            # print(f"Agent died at step {self.steps}!")
            self.terminated = True
        elif self.goalhit:
            reward += goalrew
            reward += distance_reward * distancepen  
            self.terminated = True
            print(f"Goal hit at step {self.steps}!")
            
        else: 
            reward += distance_reward * distancepen

        # print(f"steps_taken: {steps_taken}, wallhit: {wallhit}, obstaclehit: {obstaclehit}")
        # reward = reward + (steps_taken * timepen + wallhit * wallpen + obstaclehit * obstaclepen)
        reward = reward + (steps_taken * timepen)
        
        if not self.terminated and self.steps >= self.max_steps:
            self.truncated = True
            # print(f"Max steps reached: {self.steps} >= {self.max_steps}")
            if not self.goalhit:
                reward += failedrew
        self.grid[self.agent_pos] = 3  # current position
        self.grid[self.goal] = 2
        self.frame_stack.append(self.grid.copy())
        # print(f"steps_taken: {steps_taken}, total steps: {self.steps}, reward: {reward}, goalhit: {self.goalhit}, wallhit: {wallhit}, obstaclehit: {obstaclehit}, truncated: {self.truncated}, terminated: {self.terminated}")
        return self._obs(), float(reward), self.terminated, self.truncated, {'goal_hit': self.goalhit, 'wall_hits': wallhit, 'obstacle_hits': obstaclehit, 'steps_taken': steps_taken}


    def reset(self, *, seed=None, options=None):
        self.grid = self.start_grid[0].copy()
        self.end_grid.fill(0)
        self.steps = 0
        self.agent_pos = self.start
        self.grid[self.start] = 3
        self.grid[self.goal] = 2
        self.path_positions = [self.start]
        self.truncated = False
        self.terminated = False
        self.goalhit = False
        self.prev_distance = self.distances[0, self.start[0], self.start[1]]
        self.frame_stack.clear()
        for _ in range(self.stack):
            self.frame_stack.append(self.grid.copy())

        return self._obs(), {}
    
    def render(self, video=False):
        from occupy import Kinect
        kinect = Kinect()
        if video:
            kinect.videoPlayback(self.end_grid, extent=self.extent, steps=self.steps)
        else:
            if self.steps == 0:
                kinect.printgrid(self.start_grid[0], self.extent, "random_agent_initial")
            else:
                kinect.printgrid(self.end_grid[self.steps], self.extent, "random_agent_initial")
        
    def _obs(self):
        # if self.torchMode:
        #     obs = self.grid.clone()
        #     if self.cnn:
        #         obs = obs.unsqueeze(0)
        # else:
        #     obs = self.grid.clone().cpu().numpy()
        #     if self.cnn:
        #         obs = np.expand_dims(obs, axis=0)
        if self.cnn:
            obs = np.stack(list(self.frame_stack), axis=0)
            obs = torch.from_numpy(obs)
        else:
            obs = self.grid
        return obs

class Envs():
    def __init__(self, default=True):
        self.default = {
            'video_path': f"occupancypipe/videos/video_9489441.npy",
            'processed_frames_path': "occupancypipe/frames/processed_frames.npy",
            'calibration_frame_path': "occupancypipe/frames/calibration_frame.npy",
            'extent_frame_path': "occupancypipe/frames/extent.npy",
            'distances_path': "occupancypipe/hardenv_distance_map.pt",
            'processed': False,
            'duration': 0,
            'fps': 0,
            'count': 0,
            'z_min_threshold': -2.1, 
            'z_max_threshold': -1, 
            'crop': 20
        }
        
        self.data = {}
        self.json_path = "occupancypipe/environments.json"
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.data = json.load(f)
        else: 
            self.data['default'] = self.default
            self.save()
            
        self.preprocess_frames(default=True, duration=self.default['duration'], fps=self.default['fps'], count=self.default['count'])
        

    def add(self, duration, fps, count, z_min_threshold=-2.8, z_max_threshold=-1.5, crop=40):
        key = f"{duration}-{fps}-{count}"
        self.data[key] = {
            'video_path': f"occupancypipe/videos/video{duration}sec{fps}fps{count}.npy",
            'processed_frames_path': f"occupancypipe/frames/processed_frames_{duration}sec{fps}fps{count}.npy",
            # 'calibration_frame_path': f"occupancypipe/frames/calibration_frame_{duration}x{fps}{count}.npy",
            'extent_frame_path': f"occupancypipe/frames/extent_{duration}sec{fps}fps{count}.npy",
            'distances_path': f"occupancypipe/hardenv_distance_map_{duration}s_{fps}fps_ct{count}.pt",
            'processed': False,
            'duration': duration,
            'fps': fps,
            'count': count,
            'z_min_threshold': z_min_threshold, 
            'z_max_threshold': z_max_threshold,
            'crop': crop
        }
        self.save()
        self.preprocess_frames(default=False, duration=duration, fps=fps, count=count)

            

    def update(self, fps, duration, count, z_min_threshold=None, z_max_threshold=None, crop=None):
        key = f"{duration}-{fps}-{count}"
        if key in self.data:
            if z_min_threshold is not None:
                self.data[key]['z_min_threshold'] = z_min_threshold
            if z_max_threshold is not None:
                self.data[key]['z_max_threshold'] = z_max_threshold
            if crop is not None:
                self.data[key]['crop'] = crop
            self.save()
        else:
            print(f"Environment {key} not found in database.")
            
    def save(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.data, f, indent=4)
            
    def reset_env_files(self,default, duration=None, fps=None, count=None):
        # grab files paths from self.data and try to delete the file at that path
        if default:
            key = 'default'
        else:
            if duration is None or fps is None or count is None:
                print("Missing Key - reset env files.")
                return
            key = f"{duration}-{fps}-{count}"
        if key in self.data:
            data = self.data[key]
            files_to_delete = [
                data['processed_frames_path'],
                data['extent_frame_path'],
                data['distances_path']
            ]
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
            self.data[key]['processed'] = False
            self.save()
    
    def reset_env(self, default, duration, fps, count):
        if default:
            key = 'default'
        else:
            key = f"{duration}-{fps}-{count}"
        if key in self.data:
            env_data = self.data[key]
            env = harderEnv(
                torchMode=True,
                cnn=False,
                default=default,
                duration=env_data['duration'],
                fps=env_data['fps'],
                count=env_data['count'],
                cores=multiprocessing.cpu_count()-1
            )
            return env
        else:
            print(f"Environment {key} not found in database.")
            return None
        
    def test(self, default=True, duration=5, fps=5, count=4):
        if default:
            key = 'default'
        else:
            key = f"{duration}-{fps}-{count}"
            if not self.data[key]["processed"]:
                self.preprocess_frames(default=default, duration=duration, fps=fps, count=count)
        print(f"Testing environment: {key}")
        env = harderEnv(
            default=default,  
            duration=duration, 
            fps=fps,
            count=count,
            cores=6
        )
        # env.render()

        # action = torch.tensor([random.randint(0, 3) for _ in range(0, 499)], device=env.device, dtype=torch.long)  # random actions + explicit end
        action = torch.tensor([ 1 for _ in range(0, 499)], device=env.device, dtype=torch.long)
        for ep in range(3):
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

    def preprocess_frames(self, default=True, duration=5, fps=5, count=4):
        if default:
            data = self.data['default']
            # frame_name = "occupancypipe/frames/processed_frames.npy"
            # extent_name = "occupancypipe/frames/extent.npy"
        else:
            key = f"{duration}-{fps}-{count}"
            data = self.data[key]
        #     z_min_threshold = -2.8
        #     z_max_threshold = -1.5
        #     frame_name = f"occupancypipe/frames/processed_frames_{duration}sec{fps}fps{count}.npy"
        #     extent_name = f"occupancypipe/frames/extent_{duration}sec{fps}fps{count}.npy"
        #     if os.path.exists(frame_name) and os.path.exists(extent_name):
        #         print("Processed frames already exist. Skipping preprocessing.")
        #         return
        #     from occupy import Kinect
        #     kinect = Kinect()
        #     video = kinect.loadVideo(f"occupancypipe/videos/video{duration}sec{fps}fps{count}.npy")
        #     calibration_frame = kinect.loadFrame(f"occupancypipe/frames/calibration_frame_{duration}x{fps}{count}.npy", type='npy', view=False)
        #     kinect.calibrate(calibration_frame, z_min_threshold=z_min_threshold, z_max_threshold=z_max_threshold)
        #     frames, extent = kinect.createVideo(video, z_min_threshold=z_min_threshold, z_max_threshold=z_max_threshold, crop=40)
        #     frames = [kinect.denoise(frame) for frame in frames]
        #     frames = torch.from_numpy(np.stack(frames)).to(device=torch.device("cpu"), dtype=torch.float32)
        # np.save(f"occupancypipe/frames/processed_frames_{duration}sec{fps}fps{count}.npy", frames.cpu().numpy())
        # np.save(f"occupancypipe/frames/extent_{duration}sec{fps}fps{count}.npy", extent)
        
        if data['processed'] and os.path.exists(data['processed_frames_path']) and os.path.exists(data['extent_frame_path']):
            # print("Processed frames already exist. Skipping preprocessing.")
            return
        from occupy import Kinect
        kinect = Kinect()
        video = kinect.loadVideo(data['video_path'])
        
        calibrateframe = video[0]


        if default:
            kinect.calibrate(calibrateframe, z_min_threshold=data["z_min_threshold"], z_max_threshold=data["z_max_threshold"])
            frames, extent = kinect.createVideo(video)
        else:
            kinect.calibrate(calibrateframe, z_min_threshold=data["z_min_threshold"], z_max_threshold=data["z_max_threshold"])
            frames, extent = kinect.createVideo(video, z_min_threshold=data["z_min_threshold"], z_max_threshold=data["z_max_threshold"], crop=data["crop"])
            frames = [kinect.denoise(frame) for frame in frames]
        
        frames = torch.from_numpy(np.stack(frames)).to(device=torch.device("cpu"), dtype=torch.float32)
        frames = kinect.frameSkip(frames, skip=3)
        # print(f"shape after frame skip: {frames.shape}")
        np.save(data['processed_frames_path'], frames.cpu().numpy())
        print(data['processed_frames_path'])
        np.save(data['extent_frame_path'], extent)
        data['processed'] = True
        self.save()
    def reprocess_all(self):
        for key in self.data:
            if key == 'default':
                continue
            env = self.data[key]
            self.preprocess_frames(
                default=False,
                duration=env['duration'],
                fps=env['fps'],
                count=env['count']
            )
        
if __name__ == "__main__":
    envs = Envs()
    # envs.reset_env_files(default=False)
    duration = 5
    fps = 5
    count = 9
    crop = 1
    z_min_threshold = -1.8
    z_max_threshold = -0.5
    # for i in range(8, 14):
        # envs.add(duration=duration, fps=fps, count=i, z_min_threshold=z_min_threshold, z_max_threshold=z_max_threshold, crop=crop)

    #     envs.reset_env_files(default=False, duration=5, fps=5, count=i)
    # # envs.add(duration=5, fps=5, count=7, z_min_threshold=-1.8, z_max_threshold=-0.5, crop=40)
    # envs.add(duration=duration, fps=fps, count=count, z_min_threshold=z_min_threshold, z_max_threshold=z_max_threshold, crop=crop)
    # envs.test(default=False, duration=duration, fps=fps, count=count)
    # envs.test(default=False, duration=5, fps=5, count=8)
    envs.test(default=False, duration=5, fps=5, count=13)
    # envs.test(default=False, duration=5, fps=5, count=10)
    # envs.test(default=False, duration=5, fps=5, count=11)
    # envs.test(default=False, duration=5, fps=5, count=12)
    # envs.test(default=False, duration=5, fps=5, count=13)
    