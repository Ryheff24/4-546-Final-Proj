from os import times
import time
from unittest import skip
from flask import cli
import gymnasium as gym
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch
import matplotlib.pyplot as plt
import numpy as np
from hardenv import harderEnv
    
def evaluate(envinfo, model_name,env_name="2d-occupancy", episodes=100, render=True, plot=False):# -> list:

    env= harderEnv(torchMode=False, default=False, duration=envinfo[0], fps=envinfo[1], count=envinfo[2])
    model = PPO.load(model_name, env=env)
    total_rewards = []
    rewardsOverEps = []
    ran = False
    for ep in range(episodes):
        rew = 0
        obs, _ = env.reset()
        rewardArr = []
        while True:
            action, _states = model.predict(obs)
            obs, rewards, term, trunc, info = env.step(action)
            rew += rewards
            rewardArr.append(rewards)
            if term or trunc:
                break
        if render and rew >= 875 and not ran: 
            env.render(video=True)
            ran = True
        total_rewards.append(rew)
        rewardsOverEps.append(rewardArr)
        if plot: print(f"Episode {ep+1}: Reward = {rew:.2f}")

    avg = np.mean(total_rewards)
    singleps = np.mean(rewardsOverEps[0])
    if plot:
        print(f"\nAverage Reward across {episodes} eval episodes: {avg:.2f}")
        plt.figure()
        plt.plot(total_rewards, marker='o', color='red')
        plt.title(f"Evaluation Rewards per Episode - {env_name}")
        plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.show()
        plt.figure()
        plt.plot(rewardsOverEps[0], marker='o', color='red')
        plt.title(f"Evaluation Rewards per One Episode - {env_name}")
        plt.close('all')
    return avg

def retrain(envinfo, model_name, total_timesteps=1, n_envs=16):
    env = SubprocVecEnv([lambda: harderEnv(torchMode=False, default=False, duration=envinfo[0], fps=envinfo[1], count=envinfo[2]) for _ in range(n_envs)])
    env = VecMonitor(env)
    # if cnn: 
    #     model = PPO.load(model_name, env=env, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    #             learning_rate=1e-4,
    #             clip_range=0.1,
    #             ent_coef=0.01,
    #             policy_kwargs=dict(normalize_images=False))
    # else:
    model = PPO.load(model_name, env=env,
            learning_rate=3e-5,
            ent_coef=0.05)
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"2D_{model_name}_retrain")
    model.save(model_name+"_retrained")
    return model_name+"_retrained"


def train(envinfo, model_name, total_timesteps=1000000, n_envs=16, ):
    # if not cnn:
    #     device = torch.device("cpu")
    # else:
    #     if torch.backends.mps.is_available():
    #         device = torch.device("mps")
    #     elif torch.cuda.is_available():
    #         device = torch.device("cuda")
    #     else:
    device = torch.device("cpu")
        
    env = SubprocVecEnv([lambda: harderEnv(torchMode=False, default=False, duration=envinfo[0], fps=envinfo[1], count=envinfo[2]) for _ in range(n_envs)])
    env = VecMonitor(env)
    # if cnn: 
    #     model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_2d_tensorboard/", device=device, 
    #             learning_rate=1e-4,
    #             clip_range=0.1,
    #             ent_coef=0.01,
    #             policy_kwargs=dict(normalize_images=False))
    # else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_2d_tensorboard/")
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"{model_name}")
    model.save(model_name)

    # del model # remove to demonstrate saving and loading
    evaluate(model_name, plot=True)

def main():
    print("You must select the environment by inserting the duration, fps, and count values used by the video file.")
    # filename = f"occupancypipe/frames/calibration_frame_{duration}x{fps}{count}.npy"
    # check file exists
    duration = input("Enter duration (seconds): ")
    duration = 5 if duration == '' else int(duration)
    
    fps = input("Enter frames per second (fps): ")
    fps = 5 if fps == '' else int(fps)
    
    count = input("Enter count (difficulty level): ")
    count = 8 if count == '' else int(count)
    # filename = 
    # filename = input("Enter a name to save the model: ")
    model_name = f"{duration}-{fps}-{count}"
    file =(duration, fps, count)
    mode = input("Enter model type (train, retrain, evaluate): ")
    modesarr = ['train', 't', 'retrain', 'r', 're', 'evaluate', 'e', 'eval']
    while mode.lower() not in modesarr:
        mode = input("Invalid mode. Enter mode (train, retrain, evaluate): ")
        
    if mode.lower() in ['evaluate', 'e', 'eval']:
        evaluate(file, model_name, plot=True, render=True)
        return
    
    # while model_type not in ['cnn', 'mlp']:
    #     model_type = input("Invalid model type. Enter model type (cnn, mlp): ")
    # model_name = f"{filename}"
    envcount = input("Enter number(n) of envs(0 >= n <= 32): ")
    envs = 16 if envcount == '' else int(envcount)
    while envs < 1 or envs > 32:
        envs = int(input("Invalid number of envs. Enter number(n) of envs(0 >= n <= 32): "))
        
    n_envs = envs
    
    if mode.lower() in ['train', 't']:
        train(file, model_name, n_envs=n_envs)
        return

    if mode.lower() in ['retrain', 'r', 're']:
        time_steps = input("Enter total timesteps to retrain( default 500000 ): ")
        total_timesteps = 500000 if time_steps == '' else int(time_steps)
        retrain(file, model_name, n_envs=n_envs, total_timesteps=total_timesteps)
        return
    
if __name__ == "__main__":
    main()