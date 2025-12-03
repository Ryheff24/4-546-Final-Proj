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
    
def evaluate(model_name,env_name="2d-occupancy", episodes=100, render=True, plot=False, cnn=False):# -> list:

    env= harderEnv(torchMode=False, cnn=cnn)
    model = PPO.load(model_name, env=env)
    total_rewards = []
    rewardsOverEps = []
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
        if render and rew >= 875: env.render(video=True)
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

def retrain(model_name, total_timesteps=1, n_envs=16, cnn=False):
    env = SubprocVecEnv([lambda: harderEnv(torchMode=False, cnn=cnn) for _ in range(n_envs)])
    env = VecMonitor(env)
    if cnn: 
        model = PPO.load(model_name, env=env, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                learning_rate=1e-4,
                clip_range=0.1,
                ent_coef=0.01,
                policy_kwargs=dict(normalize_images=False))
    else:
        model = PPO.load(model_name, env=env)
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"2D_{model_name}_retrain")
    model.save(model_name+"_retrained")
    return model_name+"_retrained"


def main():
    filename = input("Enter a name to save the model: ")
    model_type = input("Enter model type (cnn, mlp): ")
    while model_type not in ['cnn', 'mlp']:
        model_type = input("Invalid model type. Enter model type (cnn, mlp): ")
    modelname = f"{filename}"
    cnn= model_type == 'cnn'
    skip = input("Skip training? (y/n): ")
    if skip.lower() == 'y' or skip.lower() == 'yes'or skip == '':
        evaluate(modelname, plot=True, render=True, cnn=cnn)
        return
    
    envcount = input("Enter number(n) of envs(0 >= n <= 32): ")
    envs = 16 if envcount == '' else int(envcount)
    while envs < 1 or envs > 32:
        envs = int(input("Invalid number of envs. Enter number(n) of envs(0 >= n <= 32): "))
        
    n_envs = envs
    if model_type == "mlp":
        device = torch.device("cpu")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
    env = SubprocVecEnv([lambda: harderEnv(torchMode=False, cnn=cnn) for _ in range(n_envs)])
    env = VecMonitor(env)
    if cnn: 
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_2d_tensorboard/", device=device, 
                learning_rate=1e-4,
                clip_range=0.1,
                ent_coef=0.01,
                policy_kwargs=dict(normalize_images=False))
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_2d_tensorboard/")
    model.learn(total_timesteps=1000000, tb_log_name=f"2D_{modelname}")
    model.save(modelname)

    # del model # remove to demonstrate saving and loading
    evaluate(modelname, plot=True, cnn=cnn)
if __name__ == "__main__":
    main()