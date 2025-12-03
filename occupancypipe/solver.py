import gymnasium as gym
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch

from hardenv import harderEnv
def main():
    n_envs = 10
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    env = SubprocVecEnv([lambda: harderEnv(torchMode=False, cnn=True) for _ in range(n_envs)])
    env = VecMonitor(env)
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_2d_tensorboard/", device=device, batch_size=512, n_steps=2048,
                policy_kwargs=dict(normalize_images=False))
    model.learn(total_timesteps=1000000, tb_log_name="PPO_2D_Run2")
    model.save("ppo_2d3")

    # del model # remove to demonstrate saving and loading
    env = harderEnv(torchMode=False, cnn=True)
    model = PPO("CnnPolicy", env, verbose=1)


    model = PPO.load("ppo_2d3")
    rew = 0
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term, trunc, info = env.step(action)
        rew += rewards
        if term or trunc:
            break
    print(f"Total reward: {rew}")
    env.render(video=True)
    
if __name__ == "__main__":
    main()