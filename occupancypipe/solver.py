import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch

from hardenv import harderEnv
def main():
    n_envs = 24
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:

    env = SubprocVecEnv([lambda: harderEnv(torchMode=False) for _ in range(n_envs)])
    env = VecMonitor(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("ppo_2d")

    # del model # remove to demonstrate saving and loading
    env = harderEnv(torchMode=False)
    model = PPO("MlpPolicy", env, verbose=1)


    model = PPO.load("ppo_2d")
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