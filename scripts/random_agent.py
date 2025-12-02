#!/usr/bin/env python3
"""
Random agent test for `hardenv.harderEnv`.
Usage: run from repo root with the kinect environment active (if required):

    python scripts/random_agent.py

The script loads `occupancypipe/hardenv.py` by path so it doesn't require package imports.
It runs a few episodes and renders the grid after each episode.
"""
import importlib.util
import os
import sys
import random
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARDENV_PATH = os.path.join(REPO_ROOT, 'occupancypipe', 'hardenv.py')

if not os.path.exists(HARDENV_PATH):
    print('Could not find hardenv.py at', HARDENV_PATH)
    sys.exit(1)

spec = importlib.util.spec_from_file_location('hardenv', HARDENV_PATH)
hardenv = importlib.util.module_from_spec(spec)
try:
    # Ensure the occupancypipe package directory is on sys.path so imports inside
    # `hardenv.py` (e.g. `from occupy import Kinect`) resolve correctly.
    occupy_dir = os.path.join(REPO_ROOT, 'occupancypipe')
    if occupy_dir not in sys.path:
        sys.path.insert(0, occupy_dir)
    spec.loader.exec_module(hardenv)
except Exception as e:
    print('Failed to load hardenv module:', e)
    print('Traceback:')
    raise

if not hasattr(hardenv, 'harderEnv'):
    print('hardenv.py does not define harderEnv')
    sys.exit(1)

EnvClass = hardenv.harderEnv

# Create environment (catch file/Kinect errors and print friendly message)
try:
    env = EnvClass('person.npy', max_steps=200, obstacles=5)
except Exception as e:
    print('Failed to instantiate environment:', e)
    print('\nIf this is due to missing numpy frames (git-lfs pointers), run `git lfs pull`')
    sys.exit(1)

EPISODES = 40
MAX_STEPS_PER_EP = 200

print(f'Running {EPISODES} episodes (max {MAX_STEPS_PER_EP} steps each)...')
for ep in range(EPISODES):
    obs, _, _, _, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False
    # run until env reports truncation (max steps reached)
    while True:
        # choose a random movement action only (0:up,1:down,2:left,3:right)
        action = random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if truncated:
            break
    print(f'Episode {ep+1}: total_reward={total_reward:.2f}, terminated={terminated}, truncated={truncated}, steps={env.current_step}')

# After all episodes, render one final graph with the agent path from the last episode
print('Rendering final agent path (from last episode)')
try:
    out_path = os.path.join(REPO_ROOT, 'scripts', 'last_path.png')
    env.render(save_path=out_path)
    print('Saved final path to', out_path)
except Exception as e:
    print('Render failed:', e)

print('Random agent test finished')
