# CSE 4/546 Final Project — Team 49

This repo contains our end-to-end **occupancy grid navigation** project.

The core of the project lives in the [occupancypipe/](occupancypipe/) folder:

1) (Optional) capture **Kinect v2** depth frames as point clouds
2) convert point clouds into **2D occupancy grids**
3) create a dynamic **GridWorld-like navigation environment** where obstacles change over time
4) train a navigation policy with PPO via **Stable-Baselines3** (interactive driver)

Other folders like `stream-udp/` and `stream-mem/` are earlier experiments and were not used in the final pipeline.

---

## Project overview

We model navigation through a time-varying occupancy grid sequence (a “video” of occupancy maps). At each environment step, the agent chooses a *bundle* of moves (default: 12) and receives reward based on:

- progress toward the goal (distance-to-goal shaping)
- small time penalty per move
- terminal reward for reaching goal
- terminal penalty for dying (hitting wall/obstacle)
- penalty for timing out (episode ends without reaching goal)

The environment supports multiple “difficulty” datasets parameterized by:

`duration-fps-count` (example: `5-5-13`)

where `count` is the dataset/difficulty index used in filenames.

---

## Key entrypoints

### Data capture + preprocessing (Kinect → occupancy grids)

- [occupancypipe/occupy.py](occupancypipe/occupy.py):
  - captures raw depth frames (point clouds)
  - saves “videos” (`.npy`/`.npz`) of point clouds
  - converts point-cloud videos into 2D occupancy grid frames

Notes:
- This part requires Kinect v2 + `libfreenect2`.
- Some paths are currently hard-coded for a local machine; see “Path gotchas” below.

### CPU reference environment

- [occupancypipe/hardenv.py](occupancypipe/hardenv.py): `harderEnv`
  - loads preprocessed occupancy frames and a precomputed BFS distance map
  - implements Gymnasium env logic and reward shaping
  - includes an `Envs` helper to manage datasets and preprocessing

### Training

Training is done with **Stable-Baselines3 PPO** via an interactive driver:

- [occupancypipe/solver.py](occupancypipe/solver.py)

### Analysis

- [analyze_training.py](analyze_training.py): reads TensorBoard event files and summarizes training health/plateaus

---

## Repo layout (high level)

- [occupancypipe/](occupancypipe/): main pipeline (capture → occupancy grids → envs → training)
  - [occupancypipe/frames/](occupancypipe/frames/): processed occupancy frames + extents
  - [occupancypipe/videos/](occupancypipe/videos/): point-cloud “videos” (`video{duration}sec{fps}fps{count}.npy`)
  - [occupancypipe/models/](occupancypipe/models/): saved model checkpoints (SB3)
- `libfreenect2/`: Kinect v2 driver library (source included)
- `octomap/`: octomap library (not core to final pipeline)
- `stream-udp/`, `stream-mem/`: earlier streaming prototypes (not used for final results)
- `ppo_2d_tensorboard/`: TensorBoard logs from SB3 runs

---

## Environment setup

We kept multiple conda environment files; the main one used is:

- [environment.yml](environment.yml)

Create and activate:

```bash
conda env create -f environment.yml
conda activate kinect-env
```

If you don’t need Kinect capture, you can still run training/validation using the existing `occupancypipe/frames/*` and distance maps.

---

## Quickstart (no Kinect required)

### 1) Train / retrain / evaluate with Stable-Baselines3

This is an interactive CLI:

```bash
python occupancypipe/solver.py
```

It will prompt for:
- `duration`, `fps`, `count` (selects the dataset)
- mode: train / retrain / evaluate
- number of parallel envs

SB3 logs go to `./ppo_2d_tensorboard/` and checkpoints typically go under `occupancypipe/models/`.

### 2) Quick environment sanity check

The main environment logic lives in [occupancypipe/hardenv.py](occupancypipe/hardenv.py). Running it directly will execute a basic test in its `__main__` block (it may load/preprocess frames depending on configuration):

```bash
python occupancypipe/hardenv.py
```

---

## Data pipeline (Kinect → occupancy grids)

If you have Kinect v2 hardware and want to reproduce the capture process:

1) Build `libfreenect2` (platform-specific; see `libfreenect2/README.md`).
2) Run the capture/convert script:
	- [occupancypipe/occupy.py](occupancypipe/occupy.py)

The typical flow is:

- record a point-cloud video (`occupancypipe/videos/video{duration}sec{fps}fps{count}.npy`)
- convert to 2D occupancy frames (`occupancypipe/frames/processed_frames_{duration}sec{fps}fps{count}.npy`)
- compute/load distance maps (`occupancypipe/hardenv_distance_map_{duration}s_{fps}fps_ct{count}.pt`)

The `Envs` helper in [occupancypipe/hardenv.py](occupancypipe/hardenv.py) can add/reprocess dataset variants and generate the needed `processed_frames_*` and `extent_*` files.

---

## Training logs & analysis

- To view TensorBoard logs:

```bash
tensorboard --logdir ppo_2d_tensorboard
```

- To summarize TensorBoard event files:

```bash
python analyze_training.py
```

---

## Path gotchas (important)

Some scripts were written for a specific local machine and may include absolute paths:

- [occupancypipe/occupy.py](occupancypipe/occupy.py) defaults `inputdir` to a local path.
- [run_visual.sh](run_visual.sh) uses absolute paths and sets `DYLD_LIBRARY_PATH`.

If you are running on a different machine, update those paths or pass your own `inputdir` when constructing `Kinect(...)`.

---

## Notes on legacy folders

`stream-udp/` and `stream-mem/` were early prototypes for streaming depth/pointcloud data. They’re not required to run the final pipeline and can be ignored for reproducing results.

---

## Acknowledgements

- `libfreenect2` is used for Kinect v2 access.
- `open3d`, `gymnasium`, and `stable-baselines3` are used for the learning stack.
