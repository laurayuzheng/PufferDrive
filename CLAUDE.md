# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Objectives
This project is a deep learning research project which investigates latent variable modeling with learning-based simulation. 
An open-loop version of latent variable modeling for trajectory prediction is implemented in ```pc_driving/polysona/models/polysona.py```. This project was copied into this repository; PufferDrive currently does not implement anything from Polysona.

The goal of this latent variable modeling project is to learn K (in most experiments, K=3) discrete buckets of driving styles with the latent variable with a router module, which predicts bucket logits, and K LoRA modules in the form of task-skill matrices. The attribute in ```polysona.py``` which refers to this is "skilled_variant", where "private" corresponds to standard, isolated LoRA modules. 

There are other skilled variants, which can be referenced in ```pc_driving/polysona/utils/polytropon.py```: "learned", "hyper", "sparse", "cpoly", "expert", "expert_output", "mov".

Anyways, the context of this particular project is the close-loop version of what is already implemented in the directory, ```pc_driving```. pc_driving should be the reference for extending latent variable modeling under the PufferLib framework, which is a fast simulation environment for reinforcement learning. Under close-loop training, we hypothesize that differences in driving styles will become more apparent.

The abstract of the open-loop work is here, for informational purposes: "In rare but safety-critical driving scenarios, we hypothesize that trajectory outcomes become increasingly multi-modal based on differences between driver style compared to non-critical, common scenarios. However, current approaches for trajectory prediction rarely account for differences in driving style, which may lead to ``averaged" driving style in predictions. While average-case behavior may work well in straight driving, easy scenarios, it limits the diversity of outcomes in more complex scenes or in rare events. Extraction of driving style has several benefits, as it enables simulation of counterfactual outcomes in real-world log replays and potentially more accurate predictions through style-consistent predictions. In this paper, we present a parameter-efficient Mixture-of-Experts framework for extraction of latent driving styles in trajectory prediction models."

The goal of this project is to do something similar, but in close-loop.

## Guidelines for Code Contributions

All additional implementations should be separate from original Pufferlib or pc_driving implementations, but also work under their reference frameworks. For example, latent variable experiments should have their own configs and own model class files, such that we can run both baseline and latent variable experiments at any time by changing command line arguments or config references in the code.

## Simulator Overview

PufferDrive is a high-throughput driving simulator built on PufferLib for training and evaluating RL-based autonomous driving agents. It features Box2D physics, multi-agent support (1024+ agents), and integration with Waymo Open Motion Dataset (WOMD).

The baseline logic for PufferDrive training can be found at ```pufferlib/pufferl.py```.

**Docs**: https://emerge-lab.github.io/PufferDrive

## Build Commands

```bash
# Install dependencies (use uv or pip)
uv pip install -e .

# Compile C extensions (REQUIRED after cloning or modifying C files)
python setup.py build_ext --inplace --force

# Build local visualizer
bash scripts/build_ocean.sh drive local

# Build web visualizer (requires emscripten)
bash scripts/build_ocean.sh drive web
```

## Training and Evaluation

```bash
# Basic training (baseline)
puffer train puffer_drive

# MoE training (latent variable modeling)
puffer train puffer_drive_moe

# Distributed training (6 GPUs)
torchrun --standalone --nnodes=1 --nproc-per-node=6 -m puffer train puffer_drive

# Training with W&B logging
puffer train puffer_drive --wandb --wandb-project pufferdrive

puffer train puffer_drive_moe --wandb --wandb-project pufferdrive --exp-name imit_coef=0.05

# WOSAC realism evaluation
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path <checkpoint.pt>

# Human-replay evaluation
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path <checkpoint.pt>

# Baseline WOSAC eval
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path experiments/puffer_drive_diversity.pt

# MoE WOSAC eval
puffer eval puffer_drive_moe --eval.wosac-realism-eval True --load-model-path experiments/puffer_drive_moe_social_forces.pt

# Most recent eval
puffer eval puffer_drive_mtr  --eval.wosac-realism-eval True --load-model-path experiments/puffer_drive_mtr_base.pt 
```

Results

Metric	MoE (new)	MoE (old)	Baseline
ADE	10.26	18.53	7.12
Collisions	0.56	1.95	0.26
Realism	0.746	0.734	0.768

## Exporting Weights for Visualizer

The Raylib visualizer requires weights in a binary format (`.bin`). Export PyTorch checkpoints using:

```bash
# Export specific checkpoint to visualizer location
puffer export puffer_drive --load-model-path experiments/puffer_drive_moe_x43lcn11.pt --output-path resources/drive/puffer_drive_weights.bin

# Export latest checkpoint
puffer export puffer_drive --load-model-path latest --output-path resources/drive/puffer_drive_weights.bin

# Export to default location (pufferlib/resources/drive/puffer_drive_weights.bin)
puffer export puffer_drive --load-model-path experiments/my_checkpoint.pt
```

**Paths**:
- Default export location: `pufferlib/resources/drive/{env_name}_weights.bin`
- Visualizer expects: `resources/drive/puffer_drive_weights.bin` (hardcoded in `visualize.c:419`)

**Note**: The C visualizer has two network implementations:
- `drivenet.h`: Baseline `Drive` policy architecture
- `drivenet_moe.h`: MoE `DriveMoE` policy architecture with LoRA experts

## Visualizing the Policy

To run the visualizer in a pop up:
```bash
# Build visualizer
bash scripts/build_ocean.sh visualize local

# Run baseline policy
./visualize

# Run MoE policy (use --moe flag)
./visualize --moe
```

To run the visualizer headless and export an mp4:
```bash
# Baseline
xvfb-run -s "-screen 0 1280x720x24" ./visualize

# MoE
xvfb-run -s "-screen 0 1280x720x24" ./visualize --moe
```

### Matplotlib Visualization (plt_visualize.py)

For more flexible visualization without building C extensions, use the matplotlib-based visualizer:

```bash
# Basic state visualization (2D)
python plt_visualize.py --output viz_output/state.png

# Generate rollout GIF with trained policy
python plt_visualize.py --output viz_output/rollout.gif --checkpoint experiments/puffer_drive_moe_imit.pt --num-steps 50

# 3D rendering
python plt_visualize.py --output viz_output/state_3d.png --render-3d

# 3D POV mode (camera follows ego vehicle, heading always points up)
python plt_visualize.py --output viz_output/pov.gif --render-3d --pov --checkpoint experiments/puffer_drive_moe_imit.pt

# MoE multiverse comparison (compare trajectories across experts)
python plt_visualize.py --output viz_output/multiverse.gif --mode multiverse --checkpoint experiments/puffer_drive_moe_imit.pt

# Ground truth human trajectories
python plt_visualize.py --output viz_output/trajectories.png --mode trajectories

# Adjust zoom and number of agents
python plt_visualize.py --output viz_output/zoomed.gif --zoom-radius 30 --num-agents 7

# latest command
python plt_visualize.py --mode multiverse -o viz_output/multiverse.gif  --pov --forced-agent 2 --scenario 10 --config puffer_drive_moe --checkpoint experiments/puffer_drive_moe_diversity.pt
```

**Key options**:
- `--render-3d`: Enable 3D rendering with vehicle cuboids
- `--pov`: POV mode - camera on top of ego vehicle, rotates with heading (3D only)
- `--checkpoint`: Path to trained policy checkpoint
- `--mode`: `state`, `rollout`, `trajectories`, `multiverse`, or `auto` (default)
- `--zoom-radius`: Viewport radius in meters (default: 80)
- `--num-steps`: Number of simulation steps for GIF/video
- `--show-trajectories`: Show trajectory trails in rollout mode

**POV Mode Details**:
- Camera positioned directly above the ego vehicle
- View rotates so vehicle heading always points to top of image
- Auto-zooms to 15m radius for close-up view
- Ego vehicle highlighted in gold color

### Disabling Training-time Rendering

To disable the C visualizer during training (which runs at `checkpoint_interval`):

```bash
# Via command line
puffer train puffer_drive_moe --train.render False

# Or edit config: pufferlib/config/ocean/puffer_drive_moe.ini
# render = False
```

## Testing

```bash
# Run Python tests
python tests/test_drive_config.py
python tests/test_drive_train.py

# Run C INI parser tests
./tests/ini_parser/build_n_test.sh

# Test MoE architecture matches between PyTorch and C
python -m pytest tests/test_moe_architecture.py -v
```

### MoE Architecture Test

The `tests/test_moe_architecture.py` test ensures the PyTorch `DriveMoE` architecture matches the C `drivenet_moe.h` implementation. This is critical because weight export order must match weight loading order in C.

The test verifies:
- Parameter names and shapes match expected order
- Total weight count matches (635,175 parameters)
- Exported .bin file size is correct
- Constants (input_size, hidden_size, num_experts, etc.) match between Python and C

If you modify the MoE architecture in `torch.py`, run this test to see which parameters changed, then update `drivenet_moe.h` accordingly.

## Code Formatting

- **Python**: Ruff (120-char lines, Python 3.10+). Config: `ruff.toml`
- **C/C++**: clang-format (LLVM style, 4-space indent). Config: `.clang-format`
- Pre-commit hooks run ruff-format, clang-format, and nb-clean

## Architecture

### Core Components

- **`pufferlib/pufferl.py`**: Main training/eval loop (PuffeRL class), PPO updates, checkpointing
- **`pufferlib/pufferlib.py`**: PufferEnv base class defining vectorized multi-agent interface
- **`pufferlib/vector.py`**: Multi-worker environment vectorization (Multiprocessing, Serial, MPI)
- **`pufferlib/models.py`**: Neural network architectures (includes `LSTMWrapper`)
- **`pufferlib/ocean/torch.py`**: Drive-specific policies (`Drive`, `DriveMoE`)
- **`pufferlib/ocean/moe_adapters.py`**: MoE components (LoRA layers, router, auxiliary losses)

### Driving Simulator (`pufferlib/ocean/drive/`)

- **`drive.h`**: Core C simulator (~3700 lines) - Box2D physics, agent management, road graph, collision detection, observation encoding (ORU)
- **`drive.c`**: Simulator implementation
- **`binding.c`**: Python<->C interface (generates `binding.cpython-*.so`)
- **`drive.py`**: Python wrapper, config loading, reset/step interface
- **`visualize.c`**: Raylib-based rendering
- **`drivenet.h`**: Neural network policy definitions

### Configuration

Uses INI format (not YAML/JSON):
- **`pufferlib/config/default.ini`**: Base hyperparameters
- **`pufferlib/config/ocean/drive.ini`**: Drive-specific settings (map_dir, num_agents, rewards, etc.)
- **`pufferlib/config/ocean/puffer_drive_moe.ini`**: MoE variant config with LoRA and router settings

Key env settings: `action_type` (discrete/continuous), `dynamics_model` (classic/jerk), `control_mode`, `goal_behavior`

### MoE Implementation (`pufferlib/ocean/`)

The Mixture-of-Experts variant implements latent variable modeling for driving styles:

- **`torch.py`**: Contains `DriveMoE` class - MoE policy with frozen base weights and trainable LoRA adapters
- **`moe_adapters.py`**: LoRA expert layers (`LoRAExpertsRL`), router (`PersonaRouter`), and auxiliary loss functions

**Key MoE parameters** (in `puffer_drive_moe.ini`):
- `num_experts`: Number of discrete driving style buckets (default: 3)
- `lora_rank`: Rank of LoRA decomposition (default: 8)
- `lora_alpha`: LoRA scaling factor (default: 4.0)
- `freeze_base`: Whether to freeze base encoder weights (default: True)
- `base_checkpoint`: Path to pretrained baseline checkpoint for weight initialization

**Trainable vs Frozen parameters**:
- Trainable: `router.*`, `actor.expert_A`, `actor.expert_B` (LoRA matrices)
- Frozen: `ego_encoder.*`, `road_encoder.*`, `partner_encoder.*`, `shared_embedding.*`, `value_fn.*`

**Important**: When using `LSTMWrapper`, the policy's `encode_observations()` method must return only the hidden tensor (not a tuple). Store auxiliary outputs like `expert_probs` as instance variables and retrieve them in `decode_actions()`.

### MTR Implementation (`pufferlib/ocean/torch_mtr.py`)

The MTR variant uses the full Motion Transformer (MTR) architecture from NeurIPS 2022, adapted for closed-loop RL with MPC-style GMM action output.

**Two architectures**:
1. **DriveMTR** (baseline): Full MTR backbone, train end-to-end
2. **DriveMTRMoE**: Frozen backbone + LoRA experts on actor

**GMM Action Head Format (7 values per timestep)**:
```
[0] accel: acceleration (direct)
[1] steer_raw: steering before tanh transform
[2] log_σ_accel: log std of acceleration
[3] log_σ_steer: log std of steering
[4] ρ: correlation coefficient
[5] vx: predicted velocity x
[6] vy: predicted velocity y
```

**Transforms** (from `mtr_actions.py`):
- `mu_accel = pred[:, :, :, 0]` (direct)
- `mu_steer = tanh(pred[:, :, :, 1]) * π/3` (scaled to [-60°, +60°])

**Closed-loop execution**:
1. Take raw `pred_trajs[:, :, 0, 0:2]` (first timestep)
2. Apply transforms: `accel = [0]`, `steer = tanh([1]) * π/3`
3. Discretize to action buckets

**Key files**:
- `pufferlib/ocean/torch_mtr.py`: `DriveMTR` (baseline) and `DriveMTRMoE` policies
- `pufferlib/ocean/mtr_encoder.py`: Full MTR architecture components with kinematic integration
- `pufferlib/config/ocean/puffer_drive_mtr.ini`: Baseline MTR configuration
- `pufferlib/config/ocean/puffer_drive_mtr_moe.ini`: MTR-MoE configuration

**Key parameters** (in `puffer_drive_mtr_moe.ini`):
- `d_model`: Transformer hidden dimension (default: 128)
- `nhead`: Number of attention heads (default: 4)
- `num_encoder_layers`: Encoder self-attention layers (default: 3)
- `num_decoder_layers`: Decoder cross-attention layers (default: 3)
- `num_queries`: Intention/action queries (default: 4)
- `num_future_frames`: GMM trajectory prediction horizon (default: 40)
- `freeze_base`: Whether to freeze encoder (train LoRA only)
- `base_checkpoint`: Path to pretrained DriveMTR for weight initialization
- `aux_reconstruction_coef`: Reconstruction loss weight (default: 50.0)

**Training**:
```bash
# Baseline MTR
puffer train puffer_drive_mtr

# MTR-MoE
puffer train puffer_drive_mtr_moe

# With W&B logging
puffer train puffer_drive_mtr_moe --wandb --wandb-project pufferdrive

# Distributed training (6 GPUs)
torchrun --standalone --nnodes=1 --nproc-per-node=6 -m puffer train puffer_drive_mtr_moe

# Train MoE with frozen backbone (requires pretrained baseline)
puffer train puffer_drive_mtr_moe --policy.freeze_base True --policy.base_checkpoint experiments/puffer_drive_mtr_base.pt --wandb --wandb-project pufferdrive --exp-name polysona

# Polysona
puffer train puffer_drive_mtr_moe --wandb --wandb-project pufferdrive --exp-name polysona
```

**Evaluation**:
```bash
# WOSAC realism evaluation
puffer eval puffer_drive_mtr_moe --eval.wosac-realism-eval True --load-model-path <checkpoint.pt>

# Human-replay evaluation
puffer eval puffer_drive_mtr_moe --eval.human-replay-eval True --load-model-path <checkpoint.pt>
```

**Visualization** (use `plt_visualize.py` - C visualizer does not support MTR):
```bash
# Rollout GIF
python plt_visualize.py --output viz_output/rollout.gif --checkpoint experiments/puffer_drive_mtr_moe.pt --num-steps 50 --config puffer_drive_mtr_moe

# Multiverse comparison (MoE experts)
python plt_visualize.py --mode multiverse --output viz_output/multiverse.gif --checkpoint experiments/puffer_drive_mtr_moe.pt --config puffer_drive_mtr_moe

# 3D POV mode
python plt_visualize.py --output viz_output/pov.gif --render-3d --pov --checkpoint experiments/puffer_drive_mtr_moe.pt --config puffer_drive_mtr_moe
```

**Note**: The C visualizer (`./visualize`) only supports `Drive` and `DriveMoE` architectures. Use `plt_visualize.py` for MTR visualization.

### Imitation Learning (`pufferlib/ocean/inverse_dynamics.py`)

The imitation learning module enables behavioral cloning by computing expert actions from human trajectory data via inverse dynamics.

**Training Objective**:

The total loss combines PPO policy gradient with an imitation term:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + \lambda_{\text{imit}} \mathcal{L}_{\text{imit}}$$

where the imitation loss is cross-entropy between policy logits and expert actions:

$$\mathcal{L}_{\text{imit}} = -\frac{1}{|V|} \sum_{i \in V} \log \pi_\theta(a_i^* | s_i)$$

- $V$ = set of valid samples (where both current and next trajectory timesteps are valid)
- $a_i^*$ = expert action computed via inverse dynamics from human trajectory
- $\pi_\theta$ = policy network output (discrete action logits)
- $\lambda_{\text{imit}}$ = `imit_coef` config parameter (default: 0.1 for MoE, 0 for baseline)

**Inverse Dynamics**:

Expert actions are computed by inverting the bicycle model dynamics:

1. **Acceleration**: $a = \frac{v_{t+1} - v_t}{\Delta t}$ where $v$ is signed speed
2. **Steering**: $\delta \approx \arctan\left(\frac{\dot{\psi} \cdot L}{v}\right)$ where $\dot{\psi}$ is yaw rate, $L$ is wheelbase
3. **Discretization**: Find closest action index $a^* = \arg\min_i |a - a_i| \cdot 13 + \arg\min_j |\delta - \delta_j|$

The discrete action space has 91 actions (7 acceleration × 13 steering values).

**Key files**:
- `pufferlib/ocean/inverse_dynamics.py`: Inverse dynamics computation and loss function
- `pufferlib/ocean/drive/drive.py`: `get_expert_action_at_timestep()` method
- `pufferlib/vector.py`: `get_expert_actions()` in Serial and Multiprocessing backends
- `pufferlib/pufferl.py`: Integration in training loop (lines 468-492)

**Timestep Wrapping for Imitation Learning**:

The `get_expert_action_at_timestep()` method uses modulo to wrap the timestep within episode bounds:
- Human trajectories have `episode_length` timesteps (e.g., 91)
- Expert actions are valid for timesteps 0 to `episode_length - 2` (e.g., 0-89)
- The environment's `tick` is wrapped: `effective_timestep = tick % episode_length`
- This allows `resample_frequency > episode_length` while maintaining imitation learning

The MoE config uses `resample_frequency = 455` (5 episodes) to balance map loading overhead with scenario diversity. The last timestep of each episode (`tick % 91 == 90`) has no expert action available.

**Usage**:
```bash
# Enable imitation learning
puffer train puffer_drive --train.imit-coef 0.1

# MoE with imitation (enabled by default in config)
puffer train puffer_drive_moe
```

**Testing**:
```bash
# Run imitation learning tests (includes Multiprocessing backend tests)
python -m pytest tests/test_inverse_dynamics.py::TestMultiprocessingExpertActions -v

# Add -s flag to see print statements in test output
python -m pytest tests/test_inverse_dynamics.py::TestMultiprocessingExpertActions -v -s
```

### Evaluation (`pufferlib/ocean/benchmark/`)

WOSAC realism metrics and human-replay compatibility evaluation.

### Python Visualization (`pufferlib/visualize/`)

Matplotlib-based visualization utilities for creating publication-quality figures and videos. Inspired by GPUDrive visualization code.

**Modules**:
- **`color.py`**: Color schemes for roads, agents, and states (matching GPUDrive conventions)
- **`utils.py`**: Low-level drawing utilities (bounding boxes, trajectories, road polylines, figure-to-image conversion)
- **`core.py`**: `MatplotlibVisualizer` class for rendering simulator states
- **`multiverse.py`**: `MultiverseVisualizer` class for MoE expert comparison ("what-if" visualizations)

**Basic Usage**:
```python
from pufferlib.visualize import MatplotlibVisualizer, save_img_as_png

# Create visualizer
vis = MatplotlibVisualizer(env, goal_radius=2.0, figsize=(10, 10), dpi=100)

# Render current state
img = vis.plot_simulator_state(timestep=0, zoom_radius=80.0)
save_img_as_png(img, "state.png")

# Render with collision/offroad highlighting
img = vis.plot_simulator_state(
    timestep=0,
    collision_mask=collision_mask,
    offroad_mask=offroad_mask,
)

# Render with MoE expert coloring
img = vis.plot_simulator_state(
    timestep=0,
    policy_assignments=np.array([0, 1, 2, 0, 1, 2, ...]),  # Expert index per agent
)

# Plot ground truth human trajectories
img = vis.plot_ground_truth_trajectories(max_agents=16)

# 3D rendering mode
vis_3d = MatplotlibVisualizer(env, render_3d=True)
img = vis_3d.plot_simulator_state(timestep=0)
```

**MoE Multiverse Visualization**:
```python
from pufferlib.visualize import MultiverseVisualizer

# Create multiverse visualizer
moe_vis = MultiverseVisualizer(env, policy=None, num_experts=3)

# Compare trajectories from different experts in a grid
# trajectories_by_expert: Dict[int, Dict] with 'positions' key
img = moe_vis.plot_multiverse_grid(
    trajectories_by_expert=trajectories_by_expert,
    zoom_radius=50.0,
    title="Expert Comparison",
)

# Visualize expert routing probabilities over time
# expert_probs_history: (num_timesteps, num_agents, num_experts)
img = moe_vis.plot_expert_distribution_over_time(expert_probs_history, agent_indices=[0, 1])

# Color trajectory by active expert at each timestep
img = moe_vis.plot_trajectory_with_expert_coloring(
    positions=trajectory,  # (num_steps, 2)
    expert_assignments=assignments,  # (num_steps,) expert index per step
)
```

**Video/GIF Generation**:
```python
from pufferlib.visualize.utils import save_frames_as_gif, save_frames_as_video

# Collect frames during rollout
frames = []
for step in range(100):
    img = vis.plot_simulator_state(timestep=step)
    frames.append(img)
    obs, reward, done, truncated, info = env.step(action)

# Save as GIF (always works)
save_frames_as_gif(frames, "rollout.gif", fps=10)

# Save as MP4 (requires imageio[ffmpeg])
save_frames_as_video(frames, "rollout.mp4", fps=10)
```

**Testing**:
```bash
# Run visualization tests
python -m pytest tests/test_visualization.py -v

# Test specific components
python -m pytest tests/test_visualization.py::TestMatplotlibVisualizer -v
python -m pytest tests/test_visualization.py::TestMultiverseVisualizer -v
```

## Data

Map binaries are loaded from `map_dir` config setting. Convert JSON scenes to binary:
```bash
python pufferlib/ocean/drive/drive.py
```

Download WOMD data from HuggingFace: [GPUDrive](https://huggingface.co/datasets/EMERGE-lab/GPUDrive)

## When to Rebuild C Extensions

Run `python setup.py build_ext --inplace --force` after modifying:
- `pufferlib/ocean/drive/drive.c` or `drive.h`
- `pufferlib/ocean/drive/binding.c`
- `pufferlib/ocean/drive/visualize.c`

---

## Closed-Loop MTR-MoE: Technical Report

### 1. Introduction and Motivation

The Motion Transformer (MTR) architecture, introduced by Shi et al. (NeurIPS 2022), represents the state-of-the-art in open-loop trajectory prediction for autonomous driving. MTR employs a Transformer-based encoder-decoder architecture with learnable intention queries to capture multi-modal future trajectories. However, the original MTR formulation operates in an **open-loop** setting: given a fixed history of observations, it predicts a distribution over complete future trajectories without considering how predicted actions affect subsequent states.

This work adapts MTR for **closed-loop** reinforcement learning, where the agent must:
1. Execute actions that affect the environment state
2. Receive new observations conditioned on previous actions
3. Optimize long-horizon returns rather than single-step likelihood

We introduce two architectures:
- **DriveMTR**: A baseline adaptation of MTR for closed-loop RL
- **DriveMTRMoE**: An extension with Mixture-of-Experts for latent driving style modeling

### 2. Background: Standard MTR Architecture

The original MTR architecture consists of:

**Encoder**: A context encoder that processes agent trajectories and map polylines through separate PointNet-style networks, followed by Transformer self-attention layers that model interactions between all scene elements.

**Decoder**: A query-based Transformer decoder where $K$ learnable intention queries attend to encoded context via cross-attention. Each query specializes in a different motion mode (e.g., turning left, going straight, turning right).

**Output**: For each query $k \in \{1, ..., K\}$, the decoder outputs:
- A predicted trajectory $\hat{\tau}_k = \{(x_t, y_t)\}_{t=1}^{T}$
- A confidence score $c_k$ indicating the likelihood of that mode

**Training**: Standard MTR is trained with supervised learning on logged human trajectories using a winner-takes-all loss that backpropagates only through the query closest to ground truth.

### 3. Closed-Loop Adaptations

#### 3.1 MPC-Style Action Extraction

The fundamental challenge in adapting MTR for closed-loop control is converting trajectory predictions into executable actions. Standard MTR outputs position sequences, but the PufferDrive simulator requires discrete acceleration and steering commands.

We adopt an **MPC-style** (Model Predictive Control) approach: predict a full trajectory but execute only the first timestep's action, then re-plan at the next timestep with updated observations.

**GMM Action Head**: Rather than predicting positions directly, our decoder outputs a Gaussian Mixture Model (GMM) over actions at each timestep. Each query $k$ produces a 7-dimensional output per timestep:

$$\mathbf{o}_{k,t} = [\mu_a, \tilde{\mu}_\delta, \log\sigma_a, \log\sigma_\delta, \rho, v_x, v_y]$$

where:
- $\mu_a$: Mean acceleration (direct output)
- $\tilde{\mu}_\delta$: Raw steering value (before transformation)
- $\sigma_a, \sigma_\delta$: Standard deviations for acceleration and steering
- $\rho$: Correlation coefficient between acceleration and steering
- $v_x, v_y$: Predicted velocity components

**Steering Transformation**: The raw steering output is transformed to the valid range $[-\pi/3, \pi/3]$ (approximately $\pm 60°$):

$$\mu_\delta = \tanh(\tilde{\mu}_\delta) \cdot \frac{\pi}{3}$$

This bounded transformation ensures kinematically feasible steering angles.

#### 3.2 Query Selection for Action Execution

At inference time, we must select which query's prediction to execute. We introduce **motion classification heads** that output confidence scores $c_k$ for each query:

$$c_k = \text{MLP}(\mathbf{h}_k)$$

where $\mathbf{h}_k$ is the decoded feature for query $k$.

**Training**: We use soft selection via softmax-weighted combination:
$$\mathbf{a} = \sum_{k=1}^{K} \text{softmax}(c_k) \cdot \mathbf{o}_{k,0}[:2]$$

**Inference**: We use hard selection of the highest-confidence query:
$$\mathbf{a} = \mathbf{o}_{k^*,0}[:2], \quad k^* = \arg\max_k c_k$$

#### 3.3 Action Discretization

PufferDrive uses a discrete action space with 91 actions (7 acceleration levels × 13 steering levels). We convert continuous GMM outputs to discrete action logits:

$$\text{logit}_{i,j} = -\frac{1}{\tau}\left(|\mu_a - a_i| + |\bar{\mu}_\delta - \delta_j|\right)$$

where $a_i \in \{-4, -2, -1, 0, 1, 2, 4\}$ m/s² and $\delta_j \in [-1, 1]$ are the discrete action values, $\bar{\mu}_\delta = \mu_\delta / (\pi/3)$ normalizes steering to $[-1, 1]$, and $\tau$ is a temperature parameter.

The logits are flattened to a 91-dimensional vector for compatibility with PPO training.

#### 3.4 Kinematic Integration for Trajectory Supervision

While closed-loop execution uses only the first timestep, we retain trajectory prediction as an auxiliary objective. Predicted actions are integrated through a bicycle model to obtain trajectory positions:

**Velocity Integration**:
$$v_t = v_0 + \sum_{s=1}^{t} \mu_a^{(s)} \cdot \Delta t$$

**Heading Integration**:
$$\theta_t = \sum_{s=1}^{t} \frac{v_s \cdot \tan(\mu_\delta^{(s)})}{L} \cdot \Delta t$$

where $L$ is the vehicle wheelbase.

**Position Integration**:
$$x_t = \sum_{s=1}^{t} v_s \cos(\theta_s) \Delta t, \quad y_t = \sum_{s=1}^{t} v_s \sin(\theta_s) \Delta t$$

This kinematic integration guarantees that predicted trajectories are dynamically feasible, unlike position-based predictions that may violate vehicle dynamics.

### 4. DriveMTRMoE: Mixture-of-Experts Extension

#### 4.1 Architecture Overview

DriveMTRMoE extends DriveMTR with latent driving style modeling. The hypothesis is that human driving behavior exhibits distinct styles (e.g., aggressive, conservative, defensive) that manifest more clearly in closed-loop interaction than in open-loop prediction.

**Frozen Backbone**: The MTR encoder can optionally be frozen after pretraining, with only the router and LoRA adapters trained on the RL objective. This enables parameter-efficient fine-tuning.

**LoRA Expert Adapters**: Instead of separate expert networks, we use Low-Rank Adaptation (LoRA) on the actor head:

$$\mathbf{W}_{\text{expert}} = \mathbf{W}_{\text{base}} + \sum_{e=1}^{E} p_e \cdot \mathbf{A}_e \mathbf{B}_e$$

where $\mathbf{A}_e \in \mathbb{R}^{d \times r}$ and $\mathbf{B}_e \in \mathbb{R}^{r \times d}$ are low-rank matrices for expert $e$, $r \ll d$ is the LoRA rank, and $p_e$ is the routing probability for expert $e$.

#### 4.2 Social Forces Router

Expert routing is conditioned on **social forces**—a physics-inspired representation of interactions with neighboring agents:

$$\mathbf{f}_i = \sum_{j \neq i} \frac{\mathbf{r}_{ij}}{|\mathbf{r}_{ij}|^2} \cdot \mathbb{1}[|\mathbf{r}_{ij}| < R]$$

where $\mathbf{r}_{ij}$ is the relative position vector from agent $i$ to agent $j$, and $R$ is the interaction radius.

The router computes expert probabilities:
$$\mathbf{p} = \text{softmax}\left(\text{MLP}([\mathbf{h}; \mathbf{f}]) / \tau\right)$$

where $\mathbf{h}$ is the decoded context feature, $\mathbf{f}$ is the social force vector, and $\tau$ is the Gumbel-Softmax temperature (annealed during training).

#### 4.3 Reconstruction Loss

To encourage the router to capture meaningful scene structure rather than degenerate solutions, we include a **reconstruction loss** that requires the router's hidden representation to reconstruct the input social forces:

$$\mathcal{L}_{\text{recon}} = \|\hat{\mathbf{f}} - \mathbf{f}\|_2^2$$

where $\hat{\mathbf{f}} = \text{MLP}_{\text{recon}}(\mathbf{z})$ and $\mathbf{z}$ is the router's intermediate representation.

#### 4.4 Training Objectives

The total loss combines multiple objectives:

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + \lambda_{\text{imit}}\mathcal{L}_{\text{imit}} + \lambda_{\text{kl}}\mathcal{L}_{\text{kl}} + \lambda_{\text{ent}}\mathcal{L}_{\text{ent}} + \lambda_{\text{cos}}\mathcal{L}_{\text{cos}} + \lambda_{\text{recon}}\mathcal{L}_{\text{recon}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}}$$

where:
- $\mathcal{L}_{\text{PPO}}$: Proximal Policy Optimization loss
- $\mathcal{L}_{\text{imit}}$: Imitation loss (cross-entropy with expert actions)
- $\mathcal{L}_{\text{kl}}$: KL divergence encouraging uniform expert usage
- $\mathcal{L}_{\text{ent}}$: Negative entropy encouraging confident routing
- $\mathcal{L}_{\text{cos}}$: Cosine similarity penalty for expert weight diversity
- $\mathcal{L}_{\text{recon}}$: Social forces reconstruction loss
- $\mathcal{L}_{\text{div}}$: Output diversity loss encouraging different expert behaviors

### 5. Implementation Details

#### 5.1 Architecture Parameters

| Component | Parameter | DriveMTR (minimal) | DriveMTRMoE |
|-----------|-----------|----------|-------------|
| Encoder | d_model | 64 | 128 |
| Encoder | num_layers | 2 | 3 |
| Encoder | num_heads | 2 | 4 |
| Decoder | num_layers | 2 | 3 |
| Decoder | num_queries | 4 | 4 |
| Decoder | num_future_frames | 40 | 40 |
| MoE | num_experts | - | 3 |
| MoE | lora_rank | - | 8 |
| MoE | lora_alpha | - | 4.0 |

Three configuration tiers are available:
- **Original** (d_model=256, 6 layers): ~26M params, requires multi-GPU
- **Lite** (d_model=128, 3 layers): ~3.7M params, requires ~24GB VRAM
- **Minimal** (d_model=64, 2 layers): ~1M params, fits on 16GB GPU

#### 5.2 Key Differences from Standard MTR

| Aspect | Standard MTR | Closed-Loop MTR |
|--------|--------------|-----------------|
| **Training** | Supervised learning | Reinforcement learning (PPO) |
| **Output** | Position sequences | Action GMM + discretization |
| **Execution** | Full trajectory | First timestep only (MPC-style) |
| **Loss** | Winner-takes-all NLL | PPO + imitation + auxiliary |
| **Temporal context** | Fixed history window | Recurrent state (optional) |
| **Mode selection** | Post-hoc scoring | Learned confidence heads |

#### 5.3 File Structure

```
pufferlib/ocean/
├── torch_mtr.py          # DriveMTR and DriveMTRMoE policy classes
├── mtr_encoder.py        # MTREncoder, MTRDecoder, kinematic integration
├── moe_adapters.py       # LoRAExpertsRL, PersonaRouterWithReconstruction
└── inverse_dynamics.py   # Expert action computation for imitation

pufferlib/config/ocean/
├── puffer_drive_mtr.ini      # Baseline MTR configuration
└── puffer_drive_mtr_moe.ini  # MTR-MoE configuration
```

### 6. References

- Shi, S., et al. "Motion Transformer with Global Intention Localization and Local Movement Refinement." NeurIPS 2022.
- Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv 2017.
- Helbing, D., & Molnár, P. "Social Force Model for Pedestrian Dynamics." Physical Review E, 1995.
