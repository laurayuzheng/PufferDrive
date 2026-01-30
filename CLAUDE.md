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
puffer train puffer_drive --wandb --wandb-project pufferdrive_goal --exp-name baseline

# MoE training (latent variable modeling)
puffer train puffer_drive_moe --wandb --wandb-project pufferdrive --exp-name polysona

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
puffer eval puffer_drive_moe  --eval.wosac-realism-eval True --load-model-path experiments/puffer_drive_moe_05ekubwk.pt
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
python plt_visualize.py --mode multiverse -o viz_output/multiverse.gif  --pov --forced-agent 1 --scenario 10 --config puffer_drive_moe --batch-scenarios 0-100 --checkpoint experiments/puffer_drive_moe_05ekubwk.pt
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

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + \lambda_{\text{imit}}\mathcal{L}_{\text{imit}} + \lambda_{\text{traj}}\mathcal{L}_{\text{traj}} + \lambda_{\text{kl}}\mathcal{L}_{\text{kl}} + \lambda_{\text{ent}}\mathcal{L}_{\text{ent}} + \lambda_{\text{cos}}\mathcal{L}_{\text{cos}} + \lambda_{\text{recon}}\mathcal{L}_{\text{recon}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}}$$

where:
- $\mathcal{L}_{\text{PPO}}$: Proximal Policy Optimization loss
- $\mathcal{L}_{\text{imit}}$: Imitation loss (cross-entropy with expert actions)
- $\mathcal{L}_{\text{traj}}$: Trajectory prediction loss (GMM NLL on full trajectory)
- $\mathcal{L}_{\text{kl}}$: KL divergence encouraging uniform expert usage
- $\mathcal{L}_{\text{ent}}$: Negative entropy encouraging confident routing
- $\mathcal{L}_{\text{cos}}$: Cosine similarity penalty for expert weight diversity
- $\mathcal{L}_{\text{recon}}$: Social forces reconstruction loss
- $\mathcal{L}_{\text{div}}$: Output diversity loss encouraging different expert behaviors

#### 4.5 Trajectory Prediction Loss

The trajectory loss supervises the full multi-step trajectory prediction using GMM negative log-likelihood, following the original MTR formulation. For each predicted trajectory mode $k$:

$$\mathcal{L}_{\text{traj}} = -\log \mathcal{N}(\mathbf{x}_t^{\text{gt}} | \boldsymbol{\mu}_{k^*,t}, \boldsymbol{\Sigma}_{k^*,t})$$

where $k^* = \arg\min_k \|\boldsymbol{\mu}_{k,T} - \mathbf{x}_T^{\text{gt}}\|$ (winner-takes-all selection) and:

$$\boldsymbol{\Sigma}_{k,t} = \begin{pmatrix} \sigma_x^2 & \rho\sigma_x\sigma_y \\ \rho\sigma_x\sigma_y & \sigma_y^2 \end{pmatrix}$$

The bivariate Gaussian NLL per timestep is:
$$-\log \mathcal{N} = \log\sigma_x + \log\sigma_y + \frac{1}{2}\log(1-\rho^2) + \frac{1}{2(1-\rho^2)}\left[\frac{\Delta x^2}{\sigma_x^2} + \frac{\Delta y^2}{\sigma_y^2} - \frac{2\rho \Delta x \Delta y}{\sigma_x\sigma_y}\right]$$

The loss is summed over all $T=40$ future timesteps.

#### 4.6 Loss Coefficient Tuning

Critical insight: the GMM trajectory loss is summed over 40 timesteps, making its raw magnitude **much larger** than other losses:

| Loss | Typical Magnitude | Coefficient | Contribution |
|------|-------------------|-------------|--------------|
| PPO policy gradient | 0.01 - 0.1 | 1.0 | 0.01 - 0.1 |
| Value loss | 0.1 - 1.0 | 2.0 | 0.2 - 2.0 |
| Entropy | 1 - 3 | 0.005 | 0.005 - 0.015 |
| Imitation (CE) | 1 - 4 | 0.05 | 0.05 - 0.2 |
| **Trajectory (GMM NLL)** | **40 - 160** | **0.01** | **0.4 - 1.6** |
| MoE auxiliary | ~1 | 0.01 - 0.1 | 0.01 - 0.1 |

With `traj_loss_coef = 0.01`, the trajectory loss contributes ~0.4-1.6 to the total loss, comparable to the value loss and ensuring PPO remains the dominant learning signal.

### 5. Implementation Details

#### 5.1 Architecture Parameters

| Component | Parameter | DriveMTR | DriveMTRMoE |
|-----------|-----------|----------|-------------|
| Encoder | d_model | 64 | 64 |
| Encoder | num_layers | 2 | 2 |
| Encoder | num_heads | 2 | 2 |
| Decoder | num_layers | 2 | 2 |
| Decoder | num_queries | 4 | 4 |
| Decoder | num_future_frames | 40 | 40 |
| MoE | num_experts | - | 3 |
| MoE | lora_rank | - | 8 |
| MoE | lora_alpha | - | 8.0 |

Three configuration tiers are available:
- **Original** (d_model=256, 6 layers): ~26M params, requires multi-GPU
- **Lite** (d_model=128, 3 layers): ~3.7M params, requires ~24GB VRAM
- **Minimal** (d_model=64, 2 layers): ~1M params, fits on 16GB GPU (current default)

#### 5.2 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `imit_coef` | 0.05 | Imitation learning coefficient |
| `traj_loss_coef` | 0.01 | Trajectory GMM NLL coefficient |
| `aux_kl_coef` | 0.1 | Expert routing KL divergence |
| `aux_entropy_coef` | 0.01 | Router entropy penalty |
| `aux_cosine_coef` | 0.01 | Expert weight diversity |
| `aux_reconstruction_coef` | 1.0 | Social forces reconstruction |
| `aux_output_div_coef` | 0.01 | Expert output diversity |
| `learning_rate` | 0.0003 | Adam learning rate |
| `batch_size` | 262144 | Total batch size |
| `minibatch_size` | 2048 | Minibatch for gradient updates |

#### 5.3 Key Differences from Standard MTR

| Aspect | Standard MTR | Closed-Loop MTR |
|--------|--------------|-----------------|
| **Training** | Supervised learning | Reinforcement learning (PPO) |
| **Output** | Position sequences | Action GMM + discretization |
| **Execution** | Full trajectory | First timestep only (MPC-style) |
| **Loss** | Winner-takes-all NLL | PPO + imitation + trajectory + auxiliary |
| **Temporal context** | Fixed history window | Per-step re-encoding |
| **Mode selection** | Post-hoc scoring | Learned confidence heads |

#### 5.4 File Structure

```
pufferlib/ocean/
├── torch_mtr.py          # DriveMTR and DriveMTRMoE policy classes
├── mtr_encoder.py        # MTREncoder, MTRDecoder, kinematic integration
├── moe_adapters.py       # LoRAExpertsRL, PersonaRouterWithReconstruction
└── inverse_dynamics.py   # Expert action computation for imitation

pufferlib/ocean/drive/
└── drive.py              # get_future_trajectory_at_timestep() for trajectory loss

pufferlib/
├── pufferl.py            # Training loop with trajectory loss integration
└── vector.py             # get_future_trajectories() for vectorized envs

pufferlib/config/ocean/
├── puffer_drive_mtr.ini      # Baseline MTR configuration
└── puffer_drive_mtr_moe.ini  # MTR-MoE configuration
```

#### 5.5 Trajectory Loss Implementation

Ground truth future trajectories are obtained per-timestep during rollouts and transformed to ego-centric (local) coordinates:

1. **Data collection** (`drive.py:get_future_trajectory_at_timestep`):
   - Get current ego position $(x_0, y_0)$ and heading $\theta$
   - Get GT future positions $(x_t, y_t)$ for $t \in [1, T]$
   - Transform to local: $x'_t = \cos(-\theta)(x_t - x_0) - \sin(-\theta)(y_t - y_0)$

2. **Buffer storage** (`pufferl.py`):
   - `future_traj`: (segments, horizon, T, 2) - local coordinates
   - `future_valid`: (segments, horizon, T) - validity mask

3. **Loss computation** (`torch_mtr.py:get_trajectory_loss`):
   - Integrate predicted actions to trajectory via bicycle model
   - Winner-takes-all: select mode closest to GT endpoint
   - Compute bivariate Gaussian NLL summed over timesteps

**Note**: Currently only supported with Serial backend. Multiprocessing backend returns empty arrays (trajectory loss skipped).

### 6. References

- Shi, S., et al. "Motion Transformer with Global Intention Localization and Local Movement Refinement." NeurIPS 2022.
- Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv 2017.
- Helbing, D., & Molnár, P. "Social Force Model for Pedestrian Dynamics." Physical Review E, 1995.

---

## Closed-Loop Drive and DriveMoE: Technical Report

### Abstract

We present Drive and DriveMoE, lightweight neural network policies for closed-loop autonomous driving control within the PufferDrive simulator. Drive is a baseline MLP-based architecture that processes multi-modal observations (ego state, road geometry, neighboring agents) through parallel encoders with permutation-invariant pooling. DriveMoE extends this baseline with a Mixture-of-Experts (MoE) framework for latent driving style modeling, employing Low-Rank Adaptation (LoRA) for parameter-efficient expert specialization and a physics-informed social forces router for context-aware expert selection. We describe the architectural design, training objectives, and auxiliary regularization techniques that enable stable multi-expert learning under reinforcement learning.

### 1. Introduction

Autonomous driving policies must process complex, variable-size observations and produce control actions in real-time. While Transformer-based architectures such as MTR (Shi et al., 2022) achieve state-of-the-art performance on trajectory prediction benchmarks, their computational overhead limits applicability in high-throughput simulation settings where millions of environment steps are required for policy optimization.

We propose Drive, a computationally efficient policy architecture that achieves competitive performance with significantly reduced parameter count (~70K vs. 1M+ for Transformer variants). The architecture employs separate encoders for each observation modality with max-pooling aggregation, enabling processing of variable numbers of road segments and neighboring agents.

We further propose DriveMoE, an extension that models latent driving styles through a Mixture-of-Experts framework. Rather than learning a single averaged policy, DriveMoE learns $K$ specialized expert policies and a router that selects among them based on the driving context. This enables:

1. **Style-consistent behavior**: Agents maintain coherent driving patterns across time
2. **Multi-modal action distributions**: Different experts can specialize in different maneuvers
3. **Interpretable decomposition**: Expert assignments provide insight into decision-making

### 2. Problem Formulation

We formulate closed-loop driving as a Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ where:

- **State space** $\mathcal{S}$: Observations comprising ego state $\mathbf{s}^{\text{ego}} \in \mathbb{R}^{d_e}$, road geometry $\mathbf{S}^{\text{road}} \in \mathbb{R}^{N_r \times d_r}$, and partner agents $\mathbf{S}^{\text{partner}} \in \mathbb{R}^{N_p \times d_p}$
- **Action space** $\mathcal{A}$: Discrete actions $a = (a^{\text{accel}}, a^{\text{steer}}) \in \{1, \ldots, 7\} \times \{1, \ldots, 13\}$
- **Transition dynamics** $P$: Governed by bicycle model physics in Box2D
- **Reward function** $R$: Sparse goal-reaching reward with collision penalties
- **Discount factor** $\gamma = 0.98$

The objective is to learn a policy $\pi_\theta(a|s)$ that maximizes expected discounted return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t)\right]$$

### 3. Drive Architecture

#### 3.1 Observation Encoding

The observation vector is partitioned into three modalities, each processed by a dedicated encoder.

**Ego Encoder.** The ego state vector $\mathbf{s}^{\text{ego}} \in \mathbb{R}^{d_e}$ contains kinematic information:

$$\mathbf{s}^{\text{ego}} = [x, y, v_x, v_y, \cos\theta, \sin\theta, v, \ldots]^\top$$

where $d_e = 7$ for classic dynamics or $d_e = 10$ for jerk-based dynamics. The encoder applies a two-layer MLP with layer normalization:

$$\mathbf{h}^{\text{ego}} = f_{\text{ego}}(\mathbf{s}^{\text{ego}}) = \mathbf{W}_2^{\text{ego}} \cdot \text{LN}(\mathbf{W}_1^{\text{ego}} \mathbf{s}^{\text{ego}} + \mathbf{b}_1^{\text{ego}}) + \mathbf{b}_2^{\text{ego}}$$

where $\mathbf{W}_1^{\text{ego}} \in \mathbb{R}^{d_h \times d_e}$, $\mathbf{W}_2^{\text{ego}} \in \mathbb{R}^{d_h \times d_h}$, and $d_h$ is the hidden dimension.

**Partner Encoder.** Neighboring agents are represented as a set $\mathbf{S}^{\text{partner}} = \{\mathbf{s}_i^{\text{partner}}\}_{i=1}^{N_p}$ where each element contains:

$$\mathbf{s}_i^{\text{partner}} = [\Delta x_i, \Delta y_i, w_i, l_i, \cos\psi_i, \sin\psi_i, v_i]^\top \in \mathbb{R}^{7}$$

The encoder processes each agent independently and aggregates via max-pooling for permutation invariance:

$$\mathbf{h}^{\text{partner}} = \max_{i \in \{1, \ldots, N_p\}} f_{\text{partner}}(\mathbf{s}_i^{\text{partner}})$$

**Road Encoder.** Road segments are represented as $\mathbf{S}^{\text{road}} = \{\mathbf{s}_j^{\text{road}}\}_{j=1}^{N_r}$ with continuous features (position, orientation) and a categorical road type. The categorical feature is one-hot encoded:

$$\tilde{\mathbf{s}}_j^{\text{road}} = [\mathbf{s}_{j,1:d_r-1}^{\text{road}}, \text{onehot}(s_{j,d_r}^{\text{road}})]^\top \in \mathbb{R}^{d_r + 6}$$

The encoder follows the same structure as the partner encoder:

$$\mathbf{h}^{\text{road}} = \max_{j \in \{1, \ldots, N_r\}} f_{\text{road}}(\tilde{\mathbf{s}}_j^{\text{road}})$$

#### 3.2 Feature Fusion

The modality-specific features are concatenated and projected through a shared embedding layer:

$$\mathbf{z} = \text{GELU}\left(\mathbf{W}^{\text{emb}} [\mathbf{h}^{\text{ego}}; \mathbf{h}^{\text{road}}; \mathbf{h}^{\text{partner}}] + \mathbf{b}^{\text{emb}}\right)$$

where $\mathbf{W}^{\text{emb}} \in \mathbb{R}^{d_z \times 3d_h}$ and $\mathbf{z} \in \mathbb{R}^{d_z}$ is the fused representation.

#### 3.3 Actor-Critic Heads

**Actor.** The policy head outputs logits for each action dimension:

$$\boldsymbol{\ell} = \mathbf{W}^{\text{actor}} \mathbf{z} + \mathbf{b}^{\text{actor}}, \quad \boldsymbol{\ell} \in \mathbb{R}^{|\mathcal{A}_{\text{accel}}| + |\mathcal{A}_{\text{steer}}|}$$

The logits are split into acceleration and steering components:

$$\pi(a^{\text{accel}}|s) = \text{softmax}(\boldsymbol{\ell}_{1:7}), \quad \pi(a^{\text{steer}}|s) = \text{softmax}(\boldsymbol{\ell}_{8:20})$$

**Critic.** The value function is a linear projection:

$$V(s) = \mathbf{w}^{\text{value}} \cdot \mathbf{z} + b^{\text{value}}$$

#### 3.4 Temporal Modeling

For settings requiring temporal context, the policy is wrapped with an LSTM layer that maintains hidden states $(\mathbf{h}_t, \mathbf{c}_t)$ across timesteps:

$$\mathbf{h}_t, \mathbf{c}_t = \text{LSTM}(\mathbf{z}_t, \mathbf{h}_{t-1}, \mathbf{c}_{t-1})$$

The LSTM output replaces $\mathbf{z}$ in subsequent actor-critic computations.

### 4. DriveMoE: Mixture-of-Experts Extension

#### 4.1 Low-Rank Expert Adaptation

Full expert networks with separate parameters per expert are prohibitively expensive. Following Hu et al. (2022), we employ Low-Rank Adaptation (LoRA) to parameterize expert-specific modifications to a shared base network.

For the actor layer with base weights $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, we introduce $K$ expert-specific low-rank decompositions:

$$\Delta\mathbf{W}_k = \mathbf{B}_k \mathbf{A}_k, \quad \mathbf{A}_k \in \mathbb{R}^{r \times d_{\text{in}}}, \quad \mathbf{B}_k \in \mathbb{R}^{d_{\text{out}} \times r}$$

where $r \ll \min(d_{\text{in}}, d_{\text{out}})$ is the rank hyperparameter.

Given routing weights $\mathbf{p} = [p_1, \ldots, p_K]^\top$ from the router, the effective weight matrix is:

$$\mathbf{W}_{\text{eff}} = \mathbf{W} + \frac{\alpha}{r} \sum_{k=1}^{K} p_k \cdot \mathbf{B}_k \mathbf{A}_k$$

where $\alpha$ is a scaling hyperparameter. The forward pass computes:

$$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b} + \frac{\alpha}{r} \sum_{k=1}^{K} p_k \cdot \mathbf{B}_k (\mathbf{A}_k \mathbf{x})$$

This formulation mixes expert weights before computation (the "expert" variant from Polysona), which is more parameter-efficient than mixing outputs post-computation.

#### 4.2 Expert Routing

**Learned Router.** The PersonaRouter computes routing probabilities from the concatenated features:

$$\mathbf{p} = \text{softmax}\left(\mathbf{W}_2^{\text{router}} \cdot \text{ReLU}\left(\text{LN}(\mathbf{W}_1^{\text{router}} \mathbf{h}^{\text{concat}})\right)\right)$$

where $\mathbf{h}^{\text{concat}} = [\mathbf{h}^{\text{ego}}; \mathbf{h}^{\text{road}}; \mathbf{h}^{\text{partner}}]$.

**Social Forces Router.** Inspired by pedestrian dynamics (Helbing & Molnár, 1995), we propose a physics-informed router based on social forces—repulsive interactions between the ego agent and neighbors.

For each neighbor $i$ at relative position $\mathbf{r}_i = (\Delta x_i, \Delta y_i)$, the repulsive force magnitude is:

$$F_i = A \cdot \exp\left(\frac{D - \|\mathbf{r}_i\|}{B}\right)$$

where $A$, $B$, $D$ are force parameters. The force vector is:

$$\mathbf{f}_i = F_i \cdot \frac{-\mathbf{r}_i}{\|\mathbf{r}_i\|}$$

We aggregate force statistics into a feature vector:

$$\boldsymbol{\phi} = \left[\sum_i f_{i,x}, \sum_i f_{i,y}, \bar{F}, \max_i F_i, \sigma_F, \frac{|\{i : F_i > 0\}|}{N_p}\right]^\top \in \mathbb{R}^6$$

The router optionally fuses social forces with learned context features:

$$\mathbf{p} = \text{softmax}\left(g_{\text{router}}([\boldsymbol{\phi}; \mathbf{h}^{\text{concat}}])\right)$$

or uses social forces alone when `social_forces_only=True`.

#### 4.3 Training Mode vs. Inference Mode

During training, we use soft routing with softmax probabilities to enable gradient flow through all experts. During inference, we use hard routing via argmax selection:

$$\text{Training: } \mathbf{p} = \text{softmax}(\boldsymbol{\ell}^{\text{router}}), \quad \text{Inference: } \mathbf{p} = \text{onehot}(\arg\max_k \ell_k^{\text{router}})$$

### 5. Training Objectives

#### 5.1 Primary Objective: PPO

We optimize the policy using Proximal Policy Optimization (Schulman et al., 2017). The clipped surrogate objective is:

$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the importance ratio and $\hat{A}_t$ is the GAE advantage estimate.

The value function loss is:

$$\mathcal{L}^{\text{value}}(\theta) = \mathbb{E}_t\left[(V_\theta(s_t) - V_t^{\text{target}})^2\right]$$

#### 5.2 Imitation Learning Auxiliary Loss

To accelerate learning and improve sample efficiency, we incorporate behavioral cloning from human demonstrations. Expert actions $a^*$ are computed via inverse dynamics from logged human trajectories.

The imitation loss is the cross-entropy between policy logits and expert actions:

$$\mathcal{L}^{\text{imit}}(\theta) = -\mathbb{E}_{(s, a^*) \in \mathcal{D}}\left[\log \pi_\theta(a^*|s)\right]$$

where $\mathcal{D}$ is the set of valid demonstration samples.

#### 5.3 MoE Auxiliary Losses

Training MoE models presents challenges including mode collapse (all inputs routed to one expert) and expert redundancy (experts learn identical behaviors). We employ several regularization techniques.

**KL Divergence Loss.** Encourages uniform expert utilization across the batch:

$$\mathcal{L}^{\text{KL}} = D_{\text{KL}}\left(\bar{\mathbf{p}} \| \mathbf{u}\right) = \sum_{k=1}^{K} \bar{p}_k \log\frac{\bar{p}_k}{1/K}$$

where $\bar{\mathbf{p}} = \frac{1}{B}\sum_{i=1}^{B} \mathbf{p}^{(i)}$ is the batch-averaged routing distribution and $\mathbf{u}$ is the uniform prior.

**Negative Entropy Loss.** Encourages confident (low-entropy) routing decisions:

$$\mathcal{L}^{\text{ent}} = \mathbb{E}\left[\sum_{k=1}^{K} p_k \log p_k\right]$$

Minimizing this loss pushes routing probabilities toward one-hot vectors.

**Expert Weight Diversity Loss.** Penalizes similarity between expert weight modifications:

$$\mathcal{L}^{\text{cos}} = \frac{1}{K(K-1)} \sum_{k \neq k'} \cos^2(\text{vec}(\Delta\mathbf{W}_k), \text{vec}(\Delta\mathbf{W}_{k'}))$$

where $\Delta\mathbf{W}_k = \mathbf{B}_k \mathbf{A}_k$ and $\cos(\cdot, \cdot)$ is cosine similarity.

**Output Diversity Loss.** Encourages experts to produce different action distributions:

$$\mathcal{L}^{\text{div}} = -\frac{1}{K(K-1)} \sum_{k < k'} D_{\text{JS}}(\pi_k(\cdot|s) \| \pi_{k'}(\cdot|s))$$

where $\pi_k$ is the policy using only expert $k$ and $D_{\text{JS}}$ is the Jensen-Shannon divergence.

#### 5.4 Total Loss

The complete training objective is:

$$\mathcal{L}(\theta) = \mathcal{L}^{\text{PPO}} + \lambda_v \mathcal{L}^{\text{value}} + \lambda_{\text{imit}} \mathcal{L}^{\text{imit}} + \lambda_{\text{KL}} \mathcal{L}^{\text{KL}} + \lambda_{\text{ent}} \mathcal{L}^{\text{ent}} + \lambda_{\text{cos}} \mathcal{L}^{\text{cos}} + \lambda_{\text{div}} \mathcal{L}^{\text{div}}$$

### 6. Implementation Details

#### 6.1 Architecture Hyperparameters

| Symbol | Parameter | Drive | DriveMoE |
|--------|-----------|-------|----------|
| $d_h$ | Encoder hidden dim | 64 | 64 |
| $d_z$ | Embedding dim | 256 | 256 |
| $K$ | Number of experts | — | 3 |
| $r$ | LoRA rank | — | 8 |
| $\alpha$ | LoRA scaling | — | 8.0 |

#### 6.2 Training Hyperparameters

| Symbol | Parameter | Value |
|--------|-----------|-------|
| $\gamma$ | Discount factor | 0.98 |
| $\lambda_{\text{GAE}}$ | GAE lambda | 0.95 |
| $\epsilon$ | PPO clip coefficient | 0.2 |
| $\lambda_v$ | Value loss coefficient | 2.0 |
| $\lambda_{\text{imit}}$ | Imitation coefficient | 0.05 |
| $\lambda_{\text{KL}}$ | KL divergence coefficient | 100 |
| $\lambda_{\text{ent}}$ | Entropy coefficient | 0.01 |
| $\lambda_{\text{cos}}$ | Cosine similarity coefficient | 0.1 |
| $\lambda_{\text{div}}$ | Output diversity coefficient | 0.01 |
| — | Learning rate | $3 \times 10^{-4}$ |
| — | Batch size | 524,288 |
| — | Minibatch size | 4,096 |

#### 6.3 Temperature Annealing

The router temperature $\tau$ is linearly annealed during training:

$$\tau(t) = \tau_{\max} - (\tau_{\max} - \tau_{\min}) \cdot \frac{t}{T}$$

where $\tau_{\max} = 2.0$, $\tau_{\min} = 0.1$, and $T$ is total training steps. Higher temperature produces softer routing early in training; lower temperature approaches hard selection.

#### 6.4 Parameter Counts

| Component | Parameters | Trainable (DriveMoE) |
|-----------|------------|----------------------|
| Ego encoder | 4,736 | Frozen |
| Road encoder | 5,184 | Frozen |
| Partner encoder | 4,736 | Frozen |
| Shared embedding | 49,408 | Frozen |
| Router | 16,451 | Yes |
| Actor (base) | 5,140 | Frozen |
| Actor (LoRA) | 4,160 | Yes |
| Value function | 257 | Yes |
| **Total** | **89,872** | **20,868** |

With LSTM wrapper, total parameters increase to approximately 635K.

### 7. Comparison with Transformer-Based Approaches

| Aspect | Drive/DriveMoE | DriveMTR/DriveMTRMoE |
|--------|----------------|----------------------|
| Encoder architecture | MLP + max-pool | Transformer self-attention |
| Temporal modeling | External LSTM | Internal self-attention |
| Computational complexity | $O(N)$ | $O(N^2)$ |
| Parameter count | ~70K | ~1M+ |
| Multi-modal prediction | No (single policy) | Yes ($K$ intention queries) |
| Trajectory supervision | No | Yes (GMM NLL loss) |

The Drive architecture trades representational capacity for computational efficiency, making it suitable for high-throughput simulation where millions of environment interactions are required.

### 8. References

1. Helbing, D., & Molnár, P. (1995). Social force model for pedestrian dynamics. *Physical Review E*, 51(5), 4282.

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.

3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

4. Shi, S., Jiang, L., Dai, D., & Schiele, B. (2022). Motion transformer with global intention localization and local movement refinement. *NeurIPS*.
