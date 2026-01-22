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

puffer train puffer_drive_moe --wandb --wandb-project pufferdrive

# WOSAC realism evaluation
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path <checkpoint.pt>

# Human-replay evaluation
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path <checkpoint.pt>

# MoE WOSAC eval
puffer eval puffer_drive_moe --eval.wosac-realism-eval True --load-model-path experiments/puffer_drive_moe_beswxlse.pt
```

Results

Metric	MoE (new)	MoE (old)	Baseline
ADE	10.26	18.53	7.12
Collisions	0.56	1.95	0.26
Realism	0.746	0.734	0.768


## Testing

```bash
# Run Python tests
python tests/test_drive_config.py
python tests/test_drive_train.py

# Run C INI parser tests
./tests/ini_parser/build_n_test.sh
```

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
