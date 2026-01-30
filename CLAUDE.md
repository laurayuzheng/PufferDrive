# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PufferDrive is a fast driving simulator for training and testing RL-based models, built on the PufferLib reinforcement learning framework. It features a high-performance C-based simulation engine with Python bindings, supporting up to 512+ agents and integration with Waymo driving data.

**Docs**: https://emerge-lab.github.io/PufferDrive

## Build & Development Commands

### Initial Setup
```bash
# Create and activate virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Build C extensions (required after any C code changes)
python setup.py build_ext --inplace --force
```

### Debug Build
```bash
DEBUG=1 python setup.py build_ext --inplace --force
```

### Build Visualizer
```bash
# Local build with sanitizers (debug)
bash scripts/build_ocean.sh drive local

# Optimized build
bash scripts/build_ocean.sh drive fast

# WebAssembly build (requires emscripten)
bash scripts/build_ocean.sh drive web
```

### Training
```bash
# Start training
puffer train puffer_drive

# Distributed training
torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_drive

# Training with WandB
puffer train puffer_drive --wandb --wandb-project polysona_rl --exp-name baseline_ppo
```

### Testing
```bash
# C INI parser tests
./tests/ini_parser/build_n_test.sh

# Python config tests
python tests/test_drive_config.py

# Training integration test
python tests/test_drive_train.py
```

### Evaluation
```bash
# WOSAC distributional realism benchmark
puffer eval puffer_drive --eval.wosac-realism-eval True

# With checkpoint
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path <path>.pt

# Human-compatibility evaluation
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path <path>.pt
```

### Code Quality
```bash
pre-commit run --all-files  # Runs ruff-format, clang-format, trailing whitespace fixes
```

## Architecture

### Core Components

- **`pufferlib/pufferl.py`** - Main training loop (`PuffeRL` class), handles distributed RL training with vectorized environments
- **`pufferlib/pufferlib.py`** - Base `PufferEnv` class and vectorization API
- **`pufferlib/models.py`** - Default PyTorch policies with encoder-decoder architecture
- **`pufferlib/vector.py`** - Vectorized environment handling (`Serial` backend)
- **`pufferlib/emulation.py`** - Compatibility layer for Gymnasium/Gym/PettingZoo environments

### Drive Simulator (`pufferlib/ocean/drive/`)

- **`drive.py`** - Python `Drive` environment class extending `PufferEnv`
- **`drive.c`/`drive.h`** - Core C simulation logic
- **`binding.c`** - C-Python bindings
- **`visualize.c`** - Raylib visualization

### Build System

The `setup.py` handles complex build logic:
- Downloads external dependencies (raylib 5.5, box2d, inih) automatically
- Builds C/C++ extensions with platform-specific flags
- Supports CUDA extensions (auto-detected via `which nvcc`)

**Environment variables:**
- `DEBUG=1` - Enable debug symbols and sanitizers
- `NO_OCEAN=1` - Skip ocean environment building
- `NO_TRAIN=1` - Skip training extension building

### Configuration

- **`ruff.toml`** - Python linting: line-length 120, target Python 3.10
- **`.pre-commit-config.yaml`** - Pre-commit hooks for formatting
- **`.clang-format`** - C/C++ style (LLVM-based, 4-space indent, 120 column limit)

## Key Patterns

### Environment Interface
Environments extend `pufferlib.PufferEnv` and define:
- `single_observation_space` / `single_action_space` - Per-agent spaces
- `reset()` / `step()` - Standard gym interface
- C bindings in `binding.c` expose simulation functions to Python

### Policy Architecture
The `Default` model in `models.py` uses an encoder-decoder structure:
- `encode()` - Process observations into hidden state
- `decode()` - Output action logits and value estimate

### Data Format
Drive environment uses binary map files converted from JSON. The `drive.py` module includes conversion utilities. Compatible data sources:
- Waymo Open Motion Dataset (WOMD)
- ScenarioMax format
