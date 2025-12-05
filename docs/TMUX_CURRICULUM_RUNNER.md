# Tmux Curriculum Training Runners

Two tmux-based runners are available for scheduling curriculum training experiments across GPU panels.

## Quick Start

### 1. Quick Test Runner (Recommended for testing)
Runs 3 experiments with reduced timesteps:

```bash
./scripts/training/run_curriculum_quick.sh
```

This creates a tmux session with:
- **Experiment 0**: Uniform strategy (3-5 trucks, 3-5 stops)
- **Experiment 1**: Staged curriculum
- **Experiment 2**: Mixed difficulties
- **Monitor window**: GPU monitoring with `nvidia-smi`

Settings: 500K timesteps, smaller batches, quick evaluation intervals.

### 2. Full Runner (Production)
Runs multiple combinations of strategies and ranges:

```bash
./scripts/training/run_curriculum_tmux.sh
```

This creates experiments for:
- 3 strategies × 2 truck ranges × 2 stop ranges = 12 experiments
- Each experiment runs for 2M timesteps
- Automatically distributes across available GPUs

## Usage

### Basic Commands

```bash
# Start quick test (3 experiments, attaches to session)
./scripts/training/run_curriculum_quick.sh

# Start quick test detached (runs in background)
./scripts/training/run_curriculum_quick.sh -d

# Start full runner
./scripts/training/run_curriculum_tmux.sh

# Start full runner detached
./scripts/training/run_curriculum_tmux.sh -d

# Specify number of GPUs
./scripts/training/run_curriculum_tmux.sh --gpus 4

# Preview commands without running
./scripts/training/run_curriculum_tmux.sh --dry-run
```

### Tmux Session Management

```bash
# List all tmux sessions
tmux ls

# Attach to quick test session
tmux attach -t curriculum_quick

# Attach to full session
tmux attach -t curriculum_training

# Detach from session (while attached)
Ctrl-b d

# Kill a session
tmux kill-session -t curriculum_quick
tmux kill-session -t curriculum_training
```

### Navigation Inside Tmux

**Switch between windows:**
- `Ctrl-b 0` - Experiments window
- `Ctrl-b 1` - Monitor window (nvidia-smi)
- `Ctrl-b 2` - Logs window (full runner only)

**Navigate between panes:**
- `Ctrl-b ←/→/↑/↓` - Move between panes
- `Ctrl-b q` - Show pane numbers
- `Ctrl-b z` - Zoom current pane (toggle fullscreen)
- `Ctrl-b x` - Kill current pane

**Other useful commands:**
- `Ctrl-b [` - Enter scroll mode (use arrow keys, q to exit)
- `Ctrl-b :` - Enter command mode

## Experiment Configuration

### Quick Runner (`run_curriculum_quick.sh`)

| Experiment | Strategy | Truck Range | Stop Range | Timesteps |
|------------|----------|-------------|------------|-----------|
| 0 | Uniform | 3-5 | 3-5 | 500K |
| 1 | Staged | Config file | Config file | 500K |
| 2 | Mixed | Config file | Config file | 500K |

### Full Runner (`run_curriculum_tmux.sh`)

Generates 12 experiments from combinations:
- **Strategies**: uniform, staged, mixed
- **Truck ranges**: 3-8, 5-10
- **Stop ranges**: 3-8, 5-10
- **Timesteps**: 2M per experiment

Each experiment gets a unique seed (42, 43, 44, ...) and GPU assignment (round-robin).

## GPU Assignment

Experiments are automatically distributed across available GPUs using round-robin:

```
GPU 0: Exp 0, 4, 8, ...
GPU 1: Exp 1, 5, 9, ...
GPU 2: Exp 2, 6, 10, ...
GPU 3: Exp 3, 7, 11, ...
```

Set number of GPUs:
```bash
./scripts/training/run_curriculum_tmux.sh --gpus 4
```

## Monitoring Progress

### Inside the Tmux Session

1. **Switch to monitor window**: `Ctrl-b 1`
   - Shows real-time GPU usage via `nvidia-smi`

2. **Switch to experiments window**: `Ctrl-b 0`
   - Navigate between panes to see training logs
   - Each pane shows one experiment's progress

3. **Zoom a pane**: `Ctrl-b z`
   - Focus on single experiment
   - Toggle back with `Ctrl-b z` again

### Outside the Tmux Session

```bash
# Check GPU usage
nvidia-smi

# View saved models
ls -lh saved_models/curriculum_*/

# Follow wandb link
# Each experiment logs to wandb project: evpr-curriculum or evpr-curriculum-quick
```

## Customization

### Modify Quick Runner

Edit `scripts/training/run_curriculum_quick.sh`:
- Change timesteps: `--max-timesteps 500000`
- Change ranges: `--truck-range 3 5`
- Change evaluation: `--eval-freq 2000`

### Modify Full Runner

Edit `scripts/training/run_curriculum_tmux.sh`:
- Change strategies: `STRATEGIES=("uniform" "staged")`
- Change ranges: `TRUCK_RANGES=("3 8" "5 12")`
- Change base seed: `BASE_SEED=100`

### Create Custom Runner

Copy and modify one of the scripts:
```bash
cp scripts/training/run_curriculum_quick.sh scripts/training/my_custom_runner.sh
# Edit my_custom_runner.sh
chmod +x scripts/training/my_custom_runner.sh
./scripts/training/my_custom_runner.sh
```

## Troubleshooting

### Session already exists
```bash
# Kill the existing session first
tmux kill-session -t curriculum_quick
# Or attach to it
tmux attach -t curriculum_quick
```

### Virtual environment not found
The scripts look for `.venv` in the project root. If not found, experiments will run without activating venv. Ensure your environment is set up:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### GPU not found
Check GPU availability:
```bash
nvidia-smi
```

If no GPU available, the scripts will still run but on CPU (much slower).

### Experiment crashed
1. Attach to session: `tmux attach -t curriculum_quick`
2. Navigate to crashed pane
3. Check error message
4. Fix and re-run command manually or restart session

### Want to stop all experiments
```bash
# Kill the entire tmux session
tmux kill-session -t curriculum_quick
```

## Example Workflow

### 1. Start Quick Test
```bash
# Run quick test to validate setup
./scripts/training/run_curriculum_quick.sh -d

# Check it's running
tmux ls

# Attach and monitor
tmux attach -t curriculum_quick

# Switch to monitor window to see GPU usage
Ctrl-b 1

# Switch back to experiments
Ctrl-b 0

# Navigate between panes
Ctrl-b arrow keys

# Detach when satisfied
Ctrl-b d
```

### 2. Run Full Training
```bash
# After quick test succeeds, run full training
./scripts/training/run_curriculum_tmux.sh --gpus 4 -d

# Monitor progress occasionally
tmux attach -t curriculum_training

# Or check nvidia-smi
watch -n 5 nvidia-smi
```

### 3. Check Results
```bash
# View saved models
ls -lh saved_models/curriculum_*/

# Check wandb dashboard
# Navigate to: https://wandb.ai/stavrosorf/evpr-curriculum
```

## Advanced: Manual Session Creation

If you prefer manual control:

```bash
# Create session
tmux new -s my_session

# Split panes
Ctrl-b %    # Vertical split
Ctrl-b "    # Horizontal split

# In each pane, run:
cd /home/sorfanouda/EVPR
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_curriculum.py [args]
```

## Files

- `scripts/training/run_curriculum_tmux.sh` - Full runner with 12 experiments
- `scripts/training/run_curriculum_quick.sh` - Quick runner with 3 experiments
- This file: `docs/TMUX_CURRICULUM_RUNNER.md`

## See Also

- Main curriculum guide: `docs/CURRICULUM_LEARNING.md`
- Implementation details: `CURRICULUM_IMPLEMENTATION.md`
- Training script: `scripts/training/train_curriculum.py`
