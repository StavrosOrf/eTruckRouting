#!/bin/bash
# Quick reference for tmux curriculum runners

cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TMUX CURRICULUM TRAINING - QUICK REFERENCE                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📦 TWO RUNNERS AVAILABLE:

1. QUICK TEST (3 experiments, 500K steps)
   ./scripts/training/run_curriculum_quick.sh

2. FULL TRAINING (12 experiments, 2M steps)
   ./scripts/training/run_curriculum_tmux.sh


🚀 QUICK START:

# Run quick test
./scripts/training/run_curriculum_quick.sh

# Run in background (detached)
./scripts/training/run_curriculum_quick.sh -d

# Preview without running
./scripts/training/run_curriculum_tmux.sh --dry-run

# Use multiple GPUs
./scripts/training/run_curriculum_tmux.sh --gpus 4


📋 TMUX COMMANDS:

# Attach to session
tmux attach -t curriculum_quick         # Quick test session
tmux attach -t curriculum_training      # Full training session

# List sessions
tmux ls

# Kill session
tmux kill-session -t curriculum_quick


⌨️  INSIDE TMUX (while attached):

Navigation:
  Ctrl-b 0/1/2       Switch windows (experiments/monitor/logs)
  Ctrl-b arrow keys  Move between panes
  Ctrl-b q           Show pane numbers
  Ctrl-b z           Zoom current pane (toggle)
  Ctrl-b d           Detach from session

Scrolling:
  Ctrl-b [           Enter scroll mode
  arrow keys         Scroll up/down
  q                  Exit scroll mode

Pane management:
  Ctrl-b %           Split vertically
  Ctrl-b "           Split horizontally
  Ctrl-b x           Kill current pane


📊 MONITORING:

# Inside tmux - switch to monitor window
Ctrl-b 1

# Outside tmux - check GPUs
nvidia-smi
watch -n 5 nvidia-smi

# Check saved models
ls -lh saved_models/curriculum_*/

# View wandb
https://wandb.ai/stavrosorf/evpr-curriculum


🎯 QUICK TEST EXPERIMENTS:

Panel 0: Uniform (3-5 trucks, 3-5 stops, 500K steps)
Panel 1: Staged curriculum (500K steps)
Panel 2: Mixed difficulties (500K steps)


🎯 FULL TRAINING EXPERIMENTS:

12 experiments = 3 strategies × 2 truck ranges × 2 stop ranges
- Strategies: uniform, staged, mixed
- Truck ranges: 3-8, 5-10
- Stop ranges: 3-8, 5-10
- Each: 2M timesteps


⚙️  CUSTOMIZE:

Edit the scripts to change:
- Timesteps: --max-timesteps
- Ranges: --truck-range, --stop-range
- Evaluation: --eval-freq, --eval-episodes
- Network: --gnn-hidden-dim, --mlp-hidden-dim


📚 FULL DOCUMENTATION:

docs/TMUX_CURRICULUM_RUNNER.md      Detailed tmux guide
docs/CURRICULUM_LEARNING.md         Complete curriculum guide
CURRICULUM_IMPLEMENTATION.md        Implementation summary


🆘 TROUBLESHOOTING:

Session already exists:
  tmux kill-session -t curriculum_quick

Experiment crashed:
  tmux attach -t curriculum_quick
  Navigate to pane, check error, restart manually

Stop all experiments:
  tmux kill-session -t curriculum_quick


✨ EXAMPLE WORKFLOW:

1. Quick test
   ./scripts/training/run_curriculum_quick.sh -d

2. Monitor progress
   tmux attach -t curriculum_quick
   Ctrl-b 1  (switch to GPU monitor)
   Ctrl-b 0  (back to experiments)
   Ctrl-b d  (detach)

3. After success, run full training
   ./scripts/training/run_curriculum_tmux.sh --gpus 4 -d

4. Check results
   ls -lh saved_models/
   # Check wandb dashboard


════════════════════════════════════════════════════════════════════════════════
EOF
