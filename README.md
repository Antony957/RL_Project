# Data-Efficient Reinforcement Learning with Self-Predictive Representations

*Max Schwarzer\*, Ankesh Anand\*, Rishab Goel, R Devon Hjelm, Aaron Courville, Philip Bachman*

This repo provides code for implementing the [SPR paper](https://arxiv.org/abs/2007.05929)

* [📦 Install ](#install) -- Install relevant dependencies and the project
* [🔧 Usage ](#usage) -- Commands to run different experiments from the paper

## Install 
To install the requirements, follow these steps:
```bash
# Clone the project
git clone https://github.com/mila-iqia/spr

# Install the requirements
bash setup.sh
conda activate spr
AutoROM --accept-license --install-dir $(python -c "import atari_py; import os; print(os.path.join(os.path.dirname(atari_py.__file__), 'atari_roms'))")

```

## Usage:
The default branch for the latest and stable changes is `release`. 

* To run SPR with augmentation
```bash
CUDA_VISIBLE_DEVICES=7 python -m scripts.run --public --game pong --momentum-tau 1.
```

* To run SPR without augmentation
```bash
python -m scripts.run --public --game pong --augmentation none --target-augmentation 0 --momentum-tau 0.01 --dropout 0.5
```

When reporting scores, we average across 10 seeds. 

## What does each file do? 

    .
    ├── scripts
    │   └── run.py                # The main runner script to launch jobs.
    ├── src                     
    │   ├── agent.py              # Implements the Agent API for action selection 
    │   ├── algos.py              # Distributional RL loss
    │   ├── models.py             # Network architecture and forward passes.
    │   ├── rlpyt_atari_env.py    # Slightly modified Atari env from rlpyt
    │   ├── rlpyt_utils.py        # Utility methods that we use to extend rlpyt's functionality
    │   └── utils.py              # Command line arguments and helper functions 
    │
    └── requirements.txt          # Dependencies
