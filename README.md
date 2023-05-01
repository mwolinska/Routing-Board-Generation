<!-- [![Python Versions](https://img.shields.io/pypi/pyversions/jumanji.svg?style=flat-square)](https://www.python.org/doc/versions/) -->
<!-- [![PyPI Version](https://badge.fury.io/py/jumanji.svg)](https://badge.fury.io/py/jumanji) -->
<!-- [![Tests](https://github.com/instadeepai/jumanji/actions/workflows/tests_linters.yml/badge.svg)](https://github.com/instadeepai/jumanji/actions/workflows/tests_linters.yml) -->
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MyPy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

## Smart Level Generation for the Routing Problem

This is the submission repository for the MSc AI 2022/23 Group Project at Imperial College London. The project is a collaboration between Imperial College London and InstaDeep, a London-based AI company. The project is supervised by Dr Rob Craven (Imperial College London) and Clément Bonnet (InstaDeep).

## Project Description

The project aims to develop solvable boards for the routing (deemed Connector) problem. The routing problem is a well-known combinatorial optimisation problem that consists of finding the optimal path between two points in a graph. This specification considers 2D grids with $n$ start and end points which need to be routed without any route cross-over. As an extension to InstaDeep's Deep-Learning [Jumanji library](https://github.com/instadeepai/jumanji), generating diverse, difficult, and quick boards guaranteeing solvability is advantageous for deep RL training algorithms. A variety of approaches were therefore explored, both in NumPy and JAX.

## Boards available

- BFS
- LSystems
- Numberlink
- RandomWalk
- Wave Function Collapse

## Installation and Usage

Standalone, this repo can generate solved and unsolved $m \times m$ boards with $n$ agents. Refer to the [Jumanji Connector](https://instadeepai.github.io/jumanji/environments/connector/) environment for board and wire specs.


```bash
# clone repo and install dependencies
git clone https://github.com/mwolinska/Routing-Board-Generation.git
pip install -r requirements.txt
```

Example (NumPy) BFS Board Generation:
```python
from ic_routing_board_generation.board_generator.numpy_board_generation import bfs_board
board = bfs_board.BFSBoard(rows=5, cols=5, num_agents=3)
print(board.return_solved_board())
print(board.return_training_board())
```

Output:
```
[[0 5 0 0 0]
 [6 4 0 8 7]
 [0 0 0 0 7]
 [3 1 1 2 7]
 [0 0 0 0 9]]

[[0 5 0 0 0]
 [6 0 0 8 0]
 [0 0 0 0 0]
 [3 0 0 2 0]
 [0 0 0 0 9]]
```

To use in tandem with Jumanji's RL environments:

```bash
# install Jumanji
pip install jumanji (or for the latest, git+https://github.com/instadeepai/jumanji.git)
```

Usage: (see `agent_training/configs/env/connector.yaml` and `agent_training/configs/config.yaml` for more options)

```bash
python3 agent_training/training_script.py
# env.ic_board.generation_type=seq_parallel_rw env.ic_board.board_name=none can be appended as arguments for different board generation types
```

### Trained agents

Agents trained on the various types of boards were saved as pickle files in the jumanji_routing/examples directory.  They could be loaded into a jupyter notebook (jumanji_routing/load_checkpoints.ipynb), via code similar to:

```python
file = "examples/trained_agent_10x10_5_uniform/19-27-36/training_state_10x10_5_uniform"
with open(file,"rb") as f:
    training_state = pickle.load(f)
```
    
After loading, the agents were tested on boards generated by all the various generators, including Jumanji's initial "uniform" generator.  The load_checkpoints script was also used as a repository for storing notes and old test results over the weeks of testing.
