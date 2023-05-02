<!-- [![Python Versions](https://img.shields.io/pypi/pyversions/jumanji.svg?style=flat-square)](https://www.python.org/doc/versions/) -->
<!-- [![PyPI Version](https://badge.fury.io/py/jumanji.svg)](https://badge.fury.io/py/jumanji) -->
<!-- [![Tests](https://github.com/instadeepai/jumanji/actions/workflows/tests_linters.yml/badge.svg)](https://github.com/instadeepai/jumanji/actions/workflows/tests_linters.yml) -->
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[//]: # ([![MyPy]&#40;http://www.mypy-lang.org/static/mypy_badge.svg&#41;]&#40;http://mypy-lang.org/&#41;)

# Smart Level Generation for the Routing Problem
This is the submission repository for the MSc AI 2022/23 Group Project at Imperial College London. 
The project was a collaboration between Imperial College London and InstaDeep, a London-based AI company. 
The project was supervised by Dr Rob Craven (Imperial College London) and Clément Bonnet (InstaDeep).

The development team was formed of: Ugo Okoroafor, Randy Brown, Ole Jorgensen, Danila Kurganov and Marta Wolinska. 

## Project Description

The project aims to develop solvable boards for the routing problem. 
The routing problem is a well-known combinatorial optimisation problem that consists of finding the optimal path between two points in a graph. 
This specification considers 2D grids with $n$ start and end points which need to be routed without any route cross-over. 
As an extension to InstaDeep's Deep-Learning [Jumanji library](https://github.com/instadeepai/jumanji), 
generating diverse and difficult boards quickly guaranteeing solvability is advantageous for deep RL training algorithms. 
A variety of approaches were therefore explored, both in NumPy and JAX.
The report submitted as part of this project is available here and contains a review of the generators available, 
their evaluation as well as other results and discussion.

### Boards Available 
#### NumPy
- BreadthFirstSearch (BFS) (with children methods that remove and replace wires based on heuristics) 
- LSystems
- Numberlink
- RandomWalk
- Wave Function Collapse

The NumberLink and Wave Function Collapse implementations are based on existing repositories: [NumberLink by thomasahle](https://github.com/thomasahle/numberlink) 
and [wave function collapse by ikarth](https://github.com/ikarth/wfc_2019f) that carry the AGPL and MIT licenses respectively.

#### JAX
- ParallelRandomWalk (Production ready - pull request pending in Jumanji from original project repo)
- SeedExtension
- SequentialRandomWalk

#### Use of Jumanji
Although the majority of this repository was implemented from scratch, some methods were copied from Jumanji for ease of use
or adapted to suit the particular use case. 
In cases where code was adapted from Jumanji implementations this is marked with the following at the head of the relevant file:
```python
### This file contains the following methods adapted from implementations in Jumanji:
# LIST OF METHODS / CLASSES AS RELEVANT
###
```

In cases where code was directly copied from Jumanji for ease of use within the repository this is marked with:
```python
### This file mirrors file XXX within the Jumanji package. 
```

## Installation and Usage

Standalone, this repo can generate solved and unsolved $m \times m$ boards with $n$ agents. 
Refer to the [Jumanji Connector](https://instadeepai.github.io/jumanji/environments/connector/) environment for board and wire specs.

### Installation
To install the package run:
```shell 
git clone https://github.com/mwolinska/Routing-Board-Generation.git
cd Routing-Board-Generation
```
Then, set up and activate a virtual environment and install dependencies:
```shell
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage
The board generated in this repository follow the rules as set out in [Jumanji Connector](https://instadeepai.github.io/jumanji/environments/connector/) environment.
This repo provides the following features:
1. Generation of solved / unsolved $m \times m$ boards with $n$ agents,
2. Benchmarking generators against each other using static methods and random agents,
3. Framework to use these generators to train Jumanji's reinforcement learning agent,
4. Visualisation of specific boards (or visualising samples based on all existing generators),
5. Evaluating trained agents on board generated using any generator.

#### 1. Generation of solved / unsolved $m \times m$ boards with $n$ agents (NumPy)
To use a single generator, this can be initialised as below for the BFS Generator. 

```python
from routing_board_generation.board_generation_methods.numpy_implementation.board_generation import bfs_board

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
#### 2. Benchmarking generators against each other using static methods and random agents
Two scripts are prepared to run these activities. 
They are set by default to run all board generators and to not save any of the results.
If a different configuration is desired this can e amended in the scripts directly.

To generate results, which summarise metrics based on solved boards run:

```shell
python3 -m routing_board_generation/benchmarking/scripts/run_benchmark_on_generated_board MARTA
```

To generate results, which summarise performance on a random agent run:
```shell
python3 -m routing_board_generation/benchmarking/scripts/run_benchmark_with_agent MARTA
```

#### 3. Framework to use these generators to train Jumanji's reinforcement learning agent
To launch a training simply run:
```bash
cd agent_training
python3 training_script.py
# env.ic_board.generation_type=seq_parallel_rw env.ic_board.board_name=none can be appended as arguments for different board generation types
```
Configuration can be changed from within the configuration files:
(see `agent_training/configs/env/connector.yaml` and `agent_training/configs/config.yaml` for more options).
or directly from the command line thanks to the use of [hydra](https://hydra.cc/docs/intro/), per the pattern within 
`agent_training/routing_a2c.sh` and as explained further in the associated README. 
This file also contains the guide to running jobs on Imperial's SLURM cluster. 


#### 4. Visualisation of specific boards (or visualising samples based on all existing generators)
Visualisation instructions are provided within `routing_board_generation/visualisation/render_board.py`.

#### 5. Evaluating trained agents on board generated using any generator.

Agents trained on the various types of boards were saved as pickle files in the `Routing-Board-Generation/examples` directory.  
They could be loaded into a jupyter notebook (`jumanji_routing/load_checkpoints.ipynb`), via code similar to:

```python
import pickle
file = "examples/trained_agent_10x10_5_uniform/19-27-36/training_state_10x10_5_uniform"
with open(file,"rb") as f:
    training_state = pickle.load(f)
```
    
After loading, the agents were tested on boards generated by all the various generators, including Jumanji's initial `UniformRandomGenerator`.  
The `load_checkpoints` script was also used as a repository for storing notes and old test results over the weeks of testing.

### Miscellaneous Functionalities
#### Timing Jax Generators
Timing JAX generators can be done using `package_valuation/profiling_generators.ipynb`

#### Adding Your Own Generator in Numpy
To add your own generator in NumPy you should inherit from `AbstractBoard`, then add your generator to the interface in 
`routing_board_generation/interface/board_generator_interface.py`.
