# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
### This file mirrors file XXX within the Jumanji package.
### The _make_raw_env and setup_logger functions were adapted to allow usage with this repository

from typing import Tuple, Optional

import chex
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig


from routing_board_generation.rl_training.logging.pickle_logger import PickleLogger
from routing_board_generation.rl_training.offline_generation.dataset_generator_numpy import (
    BoardDatasetGenerator,
)
from routing_board_generation.rl_training.offline_generation.dataset_generator_jax import (
    BoardDatasetGeneratorJAX,
)
from routing_board_generation.rl_training.online_generators.parallel_random_walk_generator import (
    ParallelRandomWalkGenerator,
)
from routing_board_generation.rl_training.online_generators.random_seed_generator import (
    SeedExtensionGenerator,
)
from routing_board_generation.rl_training.online_generators.sequential_random_walk_generator import (
    SequentialRandomWalkGenerator,
)
from routing_board_generation.rl_training.online_generators.uniform_generator import (
    UniformRandomGenerator,
)
from routing_board_generation.interface.board_generator_interface import BoardName
from jumanji.env import Environment
from jumanji.environments import (
    CVRP,
    TSP,
    BinPack,
    Cleaner,
    Connector,
    Game2048,
    JobShop,
    Knapsack,
    Maze,
    Minesweeper,
    RubiksCube,
    Snake,
)
from jumanji.training import networks
from jumanji.training.agents.a2c import A2CAgent
from jumanji.training.agents.base import Agent
from jumanji.training.agents.random import RandomAgent
from jumanji.training.evaluator import Evaluator
from jumanji.training.loggers import (
    Logger,
    NeptuneLogger,
    TensorboardLogger,
    TerminalLogger,
)
from jumanji.training.networks.actor_critic import ActorCriticNetworks
from jumanji.training.networks.protocols import RandomPolicy
from jumanji.training.types import ActingState, TrainingState
from jumanji.wrappers import MultiToSingleWrapper, VmapAutoResetWrapper


def setup_logger(cfg: DictConfig) -> Logger:
    logger: Logger
    if cfg.logger.type == "tensorboard":
        logger = TensorboardLogger(
            name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint
        )
    elif cfg.logger.type == "neptune":
        logger = NeptuneLogger(
            name=cfg.logger.name,
            project="ic-instadeep/jumanji",
            cfg=cfg,
            save_checkpoint=cfg.logger.save_checkpoint,
        )
    elif cfg.logger.type == "terminal":
        logger = TerminalLogger(
            name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint
        )

    elif cfg.logger.type == "pickle":
        logger = PickleLogger(
            name=cfg.logger.name, save_checkpoint=cfg.logger.save_checkpoint
        )
    else:
        raise ValueError(
            f"logger expected in ['neptune', 'tensorboard', 'terminal'], got {cfg.logger}."
        )
    return logger


def _make_raw_env(
    cfg: DictConfig, ic_generator: Optional[BoardName] = None
) -> Environment:

    # env: Environment = jumanji.make(cfg.env.registered_version)
    if cfg.env.ic_board.generation_type == "offline":
        generator = BoardDatasetGenerator(
            grid_size=cfg.env.ic_board.grid_size,
            num_agents=cfg.env.ic_board.num_agents,
            board_generator=cfg.env.ic_board.board_name,
        )
    elif (
        cfg.env.ic_board.generation_type == "offline_seed_extension"
        or cfg.env.ic_board.generation_type == "offline_parallel_rw"
    ):
        generator = BoardDatasetGeneratorJAX(
            grid_size=cfg.env.ic_board.grid_size,
            num_agents=cfg.env.ic_board.num_agents,
            randomness=cfg.env.seed_extension.randomness,
            two_sided=cfg.env.seed_extension.two_sided,
            extension_iterations=cfg.env.seed_extension.extension_iterations,
            extension_steps=cfg.env.seed_extension.extension_steps,
            board_name=cfg.env.ic_board.generation_type,
            number_of_boards=cfg.env.seed_extension.number_of_boards,
        )
    elif cfg.env.ic_board.generation_type == "online_uniform":
        generator = UniformRandomGenerator(
            grid_size=cfg.env.ic_board.grid_size,
            num_agents=cfg.env.ic_board.num_agents,
        )
    elif cfg.env.ic_board.generation_type == "online_seq_rw":
        generator = SequentialRandomWalkGenerator(
            grid_size=cfg.env.ic_board.grid_size,
            num_agents=cfg.env.ic_board.num_agents,
        )

    elif cfg.env.ic_board.generation_type == "online_random_seed":
        generator = SeedExtensionGenerator(
            grid_size=cfg.env.ic_board.grid_size,
            num_agents=cfg.env.ic_board.num_agents,
        )
    elif cfg.env.ic_board.generation_type == "online_parallel_rw":
        generator = ParallelRandomWalkGenerator(
            grid_size=cfg.env.ic_board.grid_size,
            num_agents=cfg.env.ic_board.num_agents,
        )
    elif cfg.env.ic_board.generation_type == "online_lsystems":
        raise NotImplementedError
    else:
        raise ValueError("Your connector.yml parameters do not exist")

    env = Connector(generator=generator)
    if isinstance(env, Connector):
        env = MultiToSingleWrapper(env)
    return env


def setup_env(cfg: DictConfig) -> Environment:
    env = _make_raw_env(cfg)
    env = VmapAutoResetWrapper(env)
    return env


def setup_agent(cfg: DictConfig, env: Environment) -> Agent:
    agent: Agent
    if cfg.agent == "random":
        random_policy = _setup_random_policy(cfg, env)
        agent = RandomAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            random_policy=random_policy,
        )
    elif cfg.agent == "a2c":
        actor_critic_networks = _setup_actor_critic_neworks(cfg, env)
        optimizer = optax.adam(cfg.env.a2c.learning_rate)
        agent = A2CAgent(
            env=env,
            n_steps=cfg.env.training.n_steps,
            total_batch_size=cfg.env.training.total_batch_size,
            actor_critic_networks=actor_critic_networks,
            optimizer=optimizer,
            normalize_advantage=cfg.env.a2c.normalize_advantage,
            discount_factor=cfg.env.a2c.discount_factor,
            bootstrapping_factor=cfg.env.a2c.bootstrapping_factor,
            l_pg=cfg.env.a2c.l_pg,
            l_td=cfg.env.a2c.l_td,
            l_en=cfg.env.a2c.l_en,
        )
    else:
        raise ValueError(
            f"Expected agent name to be in ['random', 'a2c'], got {cfg.agent}."
        )
    return agent


def _setup_random_policy(  # noqa: CCR001
    cfg: DictConfig, env: Environment
) -> RandomPolicy:
    assert cfg.agent == "random"
    if cfg.env.name == "bin_pack":
        assert isinstance(env.unwrapped, BinPack)
        random_policy = networks.make_random_policy_bin_pack(bin_pack=env.unwrapped)
    elif cfg.env.name == "snake":
        assert isinstance(env.unwrapped, Snake)
        random_policy = networks.make_random_policy_snake()
    elif cfg.env.name == "tsp":
        assert isinstance(env.unwrapped, TSP)
        random_policy = networks.make_random_policy_tsp()
    elif cfg.env.name == "knapsack":
        assert isinstance(env.unwrapped, Knapsack)
        random_policy = networks.make_random_policy_knapsack()
    elif cfg.env.name == "job_shop":
        assert isinstance(env.unwrapped, JobShop)
        random_policy = networks.make_random_policy_job_shop()
    elif cfg.env.name == "cvrp":
        assert isinstance(env.unwrapped, CVRP)
        random_policy = networks.make_random_policy_cvrp()
    elif cfg.env.name == "rubiks_cube":
        assert isinstance(env.unwrapped, RubiksCube)
        random_policy = networks.make_random_policy_rubiks_cube(
            rubiks_cube=env.unwrapped
        )
    elif cfg.env.name == "minesweeper":
        assert isinstance(env.unwrapped, Minesweeper)
        random_policy = networks.make_random_policy_minesweeper(
            minesweeper=env.unwrapped
        )
    elif cfg.env.name == "game_2048":
        assert isinstance(env.unwrapped, Game2048)
        random_policy = networks.make_random_policy_game_2048()
    elif cfg.env.name == "cleaner":
        assert isinstance(env.unwrapped, Cleaner)
        random_policy = networks.make_random_policy_cleaner()
    elif cfg.env.name == "maze":
        assert isinstance(env.unwrapped, Maze)
        random_policy = networks.make_random_policy_maze()
    elif cfg.env.name == "connector":
        assert isinstance(env.unwrapped, Connector)
        random_policy = networks.make_random_policy_connector()
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return random_policy


def _setup_actor_critic_neworks(  # noqa: CCR001
    cfg: DictConfig, env: Environment
) -> ActorCriticNetworks:
    assert cfg.agent == "a2c"
    if cfg.env.name == "bin_pack":
        assert isinstance(env.unwrapped, BinPack)
        actor_critic_networks = networks.make_actor_critic_networks_bin_pack(
            bin_pack=env.unwrapped,
            num_transformer_layers=cfg.env.network.num_transformer_layers,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "snake":
        assert isinstance(env.unwrapped, Snake)
        actor_critic_networks = networks.make_actor_critic_networks_snake(
            snake=env.unwrapped,
            num_channels=cfg.env.network.num_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "tsp":
        assert isinstance(env.unwrapped, TSP)
        actor_critic_networks = networks.make_actor_critic_networks_tsp(
            tsp=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
            mean_cities_in_query=cfg.env.network.mean_cities_in_query,
        )
    elif cfg.env.name == "knapsack":
        assert isinstance(env.unwrapped, Knapsack)
        actor_critic_networks = networks.make_actor_critic_networks_knapsack(
            knapsack=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "job_shop":
        assert isinstance(env.unwrapped, JobShop)
        actor_critic_networks = networks.make_actor_critic_networks_job_shop(
            job_shop=env.unwrapped,
            num_layers_machines=cfg.env.network.num_layers_machines,
            num_layers_operations=cfg.env.network.num_layers_operations,
            num_layers_joint_machines_jobs=cfg.env.network.num_layers_joint_machines_jobs,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
        )
    elif cfg.env.name == "cvrp":
        assert isinstance(env.unwrapped, CVRP)
        actor_critic_networks = networks.make_actor_critic_networks_cvrp(
            cvrp=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
            mean_nodes_in_query=cfg.env.network.mean_nodes_in_query,
        )
    elif cfg.env.name == "game_2048":
        assert isinstance(env.unwrapped, Game2048)
        actor_critic_networks = networks.make_actor_critic_networks_game_2048(
            game_2048=env.unwrapped,
            num_channels=cfg.env.network.num_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "rubiks_cube":
        assert isinstance(env.unwrapped, RubiksCube)
        actor_critic_networks = networks.make_actor_critic_networks_rubiks_cube(
            rubiks_cube=env.unwrapped,
            cube_embed_dim=cfg.env.network.cube_embed_dim,
            step_count_embed_dim=cfg.env.network.step_count_embed_dim,
            dense_layer_dims=cfg.env.network.dense_layer_dims,
        )
    elif cfg.env.name == "minesweeper":
        assert isinstance(env.unwrapped, Minesweeper)
        actor_critic_networks = networks.make_actor_critic_networks_minesweeper(
            minesweeper=env.unwrapped,
            board_embed_dim=cfg.env.network.board_embed_dim,
            board_conv_channels=cfg.env.network.board_conv_channels,
            board_kernel_shape=cfg.env.network.board_kernel_shape,
            num_mines_embed_dim=cfg.env.network.num_mines_embed_dim,
            final_layer_dims=cfg.env.network.final_layer_dims,
        )
    elif cfg.env.name == "maze":
        assert isinstance(env.unwrapped, Maze)
        actor_critic_networks = networks.make_actor_critic_networks_maze(
            maze=env.unwrapped,
            num_channels=cfg.env.network.num_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "cleaner":
        assert isinstance(env.unwrapped, Cleaner)
        actor_critic_networks = networks.make_actor_critic_networks_cleaner(
            cleaner=env.unwrapped,
            num_conv_channels=cfg.env.network.num_conv_channels,
            policy_layers=cfg.env.network.policy_layers,
            value_layers=cfg.env.network.value_layers,
        )
    elif cfg.env.name == "connector":
        assert isinstance(env.unwrapped, Connector)
        actor_critic_networks = networks.make_actor_critic_networks_connector(
            connector=env.unwrapped,
            transformer_num_blocks=cfg.env.network.transformer_num_blocks,
            transformer_num_heads=cfg.env.network.transformer_num_heads,
            transformer_key_size=cfg.env.network.transformer_key_size,
            transformer_mlp_units=cfg.env.network.transformer_mlp_units,
            conv_n_channels=cfg.env.network.conv_n_channels,
        )
    else:
        raise ValueError(f"Environment name not found. Got {cfg.env.name}.")
    return actor_critic_networks


def setup_evaluators(cfg: DictConfig, agent: Agent) -> Tuple[Evaluator, Evaluator]:
    env = _make_raw_env(cfg)
    stochastic_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.eval_total_batch_size,
        stochastic=True,
    )
    greedy_eval = Evaluator(
        eval_env=env,
        agent=agent,
        total_batch_size=cfg.env.evaluation.greedy_eval_total_batch_size,
        stochastic=False,
    )
    return stochastic_eval, greedy_eval


def setup_training_state(
    env: Environment, agent: Agent, key: chex.PRNGKey
) -> TrainingState:
    params_key, reset_key, acting_key = jax.random.split(key, 3)

    # Initialize params.
    params_state = agent.init_params(params_key)

    # Initialize environment states.
    num_devices = jax.local_device_count()
    reset_keys = jax.random.split(reset_key, agent.total_batch_size).reshape(
        (num_devices, agent.batch_size_per_device, -1)
    )
    env_state, timestep = jax.pmap(env.reset, axis_name="devices")(reset_keys)

    # Initialize acting states.
    acting_key_per_device = jax.random.split(acting_key, num_devices)
    acting_state = ActingState(
        state=env_state,
        timestep=timestep,
        key=acting_key_per_device,
        episode_count=jnp.zeros(num_devices, float),
        env_step_count=jnp.zeros(num_devices, float),
    )

    # Build the training state.
    training_state = TrainingState(
        params_state=jax.device_put_replicated(params_state, jax.local_devices()),
        acting_state=acting_state,
    )
    return training_state
