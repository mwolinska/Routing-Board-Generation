from typing import Optional

import chex
import jax
from jax import numpy as jnp

from routing_board_generation.benchmarking.utils.benchmark_data_model import \
    BoardGenerationParameters
from routing_board_generation.rl_training.offline_generation.generation_utils import \
    generate_n_boards
from routing_board_generation.rl_training.online_generators.uniform_generator import \
    Generator

from routing_board_generation.interface.board_generator_interface import \
    BoardName
from jumanji.environments.routing.connector import State
from jumanji.environments.routing.connector.types import Agent


class BoardDatasetGenerator(Generator):
    def __init__(self, grid_size: int, num_agents: int,
                 board_generator: Optional[BoardName] = None) -> None:
        super().__init__(grid_size, num_agents)
        print(board_generator)
        generation_params = BoardGenerationParameters(
            columns=grid_size,
            rows=grid_size,
            number_of_wires=num_agents,
            generator_type=board_generator,
        )
        boards, heads, targets = generate_n_boards(generation_params, number_of_boards=10)

        self.board_dataset = jnp.array(boards)
        self.heads = jnp.array(heads)
        self.targets = jnp.array(targets)

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout."""
        key, pos_key = jax.random.split(key)
        which_board = jax.random.randint(key, shape=(), minval=0, maxval=len(self.board_dataset))
        grid = self.board_dataset[which_board]

        starts_flat = self.heads[which_board]
        targets_flat = self.targets[which_board]

        starts = jnp.divmod(starts_flat, self.grid_size)
        targets = jnp.divmod(targets_flat, self.grid_size)

        # Create the agent pytree that corresponds to the grid.
        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1))

        step_count = jnp.array(0, jnp.int32)
        return State(key=key, grid=grid, step_count=step_count, agents=agents)
