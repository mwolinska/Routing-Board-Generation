from typing import Optional

import chex
import jax
from jax import numpy as jnp
from jumanji.environments.routing.connector.utils import get_position, get_target

from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters
from ic_routing_board_generation.board_generator.jax_board_generation.board_generator_random_seed_rb import \
    RandomSeedBoard
from ic_routing_board_generation.ic_rl_training.offline_generation.generation_utils import \
    generate_n_boards
from ic_routing_board_generation.ic_rl_training.online_generators.uniform_generator import \
    Generator

from ic_routing_board_generation.interface.board_generator_interface_numpy import \
    BoardName, BoardGenerator
from jumanji.environments.routing.connector import State
from jumanji.environments.routing.connector.types import Agent


class BoardDatasetGeneratorJAX(Generator):
    def __init__(self, grid_size: int, num_agents: int,
                 randomness: float = 1,
                 two_sided: bool = False,
                 extension_iterations: int = 1,
                 extension_steps: int = 1e23) -> None:
        super().__init__(grid_size, num_agents)

        self.board_generator = RandomSeedBoard(grid_size, grid_size, num_agents)
        self.board_generator_call = jax.jit(self.board_generator.generate_starts_ends)
        self.randomness = randomness
        self.two_sided = two_sided
        self.extension_iterations = extension_iterations
        self.extension_steps = extension_steps
        heads, targets = self.generate_n_boards(jax.random.PRNGKey(0), 10)

        self.heads = jnp.array(heads)
        self.targets = jnp.array(targets)

    def generate_n_boards(self, key: chex.PRNGKey, n_boards: int = 10):
        heads_list = []
        targets_list = []
        for i in range(n_boards):
            old_key, new_key = jax.random.split(key)

            heads_for_board, targets_for_board = \
                self.board_generator_call(new_key, self.randomness, self.two_sided,
                                          self.extension_iterations, self.extension_steps,
                                          )
            heads_list.append(heads_for_board)
            targets_list.append(targets_for_board)

            key = new_key
        return heads_list, targets_list

    def __call__(self, key: chex.PRNGKey) -> State:
        """Generates a `Connector` state that contains the grid and the agents' layout."""
        key, pos_key = jax.random.split(key)
        which_board = jax.random.randint(key, shape=(), minval=0, maxval=len(self.heads))

        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        starts = tuple(self.heads[which_board])
        targets = tuple(self.targets[which_board])

        agent_position_values = jax.vmap(get_position)(
            jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)

        # Create the agent pytree that corresponds to the grid.

        agents = jax.vmap(Agent)(
            id=jnp.arange(self.num_agents),
            start=jnp.stack(starts, axis=1),
            target=jnp.stack(targets, axis=1),
            position=jnp.stack(starts, axis=1),
        )


        step_count = jnp.array(0, jnp.int32)

        return State(key=key, grid=grid, step_count=step_count, agents=agents)

    def print_board(self, which_board: int):
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        starts = tuple(self.heads[which_board])
        targets = tuple(self.targets[which_board])

        agent_position_values = jax.vmap(get_position)(
            jnp.arange(self.num_agents))
        agent_target_values = jax.vmap(get_target)(jnp.arange(self.num_agents))

        # Transpose the agent_position_values to match the shape of the grid.
        # Place the agent values at starts and targets.
        grid = grid.at[starts].set(agent_position_values)
        grid = grid.at[targets].set(agent_target_values)
        return grid


if __name__ == '__main__':
    test = BoardDatasetGeneratorJAX(10, 5)



    print()
