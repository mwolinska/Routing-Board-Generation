import chex
import jax
from jax import numpy as jnp
from jumanji.environments.routing.connector import State
from jumanji.environments.routing.connector.types import Agent
from jumanji.environments.routing.connector.utils import get_position, get_target

from routing_board_generation.board_generation_methods.jax_implementation.board_generation.parallel_random_walk import \
    ParallelRandomWalkBoard
from routing_board_generation.board_generation_methods.jax_implementation.board_generation.seed_extension import \
    SeedExtensionBoard
from routing_board_generation.rl_training.online_generators.uniform_generator import \
    Generator, UniformRandomGenerator


class BoardDatasetGeneratorJAX(Generator):
    def __init__(self, grid_size: int, num_agents: int,
                 randomness: float = 1,
                 two_sided: bool = False,
                 extension_iterations: int = 1,
                 extension_steps: int = 1e23,
                 board_name: str = "offline_seed_extension",
                 number_of_boards: int = 10000,
                 generate_solved_boards: bool = False,
                 ) -> None:
        super().__init__(grid_size, num_agents)
        self.board_name = board_name
        if board_name == "offline_seed_extension":
            self.board_generator = SeedExtensionBoard(grid_size, grid_size, num_agents)
            self.board_generator_call = jax.jit(self.board_generator.generate_starts_ends)
            self.randomness = randomness
            self.two_sided = two_sided
            self.extension_iterations = extension_iterations
            self.extension_steps = extension_steps
        elif board_name == "offline_uniform":
            self.board_generator = UniformRandomGenerator(grid_size, num_agents)
            self.board_generator_call = jax.jit(self.board_generator)
            self.randomness = randomness
            self.two_sided = two_sided
            self.extension_iterations = extension_iterations
            self.extension_steps = extension_steps
        else:
            self.board_generator = ParallelRandomWalkBoard(grid_size, grid_size, num_agents)
            self.board_generator_call = jax.jit(self.board_generator.generate_board)

        heads, targets, solved_boards = self.generate_n_boards(jax.random.PRNGKey(0), number_of_boards, generate_solved_boards)

        self.heads = jnp.array(heads)
        self.targets = jnp.array(targets)
        self.solved_boards = solved_boards

    def generate_n_boards(self, key: chex.PRNGKey, n_boards: int = 10, generate_solved_boards: bool = False):
        heads_list = []
        targets_list = []
        solved_boards_list = []
        if generate_solved_boards and self.board_name == "offline_seed_extension":
            solved_board_call = jax.jit(self.board_generator.return_solved_board)
        keys = jax.random.split(key, num=n_boards)
        for i in range(n_boards):
            key = keys[i]
            if self.board_name == "offline_seed_extension":
                if generate_solved_boards:
                    heads_for_board = None
                    targets_for_board = None
                    solved_board = solved_board_call(key, self.randomness, self.two_sided,
                                                  self.extension_iterations, self.extension_steps,
                                                  )
                    solved_boards_list.append(solved_board)
                else:
                    heads_for_board, targets_for_board = \
                        self.board_generator_call(key, self.randomness, self.two_sided,
                                                  self.extension_iterations, self.extension_steps,
                                                  )
            else:
                heads_for_board, targets_for_board, solved_boards = self.board_generator_call(key)
                solved_boards_list.append(solved_boards)

            heads_list.append(heads_for_board)
            targets_list.append(targets_for_board)

        return heads_list, targets_list, solved_boards_list

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
    test = BoardDatasetGeneratorJAX(10, 5, number_of_boards=10, board_name="offline_seed_extension")
