import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from hydra import compose, initialize
import cv2


from jumanji.training.setup_train import setup_agent, setup_env
from jumanji.training.utils import first_from_device
from jumanji.env import Environment
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.connector.types import Agent, Observation, State
from jumanji.types import restart


from jumanji.environments.routing.connector.viewer import ConnectorViewer
from jumanji.environments.routing.connector.utils import get_position, get_target

from jumanji.environments.routing.connector.constants import POSITION, TARGET


from routing_board_generation.interface.board_generator_interface import (
    BoardGenerator,
    BoardName,
)


def state_from_board(board: jnp.ndarray) -> State:
    """Converts a board to a jumanji state"""
    # get start and target positions
    def find_positions(wire_id):
        wire_positions = board == 3 * wire_id + POSITION
        wire_targets = board == 3 * wire_id + TARGET

        # Compute indices where wire_positions and wire_targets are True
        start_indices = jnp.argwhere(wire_positions, size=2)
        end_indices = jnp.argwhere(wire_targets, size=2)

        # Take the first valid index (row)
        start = start_indices[0]
        end = end_indices[0]
        return start, end

    wire_ids = jnp.arange(5)
    starts_ends = jax.vmap(find_positions)(wire_ids)
    # jax.debug.print("{x}", x=starts_ends)
    # jax.debug.print("{x}", x=type(starts_ends))
    starts, ends = starts_ends[0], starts_ends[1]

    # Want starts to be a tuple of arrays, first array is x coords, second is y coords
    starts = (starts[:, 0], starts[:, 1])
    targets = (ends[:, 0], ends[:, 1])

    # Get the agent values for starts and positions.
    agent_position_values = jax.vmap(get_position)(jnp.arange(5))
    agent_target_values = jax.vmap(get_target)(jnp.arange(5))

    # Create empty grid.
    grid = jnp.zeros((10, 10), dtype=jnp.int32)

    # Place the agent values at starts and targets.
    grid = grid.at[starts].set(agent_position_values)
    grid = grid.at[targets].set(agent_target_values)

    # Create the agent pytree that corresponds to the grid.
    agents = jax.vmap(Agent)(
        id=jnp.arange(5),
        start=jnp.stack(starts, axis=1),
        target=jnp.stack(targets, axis=1),
        position=jnp.stack(starts, axis=1),
    )

    step_count = jnp.array(0, jnp.int32)

    return State(key=key, grid=grid, step_count=step_count, agents=agents)


def board_to_env(board: jnp.ndarray) -> Environment:
    """Converts a board to a jumanji environment"""
    # get proper state
    board_env = Connector()
    state = state_from_board(board)
    action_mask = jax.vmap(board_env._get_action_mask, (0, None))(
        state.agents, state.grid
    )
    observation = Observation(
        grid=board_env._obs_from_grid(state.grid),
        action_mask=action_mask,
        step_count=state.step_count,
    )
    extras = board_env._get_extras(state)
    timestep = restart(
        observation=observation, extras=extras, shape=(board_env.num_agents,)
    )
    return state, timestep


def make_target_board(viewer, args, key):
    board_generator = BoardGenerator.get_board_generator(BoardName(args.board_type))
    initial_board = board_generator(args.board_size, args.board_size, args.num_agents)
    # Generate depending on board type
    if args.board_type == "random_walk":
        training_board = initial_board.return_training_board()
    elif args.board_type == "offline_parallel_rw":
        training_board = initial_board.generate_board(key)
    elif args.board_type == "bfs_base":
        training_board = initial_board.generate_boards(1)
    elif (
        args.board_type == "bfs_min_bend"
        or args.board_type == "bfs_fifo"
        or args.board_type == "bfs_short"
        or args.board_type == "bfs_long"
    ):
        training_board = initial_board.return_training_board()
    elif args.board_type == "lsystems_standard":
        training_board = initial_board.return_training_board()
    elif args.board_type == "wfc":
        training_board = initial_board.return_training_board()
    elif args.board_type == "numberlink":
        training_board = initial_board.return_training_board()

    def close_figure_on_key(event):
        if event.key == " ":
            plt.close()

    # Make sure the board is a jnp array
    training_board = jax.numpy.array(training_board)
    viewer.render(training_board)

    # Connect the event handler to the current figure
    plt.gcf().canvas.mpl_connect("key_press_event", close_figure_on_key)
    plt.show()
    return training_board


def step_agents(args, training_board, key):
    # TODO: check we have generated board of correct size / num_agents
    if args.num_agents != 5:
        raise NotImplementedError("Only have an agent trained for 5 wires.")
    if args.board_size != 10:
        raise NotImplementedError("Only have an agent trained for 10x10 boards.")
    file = (
        "examples/trained_agent_10x10_5_uniform/19-27-36/training_state_10x10_5_uniform"
    )
    with open(file, "rb") as f:
        training_state = pickle.load(f)

    with initialize(version_base=None, config_path="./configs"):
        cfg = compose(
            config_name="config.yaml", overrides=["env=connector", "agent=a2c"]
        )

    params = first_from_device(training_state.params_state.params)
    env = setup_env(cfg).unwrapped
    agent = setup_agent(cfg, env)
    policy = agent.make_policy(params.actor, stochastic=False)

    states = []
    key = jax.random.PRNGKey(cfg.seed)

    key, _ = jax.random.split(key)
    state, timestep = board_to_env(training_board)
    states.append(state.grid)

    while not timestep.last():
        key, action_key = jax.random.split(key)
        observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)
        action, _ = policy(observation, action_key)
        state, timestep = env.step(state, action.squeeze(axis=0))
        states.append(state.grid)

    # Render the animation
    animation = viewer.animate(states)
    # Animation is a matplotlib.animation.FuncAnimation object
    # save the animation
    animation.save("animation.mp4")
    from matplotlib.animation import writers

    if not writers.is_available("ffmpeg"):
        print(
            "Error: FFmpeg is not available. Please install it to save the animation."
        )
    else:
        # Save the animation
        animation.save("animation.mp4", writer="ffmpeg")

        def play_video(filename):
            cap = cv2.VideoCapture(filename)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow("Video", frame)

                # Wait for a key press
                key = cv2.waitKey(0) & 0xFF

                # Press 'q' to exit the video window
                if key == ord("q"):
                    break
                # Press spacebar to advance to the next frame
                elif key == ord(" "):
                    continue

            cap.release()
            cv2.destroyAllWindows()

        video_filename = "animation.mp4"
        play_video(video_filename)

    return


parser = argparse.ArgumentParser()

board_choices = [board.value for board in BoardName]

parser.add_argument(
    "--board_type",
    default="random_walk",
    type=str,
    choices=BoardGenerator.board_generator_dict.keys(),
)
parser.add_argument(
    "--show", default="stepping", type=str, choices=["stepping", "board"]
)
parser.add_argument("--board_size", default=10, type=int)
parser.add_argument("--num_agents", default=5, type=int)
parser.add_argument("--seed", default=0, type=int)


if __name__ == "__main__":
    # Render the board using jumanji's method
    args = parser.parse_args()
    key = jax.random.PRNGKey(args.seed)
    viewer = ConnectorViewer("Training Board", args.num_agents)
    training_board = make_target_board(viewer, args, key)

    # Show a trained agent trying to solve the board
    if args.show == "stepping":
        step_agents(args, training_board, key)
