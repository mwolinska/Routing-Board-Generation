import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import datetime
from hydra import compose, initialize
import cv2


from jumanji.training.setup_train import setup_agent, setup_env
from jumanji.training.utils import first_from_device
from jumanji.env import Environment
from jumanji.environments.routing.connector.env import Connector
from jumanji.environments.routing.connector.types import Agent, Observation, State
from jumanji.types import TimeStep, restart, termination, transition


from jumanji.environments.routing.connector.viewer import ConnectorViewer
from jumanji.environments.routing.connector.utils import get_position, get_target

from jumanji.environments.routing.connector.constants import POSITION, TARGET, EMPTY


from routing_board_generation.interface.board_generator_interface import BoardGenerator, BoardName



def state_from_board(board: jnp.ndarray) -> State:
    """Converts a board to a jumanji state"""
    # get start and target positions
    def find_positions(wire_id):
        wire_positions = board == 3 * wire_id + POSITION
        wire_targets = board   == 3 * wire_id + TARGET

        # Compute indices where wire_positions and wire_targets are True
        start_indices = jnp.argwhere(wire_positions, size=2)
        end_indices   = jnp.argwhere(wire_targets,   size=2)

        # Take the first valid index (row)
        start = start_indices[0]
        end = end_indices[0]
        return start, end

    wire_ids = jnp.arange(5)
    starts_ends = jax.vmap(find_positions)(wire_ids)
    #jax.debug.print("{x}", x=starts_ends)
    #jax.debug.print("{x}", x=type(starts_ends))
    starts, ends = starts_ends[0], starts_ends[1]

    # Want starts to be a tuple of arrays, first array is x coords, second is y coords
    starts = (starts[:, 0], starts[:, 1])
    targets = (ends[:, 0], ends[:, 1])

    # Create 2D points from the flat arrays.
    # starts = jnp.divmod(starts_flat, self.grid_size)
    # targets = jnp.divmod(targets_flat, self.grid_size)

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
    



parser = argparse.ArgumentParser()

board_choices = [board.value for board in BoardName]

parser.add_argument(
    "--board_type",
    default="offline_parallel_rw",
    type=str,
    choices=BoardGenerator.board_generator_dict.keys()
)
parser.add_argument(
    "--show",
    default="training",
    type=str,
    choices=["training", "start", "target", "both"]
)
parser.add_argument(
    "--board_size",
    default=10,
    type=int
)
parser.add_argument(
    "--num_agents",
    default=5,
    type=int
)
parser.add_argument(
    "--seed",
    default=0,
    type=int
)
 


if __name__ == "__main__":
    args = parser.parse_args()
    key = jax.random.PRNGKey(args.seed)
    if args.show != "target":
        board_generator = BoardGenerator.get_board_generator(BoardName(args.board_type))
        initial_board = board_generator(args.board_size, args.board_size, args.num_agents)
        # Generate depending on board type
        if args.board_type == "RandomWalk":
            training_board= initial_board.return_training_board()
        elif args.board_type == "ParallelRandomWalk":
            training_board= initial_board.generate_board(key)
        elif args.board_type == "BFSBase":
            training_board = initial_board.generate_boards(1)
        elif args.board_type == "BFSMin_Bends" or args.board_type == "BFSFIFO" or args.board_type == "BFSSHORT" or args.board_type == "BFSLong":
            training_board = initial_board.return_training_board()
        elif args.board_type == "L-Systems":
            training_board = initial_board.return_training_board()
        elif args.board_type == "WFC":
            training_board = initial_board.return_training_board()
        elif args.board_type == "NumberLink":
            training_board = initial_board.return_training_board()

    def close_figure_on_key(event):
        if event.key == ' ':
            plt.close()

    # Render the board using jumanji's method
    viewer = ConnectorViewer("Ben", args.num_agents)
    # Make sure the board is a jnp array
    training_board = jax.numpy.array(training_board)
    viewer.render(training_board)

    # Connect the event handler to the current figure
    plt.gcf().canvas.mpl_connect('key_press_event', close_figure_on_key)
    plt.show()

    # TODO: add argument for showing a trained agent on the board
    
    # Show a trained agent trying to solve the board
    # TODO: change file path once rebased
    if args.show == "training":
        file = "examples/trained_agent_10x10_5_uniform/19-27-36/training_state_10x10_5_uniform"
        with open(file,"rb") as f:
            training_state = pickle.load(f)
        
        with initialize(version_base=None, config_path="./configs"):
            cfg = compose(config_name="config.yaml", overrides=["env=connector", "agent=a2c"])

        params = first_from_device(training_state.params_state.params)
        #print(params)
        env = setup_env(cfg).unwrapped
        #print(env)
        agent = setup_agent(cfg, env)
        #print(agent)
        policy = agent.make_policy(params.actor, stochastic = False)
        #print(params.num_agents)


        step_fn = env.step  # Speed up env.step
        GRID_SIZE = 10

        states = []
        key = jax.random.PRNGKey(cfg.seed)

        connections = []
        key, reset_key = jax.random.split(key)
        state, timestep = board_to_env(training_board)
        states.append(state.grid)


        while not timestep.last():
            key, action_key = jax.random.split(key)
            observation = jax.tree_util.tree_map(lambda x: x[None], timestep.observation)
            # Two implementations for calling the policy, about equivalent speed
            action, _ = policy(observation, action_key)
            #action, _ = jax.jit(policy)(observation, action_key)
            # Three implementations for updating the state/timestep.  The third is much faster.
            #state, timestep = jax.jit(env.step)(state, action.squeeze(axis=0)) # original jit = 0.32, 52sec/10
            state, timestep = env.step(state, action.squeeze(axis=0)) # no jit = 0.13, 26sec/10
            #state, timestep = step_fn(state, action.squeeze(axis=0)) # jit function = 0.003 5 sec/10, 49sec/100d
            states.append(state.grid)
        
        # Render the animation
        animation = viewer.animate(states)
        # Animation is a matplotlib.animation.FuncAnimation object
        # save the animation
        animation.save('animation.mp4')
        from matplotlib.animation import writers

        if not writers.is_available('ffmpeg'):
            print("Error: FFmpeg is not available. Please install it to save the animation.")
        else:
            # Save the animation
            animation.save('animation.mp4', writer='ffmpeg')

            def play_video(filename):
                cap = cv2.VideoCapture(filename)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    cv2.imshow('Video', frame)

                    # Wait for a key press
                    key = cv2.waitKey(0) & 0xFF

                    # Press 'q' to exit the video window
                    if key == ord('q'):
                        break
                    # Press spacebar to advance to the next frame
                    elif key == ord(' '):
                        continue

                cap.release()
                cv2.destroyAllWindows()

            video_filename = 'animation.mp4'
            play_video(video_filename)
