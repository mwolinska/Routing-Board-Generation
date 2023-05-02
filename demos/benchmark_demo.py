import argparse
import os
import matplotlib
import matplotlib.pyplot as plt

from routing_board_generation.benchmarking.benchmarks.empty_board_evaluation import \
    evaluate_generator_outputs_averaged_on_n_boards
from routing_board_generation.benchmarking.utils.benchmark_utils import \
    board_generation_params_from_grid_params


parser = argparse.ArgumentParser()

parser.add_argument(
    "--number_of_boards",
    default=20,
    type=int
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
    "--remove_figs",
    default=True,
    type=bool
)


if __name__ == '__main__':
    args = parser.parse_args()
    # Used to save figures without displaying them
    matplotlib.use('Agg')
    # Delete all figures in the figs folder if remove_figs is True
    if args.remove_figs:
        for file in os.listdir('figs'):
            os.remove(os.path.join('figs', file))
    # create a figs file if it doesn't exist
    if not os.path.exists('figs'):
        os.makedirs('figs')

    grid_params = [(args.board_size, args.board_size, args.num_agents)]
    board_list = board_generation_params_from_grid_params(grid_params)
    evaluate_generator_outputs_averaged_on_n_boards(
        board_list,
        number_of_boards=args.number_of_boards, 
        plot_individually=False
    )