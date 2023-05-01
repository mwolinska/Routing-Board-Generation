from routing_board_generation.benchmarking.benchmarks.empty_board_evaluation import \
    evaluate_generator_outputs_averaged_on_n_boards
from routing_board_generation.benchmarking.utils.benchmark_utils import \
    board_generation_params_from_grid_params

if __name__ == '__main__':
    # Option 1: run for all generators
    grid_params = [(10, 10, 5)]
    board_list = board_generation_params_from_grid_params(grid_params)

    # Option 2:specify board generation parameters
    # board_list = [
    #     BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #                               generator_type=BoardName.LSYSTEMS),
    #     # BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #     #                           generator_type=BoardName.BFS_SHORTEST),
    #     # BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #     #                           generator_type=BoardName.JAX_SEED_EXTENSION),
    #     # BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #     #                           generator_type=BoardName.LSYSTEMS),
    #     # BoardGenerationParameters(rows=10, columns=10, number_of_wires=5,
    #     #                           generator_type=BoardName.WFC),
    # ]
    evaluate_generator_outputs_averaged_on_n_boards(board_list, number_of_boards=3, plot_individually=False)
