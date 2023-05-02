from routing_board_generation.benchmarking.utils.benchmark_data_model import \
    BoardGenerationParameters
from routing_board_generation.board_generation_methods.numpy_implementation.utils.utils import \
    get_heads_and_targets
from routing_board_generation.interface.board_generator_interface import \
    BoardGenerator


def generate_n_boards(
    board_parameters: BoardGenerationParameters,
    number_of_boards: int,
):
    board_list = []
    heads_list = []
    targets_list = []
    board_class = BoardGenerator.get_board_generator(board_parameters.generator_type)
    for _ in range(number_of_boards):
        board = None
        none_counter = 0
        while board is None:
            board_generator = board_class(
                rows=board_parameters.rows, cols=board_parameters.columns,
                num_agents=board_parameters.number_of_wires,
            )
            board = board_generator.return_training_board()
            heads, targets = get_heads_and_targets(board)
            if len(heads) != board_parameters.number_of_wires:
                board = None
            none_counter += 1
            if none_counter == 100:
                raise ValueError("Failed to generate board 100 times")
        board_list.append(board)
        heads_list.append(heads)
        targets_list.append(targets)
    return board_list, heads_list, targets_list
