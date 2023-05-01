import collections
import pickle
from typing import List, Dict

import numpy as np

from ic_routing_board_generation.benchmarking.benchmark_data_model import \
    BoardGenerationParameters, BarChartData
from ic_routing_board_generation.benchmarking.benchmark_utils import \
    generate_n_boards, load_pickle
from ic_routing_board_generation.benchmarking.plotting_utils import \
    plot_heatmap, plot_comparison_heatmap
from ic_routing_board_generation.board_generator.numpy_board_generation.bfs_board_variations import \
    BFSBoardShortest
from ic_routing_board_generation.board_generator.numpy_board_generation.board_generator_random_walk_rb import \
    RandomWalkBoard
from ic_routing_board_generation.board_generator.numpy_utils.board_processor_2 import \
    BoardProcessor
from ic_routing_board_generation.ic_rl_training.offline_generation.dataset_generator_jax import \
    BoardDatasetGeneratorJAX
from ic_routing_board_generation.interface.board_generator_interface_numpy import BoardName


class EvaluateEmptyBoard:
    def __init__(self, filled_training_board: np.ndarray):
        self.board_processor = BoardProcessor(filled_training_board)
        self.filled_board = self.board_processor.get_board_layout()
        self.empty_slot_score = -2
        self.end_score = 3
        self.wire_score = 2
        self.scored_board = self.score_from_neighbours()
        self.board_statistics = self._get_board_statistics()


    def assess_board(self):
        is_zero_mask = self.filled_board == 0
        divisible_by_three_mask = self.filled_board % 3 == 0

        empty_slots_mask = np.array(is_zero_mask, dtype=int) * self.empty_slot_score
        heads = np.array(self.filled_board % 3 == 2, dtype=int) * self.end_score
        targets = np.array(divisible_by_three_mask * ~is_zero_mask, dtype=int) * self.end_score
        routes = np.array(self.filled_board % 3 == 1, dtype=int) * self.wire_score

        return np.sum([empty_slots_mask, heads, targets, routes], axis=0)

    def score_from_neighbours(self):
        individual_score = self.assess_board()
        training_board_filtered = self._change_heads_to_wire_ids()
        # total_num_wires = len(np.unique(training_board_filtered)[np.unique(training_board_filtered)!= 0])
        padded_array = np.pad(individual_score, 1, mode="constant")

        filter = np.array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ]
        )

        # filter = np.array(
        #     [
        #         [0, 1, 0],
        #         [1, 2, 1],
        #         [0, 1, 0],
        #     ]
        # )

        scores = []
        for row in range(1, len(padded_array) - 1):
            for column in range(1, len(padded_array[0]) - 1):
                wires_in_window = np.unique(training_board_filtered[row - 1: row + 2, column - 1: column + 2])
                wires_in_window = len(wires_in_window[wires_in_window != 0])
                diversity = wires_in_window
                window = padded_array[row - 1: row + 2, column - 1: column + 2]
                score = np.sum(window * filter * diversity)
                scores.append(score)

        scored_array = np.array(scores).reshape((len(self.filled_board), len(self.filled_board[0])))

        return scored_array

    def _change_heads_to_wire_ids(self) -> np.ndarray:
        wires_only_board = np.pad(self.filled_board, 1, mode="constant")
        unique_wire_ids = np.unique(self.filled_board[self.filled_board % 3 == 2])

        for wire_id in unique_wire_ids:
            wires_only_board[wires_only_board == (wire_id + 1)] = wire_id
            wires_only_board[wires_only_board == (wire_id + 2)] = wire_id

        # wires_only_board[self.filled_board % 3 == 2] = self.filled_board
        return wires_only_board

    def _local_diversity_score(self, window: np.ndarray):
        pass

    def count_detours(self, count_current_wire: bool = False) -> int:
        """Return the number of wires that have to detour around a head or target cell.

            Args:
                count_current_wire (bool): Should we count wires that wrap around their own heads/targets? (default = False)

            Returns:
                (int) : The number of wires that have to detour around a head or target cell.
        """
        num_detours = 0
        for x in range(len(self.filled_board)):
            for y in range(len(self.filled_board[0])):
                cell_label = self.filled_board[x, y]
                if (cell_label < 2) or ((cell_label % 3) == 2):
                    continue
                current_wire = self.get_wire_num(cell_label)
                above = self.filled_board[:x, y]
                above = [self.get_wire_num(cell) for cell in above if cell != 0]
                if not count_current_wire:
                    above = [wire_num for wire_num in above if wire_num != current_wire]
                below = self.filled_board[x + 1:, y]
                below = [self.get_wire_num(cell) for cell in below if cell != 0]
                if not count_current_wire:
                    below = [wire_num for wire_num in below if wire_num != current_wire]
                common = (set(above) & set(below))
                num_detours += len(common)
                left = self.filled_board[x, :y].tolist()
                left = [self.get_wire_num(cell) for cell in left if cell != 0]
                if not count_current_wire:
                    left = [wire_num for wire_num in left if wire_num != current_wire]

                right = self.filled_board[x, y + 1:].tolist()
                right = [self.get_wire_num(cell) for cell in right if cell != 0]
                if not count_current_wire:
                    right = [wire_num for wire_num in right if wire_num != current_wire]
                common = (set(right) & set(left))
                num_detours += len(common)
        return num_detours

    def get_wire_num(self, cell_label: int) -> (int):
        """ Returns the wire number of the given cell value

            Args:
                cell_label (int) : the value of the cell in self.layout

            Returns:
                (int) : The wire number that the cell belongs to. Returns -1 if not part of a wire.
        """
        if cell_label < 2:
            return -1
        else:
            return ((cell_label-2) // 3)

    def _get_board_statistics(self):
        board_stats = self.board_processor.get_board_statistics()
        board_stats["count_detours"] = self.count_detours()
        board_stats["heatmap_score_diversity"] = len(np.unique(self.scored_board))
        board_stats.pop("wire_lengths")
        board_stats.pop("wire_bends")
        return board_stats




def evaluate_generator_outputs_averaged_on_n_boards(
    board_parameters_list: List[BoardGenerationParameters],
    number_of_boards: int,
    plot_individually: bool = False,
):
    # TODO (Marta): add exception of all board_gen_parameters are not the same (with the exception of board_type)
    # TODO (Marta): write results to file
    scores_list = []
    board_names = []
    all_board_statistics = []
    for board_parameters in board_parameters_list:
        print(board_parameters)
        if board_parameters.generator_type == BoardName.JAX_PARALLEL_RW: # or board_parameters.generator_type == BoardName.JAX_SEED_EXTENSION:
            board_list = BoardDatasetGeneratorJAX(
                board_name=board_parameters.generator_type.value,
                grid_size=board_parameters.rows,
                num_agents=board_parameters.number_of_wires,
                number_of_boards=number_of_boards,
                generate_solved_boards=True,
            ).solved_boards
        else:
            board_list = generate_n_boards(board_parameters, number_of_boards)
        board_statistics = {}
        sum_all_boards = np.zeros([board_parameters.rows, board_parameters.columns])
        for board in board_list:
            board = np.array(board)
            board_evaluator = EvaluateEmptyBoard(board)
            scored_board = board_evaluator.scored_board
            sum_all_boards += scored_board
            new_board_statistics = board_evaluator.board_statistics
            board_statistics = _update_dictionary(board_statistics, new_board_statistics)

        scores_list.append(sum_all_boards / number_of_boards)
        board_names.append(str(board_parameters.generator_type.value))
        board_statistics = {k: (np.mean(np.array(v)), np.std(np.array(v))) for k, v in dict(board_statistics).items()}
        board_statistics["generator_type"] = str(board_parameters.generator_type.value)
        all_board_statistics.append(board_statistics)
    # with open("all_board_stats.pkl", "wb") as file:
    #     pickle.dump(all_board_statistics, file)
    # with open("heatmap_stats.pkl", "wb") as file:
    #     pickle.dump([scores_list, board_names, board_parameters_list[0].number_of_wires], file)

    if plot_individually or len(board_parameters_list) == 1:
        for score in scores_list:
            plot_heatmap(scores=score)
    else:
        plot_comparison_heatmap(
            scores_list, board_names, board_parameters_list[0].number_of_wires,
            number_of_boards_averaged=number_of_boards,
        )
    print(all_board_statistics)
    with open("all_board_stats.pkl", "wb") as file:
        pickle.dump(all_board_statistics, file)
    with open("heatmap_stats.pkl", "wb") as file:
        pickle.dump([scores_list, board_names, board_parameters_list[0].number_of_wires], file)
    convert_dict_lit_to_plotting_format(all_board_statistics)

def convert_dict_lit_to_plotting_format(list_of_dict: List[Dict[str, float]]):
    data_per_key = []
    list_of_keys = [x for x in list_of_dict[0].keys()]
    list_of_board_names = [x["generator_type"] for x in list_of_dict]

    titles = {'num_wires': " Number of Wires on Board",
 'avg_wire_length': "Average Wire Length",
 'avg_wire_bends': "Average Number of Bends per Wire" ,
 'avg_head_target_distance': "Average Manahttan Distance from Head to Target",
 'percent_filled': "Percent of Board Filled with Wires",
 'count_detours': "Average Number of Detours",
 'heatmap_score_diversity': "Average Heatmap Diversity Score",
              }
    y_labels = {'num_wires': "Number of Wires",
 'avg_wire_length': "Wire Length",
 'avg_wire_bends': "Number of Bends per Wire" ,
 'avg_head_target_distance': "Manhattan Distance",
 'percent_filled': "Percent of Board Filled with Wires",
 'count_detours': " Number of Detours",
 'heatmap_score_diversity': "Heatmap Diversity Score",}
    for key in list_of_keys: #and key != "generator_type"
        if key == "generator_type":
            continue
        key_data = []
        key_stds = []
        for board_data in list_of_dict:
            key_data.append(board_data[key][0])
            key_stds.append(board_data[key][1])

        bar_chart = BarChartData(
            x_axis_label="Generator Type",
            y_axis_label=y_labels[key],
            x_labels=list_of_board_names,
            y_data=key_data,
            title=titles[key],
            output_filename=None,
            stds=key_stds,
        )
        bar_chart.plot()
        data_per_key.append(key_data)
    return list_of_keys, data_per_key


def _update_dictionary(dictionary_to_update, new_dictionary):
    temp_dict = collections.defaultdict(list)
    if not dictionary_to_update:
        # temp_dict = new_dictionary
        for key, value in new_dictionary.items():
            temp_dict[key] = [value]
    else:
        temp_dict = collections.defaultdict(list)
        for key, value in dictionary_to_update.items():
            temp_dict[key] = value + [new_dictionary[key]]

    return temp_dict


if __name__ == '__main__':
    benchmark = load_pickle("all_board_stats.pkl")
    print()
    # convert_dict_lit_to_plotting_format(benchmark)
    # board = np.array(
    #    [
    #        [2, 5, 8, 11],
    #        [1, 4, 7, 10],
    #        [1, 4, 7, 10],
    #        [3, 6, 9, 12]
    #    ]
    # )

    # board = np.array(
    #    [
    #        [2, 11, 10, 10],
    #        [1, 7, 8, 10],
    #        [1, 9, 12, 10],
    #        [3, 6, 4, 5]
    #    ]
    # )
    board_ = BFSBoardShortest(8, 8, 5)
    board = board_.return_solved_board()
    # print(board_.return_solved_board())
    # boardprocessor_ = BoardProcessor(board_)
    # print(boardprocessor_.get_board_layout())

    evaluator = EvaluateEmptyBoard(board)
    scores = evaluator.score_from_neighbours()
    #
    # print(scores)
    # print(np.unique(scores, return_counts=True))
    plot_heatmap(scores)
    print()
