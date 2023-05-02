import time
from typing import List

from routing_board_generation.benchmarking.benchmarks.benchmark_on_random_agent import (
    BenchmarkOnRandomAgent,
)
from routing_board_generation.benchmarking.utils.benchmark_utils import (
    board_generation_params_from_grid_params,
    files_list_from_benchmark_experiment,
    directory_string_from_benchamrk_experiement,
)

from routing_board_generation.benchmarking.utils.benchmark_data_model import (
    BoardGenerationParameters,
)


def run_benchmark_with_simulation(
    benchmarks_list: List[BoardGenerationParameters],
    save_plots: bool = False,
    save_simulation_data: bool = False,
    num_epochs: int = 1000,
):
    benchmark = BenchmarkOnRandomAgent.from_simulation(
        benchmark_parameters_list=benchmarks_list,
        num_epochs=num_epochs,
        save_outputs=save_simulation_data,
    )
    benchmark.plot_all(save_outputs=save_plots)
    benchmark.master_plotting_loop_violin()


def run_benchmark_from_file(
    files_for_benchmark: List[str],
    directory_string: str,
    save_plots: bool = False,
):
    benchmark = BenchmarkOnRandomAgent.from_file(
        file_name_parameters=files_for_benchmark,
        directory_string=directory_string,
    )
    benchmark.save_plots = save_plots
    benchmark.master_plotting_loop_violin()
    benchmark.plot_all(save_outputs=save_plots)


if __name__ == "__main__":
    # set to True if you want to simulate the board, False if you want to run from file
    simulation = True
    tic = time.time()
    if simulation:
        ######### Change these parameters if required
        grid_params = [(10, 10, 5)]
        save_plots = True  # Change this to False if you want to just see the plots without saving
        save_simulation_data = True
        num_epochs = 1
        benchmarks_list = (
            []
        )  # replace this with list of BoardGenerationParameters per schema below
        # benchmarks_list = [BoardGenerationParameters(rows=6, columns=6, number_of_wires=3, generator_type=BoardName.NUMBERLINK)]

        #########
        if not benchmarks_list:
            benchmarks_list = board_generation_params_from_grid_params(grid_params)
        run_benchmark_with_simulation(
            benchmarks_list=benchmarks_list,
            save_plots=save_plots,
            save_simulation_data=save_simulation_data,
            num_epochs=num_epochs,  # number of boards for simulation
        )
    else:
        ######### Change these parameters are required
        folder_name = "20230412_benchmark_23_02"  # this must be a folder under ic/experiments/benchmarks
        save_plots = False
        # get all files from folder
        all_files = files_list_from_benchmark_experiment(folder_name)
        #########
        all_files = [file for file in all_files if file[-4:] == ".pkl"]
        directory_string = (
            str(directory_string_from_benchamrk_experiement(folder_name)) + "/"
        )
        run_benchmark_from_file(
            files_for_benchmark=all_files,
            directory_string=directory_string,
            save_plots=save_plots,
        )

    print(time.time() - tic)
