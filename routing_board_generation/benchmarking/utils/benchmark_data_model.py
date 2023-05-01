import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Iterable

import numpy as np
from chex import Array
import jax.numpy as jnp
from matplotlib import pyplot as plt

from routing_board_generation.interface.board_generator_interface import BoardName


@dataclass
class BoardGenerationParameters:
    rows: int
    columns: int
    number_of_wires: int
    generator_type: BoardName


@dataclass
class BenchmarkData:
    episode_length: List[float]
    episode_return: List[Array]
    num_connections: List[float]
    ratio_connections: List[float]
    time: List[float]
    total_path_length: List[int]
    generator_type: Optional[BoardGenerationParameters] = None

    def return_plotting_dict(self):
        plotting_dict = {
            "total_reward": {
                "x_label": "",
                "y_label": "",
                "bar_chart_title": "",
                "violin_plot_title": "",
                "average_value": "",
                "std": "",
                "file_name": "",

            }

        }
        return plotting_dict
    def average_reward_per_wire(self):
        return float(jnp.mean(jnp.array(self.episode_return), axis=0))

    def std_reward_per_wire(self):
        return float(jnp.std(jnp.array(self.episode_return), axis=(0)))

    def average_total_wire_length(self):
        return float(jnp.mean(jnp.array(self.total_path_length), axis=0))

    def std_total_wire_length(self):
        return float(jnp.std(jnp.array(self.total_path_length), axis=0))

    def average_proportion_of_wires_connected(self):
        return float(jnp.mean(jnp.array(self.ratio_connections), axis=0))

    def std_proportion_of_wires_connected(self):
        return float(jnp.std(jnp.array(self.ratio_connections), axis=0))

    def average_steps_till_board_terminates(self):
        return float(jnp.mean(jnp.array(self.episode_length), axis=0))

    def std_steps_till_board_terminates(self):
        return float(jnp.std(jnp.array(self.episode_length), axis=0))

@dataclass
class BarChartData:
    x_axis_label: str
    y_axis_label: str
    y_data: Iterable
    x_labels: str
    title: str
    output_filename: str
    stds: Optional[List[float]]

    def plot(self):
        # plt.style.use('science.mplstyle')
        plt.style.use('tableau-colorblind10')
        plt.rcParams["font.family"] = "Times"
        plt.rcParams["font.size"] = 14
        plt.rcParams['figure.dpi'] = 900
        plt.rcParams["figure.figsize"] = (7, 4.5)

        fig, ax = plt.subplots()
        ax.bar(self.x_labels, self.y_data)
        ax.set(
            title=self.title,
            xlabel=self.x_axis_label,
            ylabel=self.y_axis_label,
        )
        data = np.array(self.y_data)

        ax.set_xticks(self.x_labels)
        ax.set_xticklabels(self.x_labels, rotation=30, ha='right')
        inds = np.arange(0, len(data))
        ax.scatter(inds, data, marker='o', color='k', s=30, zorder=3)
        if self.stds is not None:
            stds = np.array(self.stds)
            ax.vlines(inds, data - (stds / 2), data + (stds / 2), color='blue',
                      linestyle='-',
            lw = 3)
        plt.tight_layout()
        if self.output_filename is not None:
            fig.savefig(str(self.directory_string) + str(self.output_filename))
        else:
            time = datetime.now().strftime("%H_%M")
            plt.savefig(f"figs/{self.title}_{time}.pdf")
            plt.savefig(f"figs/{self.title}_{time}.png")
            plt.show()

if __name__ == '__main__':
    test = BenchmarkData
    # print(test.plotting_dict)
    for field in dataclasses.fields(BenchmarkData):
        print(field.name)
