import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def plot_compile_time():
    plt.style.use('science.mplstyle')

    fig, ax = plt.subplots()
    single_board = [1.09, 9.83, 2.93, 7.43]
    batch_boards = [0.719, 16.5, 4.2, 13.2]

    n = len(single_board)
    r = np.arange(n)
    width = 0.25

    ax.bar(r, single_board, color='b',
            width=width, edgecolor='black',
            label='single board')
    # ax2 = ax.twinx()
    ax.bar(r + width, batch_boards, color='g',
            width=width, edgecolor='black',
            label='batch of 100 boards')

    ax.set_xticklabels(r, rotation=30, ha='right')
    ax.set(
        title="Board Generation Including Compilation Time",
        xlabel="Generator",
        ylabel="Compilation and generation time, s",
    )
    # ax2.set(ylabel="Compilation time for batch of 100 board, s")
    # ax.set_xticks(r)
    ax.set_xticks(r + width / 2, ['UniformRandom', 'SequentialRandomWalk', 'ParallelRandomWalk', 'SeedExtension'])

    # ax.yaxis.label.set_color('b')
    # ax.tick_params(axis='y', colors='b')
    # ax2.spines['left'].set_color('b')
    #
    # ax2.yaxis.label.set_color('g')
    # ax2.tick_params(axis='y', colors='g')
    # ax2.spines['right'].set_color('g')

    lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    #
    # ax2.legend(lines + lines2, labels + labels2, loc=2)
    # ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=2)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cache_time():
    plt.style.use('science.mplstyle')

    fig, ax = plt.subplots()
    single_board = [213, 545, 255, 998]
    batch_boards = [4.18, 515, 19.7, 427]

    n = len(single_board)
    r = np.arange(n)
    width = 0.1

    ax.bar(r, single_board, color='b',
            width=width, edgecolor='black',
            label='single board')
    ax2 = ax.twinx()
    ax2.bar(r + width, batch_boards, color='g',
            width=width, edgecolor='black',
            label='batch of 100 boards')

    ax.set_xticklabels(r, rotation=30, ha='right')
    ax.set(
        title="Board Generation from Cache Time",
        xlabel="Generator",
        ylabel="Generation time of single board, µs",
    )
    ax2.set(ylabel="Generation time for batch of 100 boards, ms")
    ax.set_xticks(r)
    ax.set_xticks(r + width / 2, ['UniformRandom', 'SequentialRandomWalk', 'ParallelRandomWalk', 'SeedExtension'])

    ax.yaxis.label.set_color('b')
    ax.tick_params(axis='y', colors='b')
    ax2.spines['left'].set_color('b')

    ax2.yaxis.label.set_color('g')
    ax2.tick_params(axis='y', colors='g')
    ax2.spines['right'].set_color('g')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax2.legend(lines + lines2, labels + labels2, loc=2)
    # ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=2)
    # plt.legend()
    plt.tight_layout()
    plt.show()




def plot_connections_trained_agents():
    # plt.style.use('science.mplstyle')

    fig, ax = plt.subplots()

    fig.set_size_inches(8, 3.5)
    bfs_base = [4.55, 2.94, 4.29, 4.38, 3.23]
    bfs_min = [4.43, 2.8, 4.25, 4.21, 2.97]
    bfs_longest = [4.72, 3.16, 4.56, 4.57, 3.68]
    random_walk_np = [4.67, 3.1, 4.53, 4.46, 3.53]
    random_seed = [3.85, 2.34, 3.49, 3.45, 2.42]
    number_link = [4.01, 3.84, 3.66, 3.68, 2.74]
    eval_trained_agent = [3.75, 4.17, 4.53, 3.8, 4.98]

    n = len(bfs_base)
    r = np.arange(n)
    width = 0.1

    ax.bar(r, bfs_base,
            width=width, edgecolor='black',
            label='BFS base')
    ax.bar(r + width, bfs_min,
            width=width, edgecolor='black',
            label='BFS min_bends')
    ax.bar(r + 2 * width, bfs_longest,
           width=width, edgecolor='black',
           label='BFS longest')
    ax.bar(r + 3 * width, random_walk_np,
           width=width, edgecolor='black',
           label='Numpy Random Walk')
    ax.bar(r + 4 * width, random_seed,
           width=width, edgecolor='black',
           label='Seed Extension')
    ax.bar(r + 5 * width, number_link,
           width=width, edgecolor='black',
           label='Number Link')
    ax.bar(r + 6 * width, eval_trained_agent,
           width=width, edgecolor='black',
           label='Evaluation during training')



    ax.set_xticklabels(r, rotation=30, ha='right')
    ax.set(
        title="Number of Wires Connected using Trained Agents",
        xlabel="Generator",
        ylabel="Number of wires connected",
    )
    # ax2.set(ylabel="Generation time for batch of 100 boards, ms")
    # ax.set_xticks(r)
    ax.set_xticks(r + 6 * width / 2, ['UniformRandom', "BFS_Short", 'NumpyRandomWalk', 'NumberLink', 'LSystems'])

    # ax.yaxis.label.set_color('b')
    # ax.tick_params(axis='y', colors='b')
    # ax2.spines['left'].set_color('b')
    #
    # ax2.yaxis.label.set_color('g')
    # ax2.tick_params(axis='y', colors='g')
    # ax2.spines['right'].set_color('g')
    #
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    #
    # ax2.legend(lines + lines2, labels + labels2, loc=2)
    # ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
    #            mode="expand", borderaxespad=0, ncol=2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    plot_connections_trained_agents()
