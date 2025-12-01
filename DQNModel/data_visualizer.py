#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


def read_file(file_name: str) -> tuple[list[int], list[int], list[list[int]]]:
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.rstrip("\n") for line in lines[1:]]
        scores, max_tiles, board_states = [], [], []
        for line in lines:
            x, y, z = line.split(",")
            scores.append(int(x))
            max_tiles.append(int(y))
            board_states.append(reconstruct_board_state(z))

        return (np.array(scores), np.array(max_tiles), np.array(board_states))


def generate_graphs(data_set_name, scores, max_tiles, board_states):
    # Score histogram
    plt.figure()
    plt.hist(scores, bins=30)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.xlim(0, 60000)
    plt.xticks(np.arange(0, 60001, 10000))
    plt.title(f"{data_set_name} Score Distribution")
    plt.savefig(
        f"graphs/{data_set_name.replace(' ', '_').lower()}_score.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    tile_ticks = range(4, 13)

    # Max tile histogram
    plt.figure()
    plt.hist(np.log2(max_tiles), bins=30)
    plt.xlabel("Max Tile Value (log2)")
    plt.ylabel("Count")
    plt.xticks(tile_ticks)
    plt.title(f"{data_set_name} Max Tile Distribution")
    plt.savefig(
        f"graphs/{data_set_name.replace(' ','_').lower()}_max_tiles.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    tiles = np.array([128, 256, 512, 1024, 2048])
    log_tiles = np.log2(tiles)
    probs = [(np.log2(max_tiles) >= t).mean() for t in log_tiles]
    plt.figure()
    plt.bar(log_tiles, probs)
    plt.xlabel("Tile Value (log2)")
    plt.ylabel("Probability")
    plt.xticks(log_tiles, tiles)
    plt.title(f"{data_set_name} Tile-Reach Probability")
    plt.ylim(0, 1)
    plt.savefig(
        f"graphs/{data_set_name.replace(' ','_').lower()}_tile_reach_prob.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def reconstruct_board_state(board_str: str) -> list[int]:
    board_list = board_str.lstrip("[").rstrip("]").split()
    return np.array([int(cell) for cell in board_list])


def main():
    parser = ArgumentParser()
    parser.add_argument("--file-name", required=True, type=str)
    args = parser.parse_args()
    step_counts = [0, 10, 20, 30, 40, 50, 60, 72]
    average_scores = []
    median_scores = []
    high_scores = []
    median_max_tiles = []
    highest_tiles = []
    for step_count in step_counts:
        file_base = args.file_name
        file_name = f"{step_count}_{file_base}"
        scores, max_tiles, board_states = read_file(file_name)
        generate_graphs(
            f"{step_count} Million Step Trained", scores, max_tiles, board_states
        )

        average_scores.append(np.average(scores))
        median_scores.append(np.median(scores))
        high_scores.append(np.max(scores))
        median_max_tiles.append(np.median(max_tiles))
        highest_tiles.append(np.max(max_tiles))

    plt.figure()
    plt.bar(range(len(step_counts)), average_scores)
    plt.xticks(range(len(step_counts)), step_counts)
    plt.xlabel("Training Steps (million steps)")
    plt.ylabel("Average Score")
    plt.title("Average Score vs. Training Steps")
    plt.tight_layout()
    plt.savefig("graphs/avg_score_vs_steps_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.bar(range(len(step_counts)), median_scores)
    plt.xticks(range(len(step_counts)), step_counts)
    plt.xlabel("Training Steps (million steps)")
    plt.ylabel("Median Score")
    plt.title("Median Score vs. Training Steps")
    plt.tight_layout()
    plt.savefig("graphs/med_score_vs_steps_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.bar(range(len(step_counts)), high_scores)
    plt.xticks(range(len(step_counts)), step_counts)
    plt.xlabel("Training Steps (million steps)")
    plt.ylabel("Highest Score")
    plt.title("Highest Score vs. Training Steps")
    plt.tight_layout()
    plt.savefig("graphs/high_score_vs_steps_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    tile_ticks = np.array([128, 256, 512, 1024, 2048, 4096])
    plt.figure()
    plt.bar(range(len(step_counts)), median_max_tiles)
    plt.yscale("log", base=2)
    plt.xticks(range(len(step_counts)), step_counts)
    plt.yticks(tile_ticks, tile_ticks)
    plt.xlabel("Training Steps (million steps)")
    plt.ylabel("Median Max Tiles")
    plt.title("Median Max Tile vs. Training Steps")
    plt.tight_layout()
    plt.savefig("graphs/med_tile_vs_steps_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    # plt.figure()
    # plt.bar(range(len(step_counts)), highest_tiles)
    # plt.xticks(range(len(step_counts)), step_counts)
    # plt.yticks(tile_ticks)
    # plt.xlabel("Training Steps (million steps)")
    # plt.ylabel("Highest Max Tile")
    # plt.title("Highest Max Tile vs. Training Steps (Bar Chart)")
    # plt.tight_layout()
    # plt.savefig("graphs/high_tile_vs_steps_bar.png", dpi=300, bbox_inches="tight")
    # plt.close()

    plt.figure()
    plt.bar(range(len(step_counts)), highest_tiles)
    plt.yscale("log", base=2)
    plt.xticks(range(len(step_counts)), step_counts)
    plt.yticks(tile_ticks, tile_ticks)
    plt.xlabel("Training Steps (million steps)")
    plt.ylabel("Highest Max Tiles")
    plt.title("Highest Max Tile vs. Training Steps")
    plt.tight_layout()
    plt.savefig("graphs/high_tile_vs_steps_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
