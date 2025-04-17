import matplotlib.pyplot as plt
import matplotlib
import argparse
import re
import os
import numpy as np

rstarting_game = re.compile(r"STARTING GAME #(\d+)")
reps_threshold = re.compile(r"eps_threshold=(0.\d+)")
ravg_score_chips_per_game = re.compile(r"avg_score_chips=(\d+\.\d+)")
ravg_chips_per_hand = re.compile(r"avg_reward_per_hand=(\d+\.\d+)")
rnn_freq = re.compile(r"freq_to_prob\(nn_hand_freq\) = {(.*)}")
rloss = re.compile(r"loss=\[((\d+\.\d+)(, )?)*\]")
card_types = {
    "high card": r"'HIGH_CARD': (\d+\.\d+)",
    "pair": r"'PAIR': (\d+\.\d+)",
    "two pair": r"'TWO_PAIR': (\d+\.\d+)",
    "three of a kind": r"'THREE_OF_A_KIND': (\d+\.\d+)",
    "straight": r"'STRAIGHT': (\d+\.\d+)",
    "full house": r"'FULL_HOUSE': (\d+\.\d+)",
    "flush": r"'FLUSH': (\d+\.\d+)",
    "four of a kind": r"'FOUR_OF_A_KIND': (\d+\.\d+)",
    "straight flush": r"'STRAIGHT_FLUSH': (\d+\.\d+)",
}
rcard_types = {l:re.compile(s) for l,s in card_types.items()}

def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_log_path")
    args = parser.parse_args()

    log_path = args.output_log_path
    out_dir = os.path.dirname(log_path)
    with open(log_path) as f:
        content = f.read()

    sstarting_game = rstarting_game.findall(content)
    seps_threshold = reps_threshold.findall(content)
    savg_score_chips_per_game = ravg_score_chips_per_game.findall(content)
    savg_chips_per_hand = ravg_chips_per_hand.findall(content)
    snn_freq = rnn_freq.findall(content)
    scard_types = {
        l: [r.search(s).group(1) if r.search(s) is not None else 0.0 for s in snn_freq]
        for l,r in rcard_types.items()
    }
    sloss = [i for l in rloss.findall(content) for i in l if i not in [", ", ""]]

    # sanity checks
    assert(
        len(sstarting_game)\
        == len(seps_threshold) \
        == len(savg_score_chips_per_game) \
        == len(savg_chips_per_hand)
    )

    game_index = [int(i) for i in sstarting_game]
    eps_threshold = [float(i) for i in seps_threshold]
    avg_score_chips_per_game = [float(i) for i in savg_score_chips_per_game]
    avg_chips_per_hand = [float(i) for i in savg_chips_per_hand]
    card_types = {l: [float(j) for j in i] for l, i in scard_types.items()}
    loss = [float(i) for i in sloss]


    plt.title("Average Scored Chips Per Game")
    plt.xlabel("Game #")
    plt.ylabel("Average Score Chips")
    plt.plot(game_index, avg_score_chips_per_game, label="Exponential Moving Avg")
    plt.plot(game_index, smooth(avg_score_chips_per_game, 0.99) , label="Smoothed EMA")
    plt.plot(
        game_index,
        np.array(eps_threshold) * max(avg_score_chips_per_game),
        label="Scaled Epsilon Threshold"
    )
    plt.legend()
    plt.savefig(f"{out_dir}/avg_score_chips_per_game.png")
    plt.clf()

    plt.title("Average Chips Per Hand")
    plt.xlabel("Game #")
    plt.ylabel("Chips")
    plt.plot(game_index, avg_chips_per_hand, label="Chips/Hand")
    plt.plot(game_index, smooth(avg_chips_per_hand, 0.99), label="Smoothed Chips/Hand")
    plt.savefig(f"{out_dir}/avg_chips_per_hand.png")
    plt.clf()

    plt.title("Hand Type Likelihoods")
    plt.xlabel("Game #")
    plt.ylabel("Percent")
    for l,pl in card_types.items():
        plt.plot(game_index[:len(pl)],pl[:len(game_index)],label=l)
    plt.legend()
    plt.savefig(f"{out_dir}/hand_percentages.png")
    plt.clf()

    plt.title("Loss")
    plt.xlabel("Iteration #")
    plt.ylabel("Loss")
    plt.plot(loss,label="loss")
    plt.plot(smooth(loss,0.99),label="smooth loss")
    plt.legend()
    plt.savefig(f"{out_dir}/loss.png")
    plt.clf()


if __name__ == "__main__":
    main()
