import matplotlib.pyplot as plt
import numpy as np

RANDOM_STRATEGY='random_strategy'
HIGHEST_REWARD_STRATEGY='highest_immediate_reward'
FULLY_CONNECTED='fully_connected'
EXPECTIMAX='expectimax'

def main():
    stats = {}
    stats[RANDOM_STRATEGY] = (1045.36, 0.0)
    stats[HIGHEST_REWARD_STRATEGY] =  (3228.0, 0.0)
    stats[EXPECTIMAX] = (19135.08, 0.0)
    stats[FULLY_CONNECTED] = (2686.5, 0.0)

    category = np.arange(len(stats))
    scores = [x[0] for x in stats.values()]
    winrate = [x[1] for x in stats.values()]
    plt.bar(category, scores, alpha=0.5)
    # plt.bar(category, winrate, alpha=0.5, color='g', label='win rate')
    plt.xticks(category, stats.keys())
    plt.title('Average Score')
    plt.show()

    plt.bar(category, winrate, alpha=0.5)
    # plt.bar(category, winrate, alpha=0.5, color='g', label='win rate')
    plt.xticks(category, stats.keys())
    plt.title('win rate')
    plt.show()





if __name__=='__main__':
    main()
