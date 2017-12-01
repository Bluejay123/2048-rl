"""Script to play a single game from a checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from py_2048_rl.game import play
from py_2048_rl.learning import learning
from py_2048_rl.learning.model import FeedModel

import sys

import tensorflow as tf
import numpy as np


def average_score(strategy):
  """Plays 100 games, returns average score."""

  scores = []
  results = []
  for _ in range(100):
    score, _, game_over = play.play(strategy, verbose=False, allow_unavailable_action=False)
    scores.append(score)
    results.append(0 if game_over == True else 1)

  return np.mean(scores), np.mean(results)

def random_strategy(_, actions):
  """Strategy that always chooses actions at random."""
  return np.random.choice(actions)

def static_preference_strategy(_, actions):
  """Always prefer left over up over right over top."""
  return min(actions)

def highest_reward_strategy(state, actions):
  """Strategy that always chooses the action of highest immediate reward.

  If there are any ties, the strategy prefers left over up over right over down.
  """

  sorted_actions = np.sort(actions)[::-1]
  rewards = map(lambda action: Game(np.copy(state)).do_action(action),
                sorted_actions)
  action_index = np.argsort(rewards, kind="mergesort")[-1]
  return sorted_actions[action_index]

def make_greedy_strategy(train_dir, verbose=False):
  """Load the latest checkpoint from train_dir, make a greedy strategy."""

  session = tf.Session()
  model = FeedModel()
  saver = tf.train.Saver()
  saver.restore(session, tf.train.latest_checkpoint(train_dir))

  get_q_values = learning.make_get_q_values(session, model)
  greedy_strategy = play.make_greedy_strategy(get_q_values, verbose)

  return greedy_strategy


def play_single_game(train_dir):
  """Play a single game using the latest model snapshot in train_dir."""

  s, _, _ = play.play(make_greedy_strategy(train_dir, True), verbose=True,
                   allow_unavailable_action=False)
  # s, _, _ = play.play(random_strategy,
  #                 allow_unavailable_action=False)
  print(s)


def print_average_score(train_dir):
  """Prints the average score of 100 games."""

  # score, result = average_score(random_strategy)
  score, result = average_score(make_greedy_strategy(train_dir))
  print('Average Score: {}, Win rate: {}\n'.format(score, result))
  # print("Average Score: ", average_score(random_strategy))



def main(args):
  """Main function."""

  if len(args) != 3:
    print("Usage: %s (single|avg) train_dir" % args[0])
    sys.exit(1)

  _, mode, train_dir = args

  if mode == "single":
    play_single_game(train_dir)
  elif mode == "avg":
    print_average_score(train_dir)
  else:
    print("Unknown mode:", mode)


if __name__ == '__main__':
  tf.app.run()
