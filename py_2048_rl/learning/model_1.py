"""Model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


NUM_TILES = 16
NUM_ACTIONS = 4

HIDDEN_SIZES = [128]

WEIGHT_INIT_SCALE = 0.01
INIT_LEARNING_RATE = 1e-4
LR_DECAY_PER_100K = 0.98
OPTIMIZER_CLASS = tf.train.AdamOptimizer
ACTIVATION_FUNCTION = tf.nn.relu

# Convolution parameters
CONV_FEATURES = 32
INCEPTION_FEATURES = 32


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class FeedModel(object):
  """Class to construct and collect all relevant tensors of the model."""

  def __init__(self):
    self.state_batch_placeholder = tf.placeholder(
        tf.float32, shape=(None, NUM_TILES))
    self.targets_placeholder = tf.placeholder(tf.float32, shape=(None,))
    self.actions_placeholder = tf.placeholder(tf.int32, shape=(None,))
    self.placeholders = (self.state_batch_placeholder,
                         self.targets_placeholder,
                         self.actions_placeholder)

    self.input_layer = tf.reshape(self.state_batch_placeholder, [4,-1])
    self.conv_a = tf.layers.conv2d(
      inputs=self.input_layer, 
      filters = 512, 
      kernel_size=[2,1],
      padding='valid'
      activation=tf.nn.relu)
    self.conv_b = tf.layers.conv2d(
      inputs=self.input_layer, 
      filters = 512, 
      kernel_size=[1,2],
      padding='valid'
      activation=tf.nn.relu)
    self.conv_aa = tf.layers.conv2d(
      inputs=self.conv_a, 
      filters = 4096, 
      kernel_size=[2,1],
      padding='valid'
      activation=tf.nn.relu)
    self.conv_ab = tf.layers.conv2d(
      inputs=self.conv_a, 
      filters = 4096, 
      kernel_size=[1,2],
      padding='valid'
      activation=tf.nn.relu)
    self.conv_ba = tf.layers.conv2d(
      inputs=self.conv_b, 
      filters = 4096, 
      kernel_size=[2,1],
      padding='valid'
      activation=tf.nn.relu)
    self.conv_bb = tf.layers.conv2d(
      inputs=self.conv_b, 
      filters = 4096, 
      kernel_size=[1,2],
      padding='valid'
      activation=tf.nn.relu)
    _ = tf.concat([Flatten()(x) for x in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]])
    l_out = tf.layers.dense(_, num_units=1,  nonlinearity=None)
    



    self.weights, self.biases, self.activations = build_inference_graph(
        self.state_batch_placeholder, HIDDEN_SIZES)
    self.q_values = self.activations[-1]
    self.loss = build_loss(self.q_values, self.targets_placeholder,
                     self.actions_placeholder)
    self.train_op, self.global_step, self.learning_rate = (
        build_train_op(self.loss))

    tf.scalar_summary("Average Target",
                      tf.reduce_mean(self.targets_placeholder))
    tf.scalar_summary("Learning Rate", self.learning_rate)
    tf.scalar_summary("Loss", self.loss)
    tf.histogram_summary("States", self.state_batch_placeholder)
    tf.histogram_summary("Targets", self.targets_placeholder)

    self.init = tf.initialize_all_variables()
    self.summary_op = tf.merge_all_summaries()


def build_inference_graph(state_batch, hidden_sizes):
  """Build inference model.

  Args:
    state_batch: [batch_size, NUM_TILES] float Tensor.
    hidden_sizes: Array of numbers where len(hidden_sizes) is the number of
        hidden layers and hidden_sizes[i] is the number of hidden units in the
        ith layer.

  Returns:
    q_values: Output tensor with the computed Q-Values.
  """
  weights = []
  biases = []
  activations = []

  input_batch = tf.reshape(state_batch, [-1, 4, 4, 1])

  # Incpetion layer

  filter_tensor_inception, hidden_output_inception = build_conv_layer(
      "Inception", input_batch, [1, 1, 1, INCEPTION_FEATURES],
      ACTIVATION_FUNCTION)

  weights.append(filter_tensor_inception)
  activations.append(hidden_output_inception)

  # Conv layer
  input_batch = hidden_output_inception

  filter_tensor_col, hidden_output_col = build_conv_layer(
      "ColumnConv", input_batch, [4, 1, INCEPTION_FEATURES, CONV_FEATURES],
      ACTIVATION_FUNCTION)
  filter_tensor_row, hidden_output_row = build_conv_layer(
      "RowConv", input_batch, [1, 4, INCEPTION_FEATURES, CONV_FEATURES],
      ACTIVATION_FUNCTION)

  weights += [filter_tensor_col, filter_tensor_row]
  activations += [hidden_output_col, hidden_output_row]

  # Input to fully connected layer is all conv features concatenated
  input_batch = tf.concat(1, [
      tf.reshape(hidden_output_col, [-1, 4 * CONV_FEATURES]),
      tf.reshape(hidden_output_row, [-1, 4 * CONV_FEATURES])])
  input_size = 8 * CONV_FEATURES

  for i, hidden_size in enumerate(hidden_sizes):
    weights_i, biases_i, hidden_output_i = build_fully_connected_layer(
        'hidden' + str(i), input_batch, input_size, hidden_size,
        ACTIVATION_FUNCTION)

    weights.append(weights_i)
    biases.append(biases_i)
    activations.append(hidden_output_i)

    input_batch = hidden_output_i
    input_size = hidden_size

  weights_qvalues, biases_qvalues, output = build_fully_connected_layer(
      'q_values', input_batch, input_size, NUM_ACTIONS)

  weights.append(weights_qvalues)
  biases.append(biases_qvalues)
  activations.append(output)

  return weights, biases, activations


def build_conv_layer(name, input_batch, filter_size, activation_function):
  """Builds a conv layer.

  Args:
    name: Name of the layer (-> Variable scope).
    input_batch: [batch_size, height, width, 1] Tensor that this layer is
        connected to.
    filter_size: [filter_height, filter_width, input_depth, output_depth]
    activation_function: Activation Function to use.

  Returns:
    [filter_tensor, output_batch], where output_batch is the
    [batch_size, layer_height, layer_width, output_depth] output Tensor.
  """

  with tf.name_scope(name):
    filter_tensor = tf.Variable(
        tf.truncated_normal(filter_size), name="filter")
    output_batch = tf.nn.conv2d(input_batch, filter_tensor,
                                strides=[1, 1, 1, 1],
                                padding="VALID")

    tf.histogram_summary("Filter " + name, filter_tensor)
    tf.histogram_summary("Activations " + name, output_batch)

    return filter_tensor, output_batch


def build_fully_connected_layer(name, input_batch, input_size, layer_size,
                                activation_function=lambda x: x):
  """Builds a fully connected layer.

  Args:
    name: Name of the layer (-> Variable scope).
    input_batch: [batch_size, input_size] Tensor that this layer is
        connected to.
    input_size: Number of input units.
    layer_size: Number of units in this layer.
    activation_function: Activation Function to use. Defaults to none.

  Returns:
    [weights, biases, output_batch], where output_batch is the
    [batch_size, layer_size] output Tensor.
  """
  with tf.name_scope(name):
    weights = tf.Variable(tf.truncated_normal([input_size, layer_size],
                                              stddev=WEIGHT_INIT_SCALE),
                          name='weights')
    biases = tf.Variable(tf.zeros([layer_size]), name='biases')
    output_batch = activation_function(tf.matmul(input_batch, weights) + biases)

    tf.histogram_summary("Weights " + name, weights)
    tf.histogram_summary("Biases " + name, biases)
    tf.histogram_summary("Activations " + name, output_batch)

    return weights, biases, output_batch


def build_loss(q_values, targets, actions):
  """Calculates the loss from the Q-Values, targets and actions.

  Args:
    q_values: A [batch_size, NUM_ACTIONS] float Tensor. Contains the current
        computed Q-Values.
    targets: A [batch_size] float Tensor. Contains the current target Q-Values
        for the action taken.
    actions: A [batch_size] int Tensor. Contains the actions taken.

  Returns:
    loss: Loss tensor of type float.
  """
  # Get Q-Value prodections for the given actions
  batch_size = tf.shape(q_values)[0]
  q_value_indices = tf.range(0, batch_size) * NUM_ACTIONS + actions
  relevant_q_values = tf.gather(tf.reshape(q_values, [-1]), q_value_indices)

  # Compute L2 loss (tf.nn.l2_loss() doesn't seem to be available on CPU)
  return tf.reduce_mean(tf.pow(relevant_q_values - targets, 2))


def build_train_op(loss):
  """Sets up the training Ops.

  Args:
    loss: Loss tensor, from build_loss().

  Returns:
    train_op, global_step, learning_rate.
  """
  global_step = tf.Variable(0, name='global_step', trainable=False)
  learning_rate = tf.train.exponential_decay(
      INIT_LEARNING_RATE, global_step, 100000, LR_DECAY_PER_100K)

  optimizer = OPTIMIZER_CLASS(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step, learning_rate
