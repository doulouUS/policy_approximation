# Copyright 2017 Louis Douge
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import sys
import os.path

# Data
import read_data_fedex as rdf

import math
import tensorflow as tf

FLAGS = None

# Load data
NUM_CLASSES = 16077 # 46521
ENTRIES_FEAT = NUM_CLASSES  #  input are of the same shape as output


# ----------------------------------------------------------------------
#
#       Collect Data
#
# ----------------------------------------------------------------------

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# ----------------------------------------------------------------------
#
#       BUILD THE GRAPH
#
# ----------------------------------------------------------------------


def inference(entries, hidden_unit):
    """
    Build the graph for inference only.
    :param entries, placeholder from inputs
    :param hidden_unit, placeholder

    :return: softmax_linear, output logits
    """
    # TODO: after modifying placeholder_inputs, assemble indices, values and shape into a sparse matrix HERE
    with tf.name_scope('hidden'):
        weights = tf.Variable(
            tf.truncated_normal([ENTRIES_FEAT, hidden_unit],
                                stddev=1.0 / math.sqrt(float(ENTRIES_FEAT))),
            name='weights')
        variable_summaries(weights)
        biases = tf.Variable(tf.zeros([hidden_unit]),
                             name='biases'
                             )

        hidden = tf.nn.relu(tf.matmul(entries, weights) + biases)
    # TODO 2) implement matmul with sparse matrix (sparse weights for instance)?
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden_unit, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden_unit))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases'
                             )
        logits = tf.matmul(hidden, weights) + biases

    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='xentropy'
    )

    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    """Sets up the training Ops.
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.
    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer()  # we trust default parameters, no learning_rate
    # Create a variable to track the global step.
    global_step = tf.Variable(
        0,
        name='global_step',
        trainable=False
    )
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    # TODO Use this function with k greater than 1 to evaluate more generally your net's behavior
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))

# ----------------------------------------------------------------------
#
#       TRAIN THE MODEL
#
# ----------------------------------------------------------------------


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    entries_placeholder: entries placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    with tf.name_scope('input'):
        # TODO replace entries_placeholder by 3 placeholders: indices, values and shape
        entries_placeholder = tf.placeholder(tf.float32,
                                             shape=(batch_size, NUM_CLASSES),
                                             name='input-entries'
                                             )

        indices_placeholder = tf.placeholder(tf.int64)
        values_placeholder = tf.placeholder(tf.float32)
        shape_placeholder = tf.placeholder(tf.int32)

        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name="y-input")
    return entries_placeholder, labels_placeholder


def fill_feed_dict(data_set, entries_pl, labels_pl):
    # TODO are you sure of dataset's type?
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }
    Args:
    data_set: Dataset() object
    entries_pl: The entries placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    # TODO: after creating placeholders, transform entries_feed into COO coordinates and adapt the feed dictionary !!
    entries_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
    # entries_feed = tf.constant(entries_feed)
    # idx = tf.where(tf.not_equal(entries_feed, 0))
    # entries_feed = tf.SparseTensor(idx, tf.gather_nd(entries_feed, idx),entries_feed.get_shape())

    feed_dict = {
      entries_pl: entries_feed,
      labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            entries_placeholder,
            labels_placeholder,
            data_set
            ):
    """Runs one evaluation against the full epoch of data.
    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    entries_placeholder: The entries placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of entries and labels to evaluate, from
      Dataset().

    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    # TODO reassign on the correct amount of entries_train you want to evaluate your network
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   entries_placeholder,
                                   labels_placeholder
                                   )
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of entries and labels for training and Test
    whole_data = rdf.ReadDataFedex()
    percentage_training = 0.7
    num_examples = whole_data.num_examples

    # Train and test data
    train_entries = whole_data.entries[:math.floor(percentage_training*num_examples)]
    train_labels = whole_data.labels[:math.floor(percentage_training*num_examples)]

    test_entries = whole_data.entries[math.floor(percentage_training*num_examples):]
    test_labels = whole_data.labels[math.floor(percentage_training * num_examples):]

    data_set_train = rdf.Dataset(train_entries, train_labels)
    data_set_test = rdf.Dataset(test_entries, test_labels)

    # Tell TensorFlow that the model will be built into the default Graph.

    with tf.Graph().as_default():
        # Generate placeholders for the entries and labels.
        entries_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size
        )

        # Build a Graph that computes predictions from the inference model.
        logits = inference(entries_placeholder,
                           FLAGS.hidden
                           )

        # Add to the Graph the Ops for loss calculation.
        loss_ = loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss_, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # TODO explore the possibility of creating check points
        """
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        """

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.

        for step in range(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of entries and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_set_train,
                                       entries_placeholder,
                                       labels_placeholder
                                       )

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss_],
                                     feed_dict=feed_dict
                                     )

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            full_test = [1000, 5000]
            if step % 100 == 0 and step not in full_test:
                # Print status to stdout.
                print('-- Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            elif step in full_test:
                # Print status to stdout.
                print('-- Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                """
                # Save a checkpoint and evaluate the model periodically.
                if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.log_dir,
                                                   'model.ckpt'
                                                   )
                saver.save(sess, checkpoint_file, global_step=step)
                """
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        entries_placeholder,
                        labels_placeholder,
                        data_set_train)

                # TODO no validation set yet
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        entries_placeholder,
                        labels_placeholder,
                        data_set_test
                        )
    # TODO 5) Read up on SSE4.2 Instructions and GPU computations too
    # TODO 6) Only after having real performance results, you will use a more complex model (RNN for instance)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Percentage for training
    parser.add_argument(
        '--percentage_train',
        type=float,
        default=0.7,
        help='Percentage of data used for training.'
    )

    # Learning rate command line
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )

    # Max steps
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )

    # Number of hidden units
    # TODO nb of units is too low here
    parser.add_argument(
        '--hidden',
        type=int,
        default=1000,
        help='Number of units in hidden layer 1.'
    )

    # Batch size
    # TODO maybe do not evenly divide dataset...
    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    if sys.platform == 'darwin':
        parser.add_argument(
            '--input_data_dir',
            type=str,
            default='/Users/Louis/PycharmProjects/policy_approximation/',
            help='Directory to put the input data.'
        )
    elif sys.platform == 'linux':
        parser.add_argument(
            '--input_data_dir',
            type=str,
            default='home/louis/Documents/Research/policy_approximation-master/logs',
            help='Directory to put the input data.'
        )

    if sys.platform == 'darwin':
        parser.add_argument(
            '--log_dir',
            type=str,
            default='/Users/Louis/PycharmProjects/policy_approximation/logs/log_adam_30_btch',
            help='Directory to put the log data.'
        )

    elif sys.platform == 'linux':
        parser.add_argument(
            '--log_dir',
            type=str,
            default='home/Research/policy_approximation/logs/log_adam_30_btch',
            help='Directory to put the log data.'
        )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
