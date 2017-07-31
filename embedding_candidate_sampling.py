# ==============================================================================
# Copyright 2017 Louis Douge
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Data
import pickle
import pandas as pd

import numpy as np
import math
import tensorflow as tf

PATH_TO_DATA = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned_no_0.data"
data_index = 0

# Data prep
# Location to ID lookup and vice-versa
with open("DATA/postal_codes_fedex", 'rb') as f:
    id_to_pc = pickle.load(f)

pc_to_id = {j:i for i, j in enumerate(id_to_pc)}

# print(id_to_pc[1])
# print(pc_to_id[10003])

# "sentences" of locations
def get_context_target(window):
    """
    Function used to generate the data for the locations vectorial representation (skip_gram_locations.npz)
    :param window, integer. Gives the number of preceding words giving the context of the target word
    :return:
    """
    raw_data = pd.read_csv(
        PATH_TO_DATA,
        header=0,
        delim_whitespace=True
    )

    print(len(set(raw_data['PostalCode'][:90000])))

    stop_order_col = raw_data["StopOrder"]  # stop order column
    start_tour_indices = list(stop_order_col[stop_order_col == 1 ].index)  # start of tours indices
    range_tour = [1, len(start_tour_indices)-1]  # tours collected
    print(start_tour_indices)

    context_words = []
    target_word = []

    for start, end in itertools.zip_longest(
                    start_tour_indices[range_tour[0]-1:range_tour[1]],
                    start_tour_indices[range_tour[0]:range_tour[1]+1],
                    fillvalue=0
            ):

        # retrieve sentence
        sentence = list(raw_data['PostalCode'][start:end])
        len_sentence = len(sentence)

        for i in range(window, len_sentence):
            for j in range(1, window+1):
                context_words.append(pc_to_id[sentence[i-j]])
                target_word.append(pc_to_id[sentence[i]])  # label of sentence[i-j]

    # print(context_words[:10])
    # print(target_word[:10])
    # print(len(target_word)/2)
    # print(id_to_pc[target_word[17]])

    """
    np.savez_compressed('/Users/Louis/PycharmProjects/policy_approximation/DATA/skip_gram_locations',
                        context=context_words,
                        target=target_word
                        )
    """


def generate_batch(btch_size, ctxt, target):
    global data_index
    nb_samples = len(target)
    assert data_index < nb_samples

    if data_index > nb_samples - btch_size:  # end of epoch
        batch = np.asarray(ctxt[data_index:])
        labels = np.asarray(target[data_index:])
        print(data_index)
        print("before append", batch)
        batch = np.append(batch, context[0:(batch_size - (nb_samples - data_index))])
        labels = np.append(labels, target[0:(batch_size - (nb_samples - data_index))])
        print("after append", batch)
        print("should append ", context[0:(batch_size - (nb_samples - data_index))] )

        print("Epoch completed")
        data_index = ((data_index + btch_size) % nb_samples) - 1  # new epoch

    else:
        batch = context[data_index:data_index + btch_size]
        labels = target[data_index:data_index + btch_size]
        data_index += (btch_size + 1)

    return batch, np.reshape(labels, (btch_size, 1))

if __name__ == "__main__":

    # From 2 word skip gram
    # embeddings characteristics
    raw_data = pd.read_csv(
        PATH_TO_DATA,
        header=0,
        delim_whitespace=True
    )
    vocabulary_size = len(set(raw_data['PostalCode']))
    print("Vocab size ", vocabulary_size)

    loaded = np.load('DATA/skip_gram_locations.npz')
    context = np.asarray(loaded['context'])
    target_word = np.asarray(loaded['target'])
    print(len(target_word))

    num_samples = len(context)
    embedding_size = 128
    batch_size = 10

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64  # Number of negative examples to sample.

    graph = tf.Graph()
    with graph.as_default():
        # Placeholders for inputs
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Compute the NCE loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size)
        )

        # We use the SGD optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    # Begin training.
    num_steps = 100001
    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size,
                context,
                target_word
            )
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print(loss_val)
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0



