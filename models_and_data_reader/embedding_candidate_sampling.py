# ==============================================================================
# Copyright 2017 Louis Douge
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import itertools
import random

# Data
import pickle
import pandas as pd

import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


PATH_TO_DATA = "/Users/Louis/PycharmProjects/policy_approximation/DATA/fedex_pc_cleaned_no_0.data"
data_index = 0

LOG_DIR = '/Users/Louis/PycharmProjects/policy_approximation/DATA/embeddings_validation/log_30dim_10btch'
PROJ_DIR = '/Users/Louis/PycharmProjects/policy_approximation/DATA/embeddings_validation/'
# Data prep
# Location to ID lookup and vice-versa

with open("DATA/postal_codes_fedex", 'rb') as f:
    id_to_pc = pickle.load(f)

pc_to_id = {j: i for i, j in enumerate(id_to_pc)}


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


def write_metadata(id_to_pc_local):

    with open("/Users/Louis/PycharmProjects/policy_approximation/DATA/embeddings_validation/metadata.tsv", 'w') as f:
        for i in id_to_pc_local:
            f.write(
                str(i)+"\n"
            )


# "sentences" of locations
def get_context_target(window):
    """
    Function used to generate the data for the locations vectorial representation (skip_gram_locations.npz)
    :param window, integer. Gives the number of preceding words giving the context of the target word
    :return:
    """
    raw_data_loc = pd.read_csv(
        PATH_TO_DATA,
        header=0,
        delim_whitespace=True
    )

    raw_data_loc = raw_data_loc[raw_data_loc["FedExID"] == 868386]
    # data = raw_data_loc[raw_data_loc["Latitude"] < 1.31216860]
    # data = data[data["Longitude"] > 103.78200827]
    # raw_data_loc = data[data["Longitude"] < 103.8705063913]
    data_indices = list(raw_data_loc.index)
    id_to_pc_loc = sorted(list(set(raw_data_loc["PostalCode"])))

    # PC to ID on the involved locations
    pc_to_id_loc = {j: i for i, j in enumerate(id_to_pc_loc)}

    # Find tours index
    start_tour_index = []
    end_tour_index = []

    start_tour_index.append(data_indices[0])
    for i, j in zip(data_indices, range(0, len(data_indices) - 1)):
        if data_indices[j + 1] - i - 1 > 10:
            start_tour_index.append(data_indices[j+1])
            end_tour_index.append(i)

    end_tour_index.append(data_indices[-1])
    # print("START ", start_tour_index)
    # print("END ", end_tour_index)
    # print("nb of tour? ", len(end_tour_index))
    # print(len(raw_data_loc["PostalCode"]))
    # print("nb of diff locations ", len(set(raw_data_loc["PostalCode"])))

    context_words = []
    target_word_loc = []

    # retrieve for each entry, known jobs remaining to be done at the time of the entry in the current tour
    for start, end in zip(start_tour_index, end_tour_index):
        data_indices_tour = [i for i in raw_data_loc.index if i > start and i < end + 1]

        # retrieve sentence
        sentence = list(raw_data_loc.loc[data_indices_tour]["PostalCode"])
        len_sentence = len(sentence)

        for i in range(window, len_sentence-1):
            for j in range(1, window+1):
                context_words.append(pc_to_id_loc[sentence[i-j]])
                target_word_loc.append(pc_to_id_loc[sentence[i]])  # label of sentence[i-j]

            context_words.append(pc_to_id_loc[sentence[i+1]])
            target_word_loc.append(pc_to_id_loc[sentence[i]])

        # print("sentence ", sentence)
        print("target word ", target_word_loc)
        print("context word ", context_words)
    # print("end tour idx ", end_tour_index)

    return context_words, target_word_loc, id_to_pc_loc, pc_to_id_loc


    # np.savez_compressed('/Users/Louis/PycharmProjects/policy_approximation/DATA/skip_gram_locations_2bef_1af_reduced',
    #                     context=context_words,
    #                     target=target_word_loc
    #                    )


def generate_batch(btch_size, ctxt, target):
    global data_index
    nb_samples = len(target)
    assert data_index <= nb_samples

    if data_index > nb_samples - btch_size-1:  # end of epoch
        batch = np.asarray(ctxt[data_index:])
        labels = np.asarray(target[data_index:])
        # print(data_index)
        # print("before append", batch)
        batch = np.append(batch, ctxt[0:(btch_size - (nb_samples - data_index))])
        labels = np.append(labels, target[0:(btch_size - (nb_samples - data_index))])
        # print("after append", batch)
        # print("should append ", context[0:(batch_size - (nb_samples - data_index))] )

        # print("Epoch completed")
        data_index = ((data_index + btch_size) % nb_samples)  # new epoch

    else:
        batch = ctxt[data_index:data_index + btch_size]
        labels = target[data_index:data_index + btch_size]
        data_index += btch_size

    return batch, np.reshape(labels, (btch_size, 1))

if __name__ == "__main__":

    # From 2 word skip gram
    # embeddings characteristics

    # TODO: retrieve reduced vocabulary

    # FULL DATASET
    raw_data = pd.read_csv(
        PATH_TO_DATA,
        header=0,
        delim_whitespace=True
    )
    vocabulary_size = len(set(raw_data['PostalCode']))
    print("Vocab size ", vocabulary_size)

    loaded = np.load('DATA/skip_gram_locations_2bef_1af.npz')
    context = np.asarray(loaded['context'])

    target_word = np.asarray(loaded['target'])
    print(len(target_word))

    # REDUCED
    """

    # Encoding and decoding
    context, target_word, id_to_pc_loc, pc_to_id_loc = get_context_target(2)
    print("Context ", context)
    vocabulary_size = len(id_to_pc_loc)
    print("Vocabulary size ", vocabulary_size)
    """

    # TODO: change the vocabulary encoding and decoding
    # write metadata.tsv
    # write_metadata(id_to_pc)
    write_metadata(id_to_pc)

    valid_size = 8  # nb of locations to check for the most similar locations
    valid_examples = np.array(random.sample(range(1000), valid_size))

    # TODO: maybe reduce batch size
    num_samples = len(context)
    embedding_size = 30
    batch_size = 10

    # TODO: change num_sampled according to the dataset used (45 for REDUCED and up to 16077 for FULL DATASET
    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 6  # Random set of words to evaluate similarity on.
    valid_window = 45
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 500  # Number of negative examples to sample.

    graph = tf.Graph()
    with graph.as_default():
        # Placeholders for inputs
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
            name="embeddings_de_dingue_postal_codes"
        )
        variable_summaries(embeddings)
        # embedded_word_ids = tf.gather(embeddings, range(0,16077))

        # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
        config = projector.ProjectorConfig()

        # You can add multiple embeddings. Here we add only one.
        embedding_projection = config.embeddings.add()
        embedding_projection.tensor_name = embeddings.name

        # Link this tensor to its metadata file (e.g. labels).
        embedding_projection.metadata_path = os.path.join(PROJ_DIR, 'metadata.tsv')

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Compute the NCE loss, using a sample of the negative labels each time.
        """
        loss = tf.nn.sampled_softmax_loss(weights=nce_weights,
                                          biases=nce_biases,
                                          labels=train_labels,
                                          inputs=embed,
                                          num_sampled=num_sampled,
                                          num_classes=vocabulary_size
                                          )
        """
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocabulary_size)
        )
        tf.summary.scalar('loss', loss)
        # We use the SGD optimizer.
        # optimizer = tf.train.AdamOptimizer().minimize(loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=.1).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
            normalized_embeddings, valid_dataset)
        similarity = tf.matmul(
            valid_embeddings, normalized_embeddings, transpose_b=True)

        """
        # Compute the euclidean distance btw minibatch ex and all embeddings
        valid_embeddings = tf.nn.embedding_lookup(
            embeddings, valid_dataset
        )
        # Calculate L1 Distance
        distance = tf.reduce_sum(
            tf.abs(
                tf.add(embeddings,
                       tf.negative(valid_embeddings)
                       )
            ),
            reduction_indices=1
        )
        """
        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Add variable initializer.
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

    # Begin training.
    num_steps = 700001
    with tf.Session(graph=graph) as session:

        # Use the same LOG_DIR where you stored your checkpoint.
        summary_writer = tf.summary.FileWriter(LOG_DIR, session.graph)

        # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
        # read this file during startup.
        projector.visualize_embeddings(summary_writer, config)

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

                # Update the events file.
                summary_str = session.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # TODO: change id_to_pc to id_to_pc_loc if using REDUCED dataset
            if step % 100000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = id_to_pc[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = id_to_pc[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

        saver.save(session,
                   os.path.join(LOG_DIR,
                                "model.ckpt"
                                ),
                   step
                   )

        # embeddings_final = np.asarray(session.run(normalized_embeddings))
        # np.save('/Users/Louis/PycharmProjects/policy_approximation/DATA/embeddings.trained', embeddings_final)

        # np.save('/Users/Louis/PycharmProjects/policy_approximation/DATA/embeddings_validation/embeddings_validation')


