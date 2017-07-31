import os, argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util
from read_data_fedex import ReadDataFedex, Dataset
import math
import numpy as np

from sparse_naive import evaluation

dir = os.path.dirname(os.path.realpath(__file__))  # /Users/Louis/PycharmProjects/policy_approximation


def freeze_graph(model_folder):
    """
    Write a .pb file, frozen model initially present as a .checkpoint + .meta files in model_folder

    /!\ Make sure you define precisely the nodes you're interested in /!\

    If not sure about the nodes names:
    Write the graph as a text file using as_text=True in write_graph.
    Open the graph file, it will have some node entries written in plane text.
    Check the node name you are passing is in the graph file.

    :param model_folder: path to the saved model (.meta + checkpoint files etc.)
    :return: write the serialized frozen model to a .pb file
    """
    # We retrieve our checkpoint fullpath: VALID WHEN OPERATING ON THE SAME MACHINE
    # checkpoint = tf.train.get_checkpoint_state(model_folder)
    # input_checkpoint = checkpoint.model_checkpoint_path
    # input_checkpoint = "/Users/Louis/PycharmProjects/policy_approximation/trained_graph/model.ckpt-200000"

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/Users/Louis/PycharmProjects/policy_approximation/trained_graph"
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = ["softmax_linear/add"]

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(
        input_checkpoint + ".meta",
        clear_devices=clear_devices
    )

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
                    sess,  # The session is used to retrieve the weights
                    input_graph_def,  # The graph_def is used to retrieve the nodes
                    output_node_names  # The output node names are used to select the usefull nodes
                )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(frozen_graph_filename):
    """
    From the frozen file obtained from the above function freeze_graph()
    :param frozen_graph_filename: .pb file
    :return:
    """
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':

    """
    # Freeze a model using the above defined function freeze_graph()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder",
                        default="/Users/Louis/PycharmProjects/policy_approximation/trained_graph" ,
                        type=str, help="Model folder to export"
                        )
    args = parser.parse_args()

    freeze_graph(args.model_folder)
    """
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="trained_graph/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # Retrieve inputs and desired outputs
    indices_pl = graph.get_operation_by_name('prefix/input/Placeholder').outputs[0]
    values_pl = graph.get_operation_by_name('prefix/input/Placeholder_1').outputs[0]
    shape_pl = graph.get_operation_by_name('prefix/input/Placeholder_2').outputs[0]

    logits = graph.get_operation_by_name('prefix/softmax_linear/add').outputs[0]
    prediction  = tf.argmax(logits, axis=1)

    # Data generation
    whole_data = ReadDataFedex()
    percentage_training = 0.7
    num_examples = whole_data.num_examples

    test_entries = whole_data.entries[:math.floor(percentage_training*num_examples)]
    test_labels = whole_data.labels[:math.floor(percentage_training * num_examples):]

    data_set_test = Dataset(test_entries, test_labels)

    indices_feed, values_feed, shape_feed, labels_feed = data_set_test.next_sp_batch(
        math.floor(0.7*num_examples -1),
        shuffle=False
    )

    with tf.Session(graph=graph) as sess:
        # Note: we didn't initialize/restore anything, everything is stored in the graph_def
        logits_out = sess.run([prediction], feed_dict={
            indices_pl:indices_feed,
            values_pl:values_feed,
            indices_pl:indices_feed,
            shape_pl:shape_feed
        })
        print("DATA CHECKING")
        print("First 10 entries     ", test_labels[:10])
        print("logits_out[0] shape  ", logits_out[0].shape)
        print("First 10 predictions ", logits_out[0][:10])
        print("First 10 labels      ", labels_feed[:10])
        errors = np.asarray(logits_out[0] - labels_feed)
        print("Accuracy on the test set ", np.count_nonzero(errors == 0))
        # print("Shape of the logits ", logits_out[0].shape)
        # print("Prediction : ", logits_out[1])
        # print("Label      : ", labels_feed)