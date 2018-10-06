import tensorflow as tf
import numpy as np
import os
import matplotlib
from aux_functions import getImage, create_sprite_image, vector_to_matrix, invert_grayscale, gen_dict
from tensorflow.contrib.slim.python.slim.learning import train_step
import shutil
matplotlib.use('Agg')
import matplotlib.pyplot as plt
slim = tf.contrib.slim
import tensorflow.contrib.slim.nets
inception = tf.contrib.slim.nets.inception
vgg = tf.contrib.slim.nets.vgg
resnet_v1 = tf.contrib.slim.nets.resnet_v1
resnet_v2 = tf.contrib.slim.nets.resnet_v2
lenet = tf.contrib.slim.nets.lenet

def get_evaluation_inception(model_name,
                             data_directory,
                             path_to_val_file,
                             path_to_labels_file,
                             bsize,
                             train_log_dir,
                             initial_checkpoint):

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    # Create model and obtain the predictions:
    with graph.as_default():
        name_dict, nclass = gen_dict(data_directory, path_to_labels_file)

        vlabel, vimage = getImage(path_to_val_file, nclass)

        vimageBatch, vlabelBatch = tf.train.shuffle_batch(
            [vimage, vlabel], batch_size=bsize,
            capacity=2000,
            min_after_dequeue=1000)

        with sess.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.run(tf.global_variables_initializer())
            batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
            vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])

            if model_name == 'inception_v1':
                with tf.variable_scope("InceptionV1") as scope:
                    vlogits, vend_points = inception.inception_v1(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=False)
            elif model_name == 'inception_v2':
                with tf.variable_scope("InceptionV2") as scope:
                    vlogits, vend_points = inception.inception_v2(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=False)
            elif model_name == 'inception_v3':
                with tf.variable_scope("InceptionV3") as scope:
                    vlogits, vend_points = inception.inception_v3(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=False)

        predictions_validation = vend_points['Predictions']
        vcorrect_prediction = tf.equal(tf.argmax(predictions_validation, 1), tf.argmax(vbatch_ys, 1))

        accuracy_validation = tf.reduce_mean(tf.cast(vcorrect_prediction, tf.float32))

        tf.losses.softmax_cross_entropy(vbatch_ys, vlogits)

        total_loss = tf.losses.get_total_loss()

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map\
                ({
                    "accuracy": slim.metrics.accuracy(predictions, labels),
                    "mse": slim.metrics.mean_squared_error(predictions, labels)
                 })

        # Define the summaries to write:
        for metric_name, metric_value in metrics_to_values.iteritems():
            tf.summary.scalar(metric_name, metric_value)

        # We'll evaluate 1000 batches:
        num_evals = 1000

        # Evaluate every 10 minutes:
        slim.evaluation_loop(
            '',
            train_log_dir,
            train_log_dir,
            num_evals=num_evals,
            eval_op=names_to_updates.values(),
            summary_op=tf.summary.merge_all(),
            eval_interval_secs=10)


def get_evaluation_lenet(data_directory,
                         path_to_val_file,
                         path_to_labels_file,
                         bsize,
                         train_log_dir,
                         initial_checkpoint):

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    # Create model and obtain the predictions:
    with graph.as_default():
        name_dict, nclass = gen_dict(data_directory, path_to_labels_file)

        vlabel, vimage = getImage(path_to_val_file, nclass)

        vimageBatch, vlabelBatch = tf.train.shuffle_batch(
            [vimage, vlabel], batch_size=bsize,
            capacity=2000,
            min_after_dequeue=1000)

        with sess.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            sess.run(tf.global_variables_initializer())

            vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])

            with tf.variable_scope("LeNet") as scope:
                vlogits, vend_points = lenet.lenet(vbatch_xs,
                                                   num_classes=2,
                                                   is_training=False)

        predictions_validation = vend_points['Predictions']
        vcorrect_prediction = tf.equal(tf.argmax(predictions_validation, 1), tf.argmax(vbatch_ys, 1))

        accuracy_validation = tf.reduce_mean(tf.cast(vcorrect_prediction, tf.float32))

        tf.losses.softmax_cross_entropy(vbatch_ys, vlogits)

        total_loss = tf.losses.get_total_loss()

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map\
                ({
                    "accuracy": slim.metrics.accuracy(predictions, labels),
                    "mse": slim.metrics.mean_squared_error(predictions, labels)
                 })

        # Define the summaries to write:
        for metric_name, metric_value in metrics_to_values.iteritems():
            tf.summary.scalar(metric_name, metric_value)

        # We'll evaluate 1000 batches:
        num_evals = 1000

        # Evaluate every 10 minutes:
        slim.evaluation_loop(
            '',
            train_log_dir,
            train_log_dir,
            num_evals=num_evals,
            eval_op=names_to_updates.values(),
            summary_op=tf.summary.merge_all(),
            eval_interval_secs=10)