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
# lenet = tf.contrib.slim.nets.lenet


def run_inception_training(model_name,
                           data_directory,
                           path_to_train_file,
                           path_to_val_file,
                           path_to_labels_file,
                           bsize,
                           num_steps,
                           train_log_dir,
                           optimizer,
                           initial_checkpoint):

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with graph.as_default():
        name_dict, nclass = gen_dict(data_directory, path_to_labels_file)

        label, image = getImage(path_to_train_file, nclass)
        vlabel, vimage = getImage(path_to_val_file, nclass)

        imageBatch, labelBatch = tf.train.shuffle_batch(
            [image, label], batch_size=bsize,
            capacity=2000,
            min_after_dequeue=1000)

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

            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            if model_name == 'inception_v1':
                with tf.variable_scope("InceptionV1") as scope:
                    logits, end_points = inception.inception_v1(batch_xs,
                                                                num_classes=2,
                                                                is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = inception.inception_v1(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=True)
            elif model_name == 'inception_v2':
                with tf.variable_scope("InceptionV2") as scope:
                    logits, end_points = inception.inception_v2(batch_xs,
                                                                num_classes=2,
                                                                is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = inception.inception_v2(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=True)
            elif model_name == 'inception_v3':
                with tf.variable_scope("InceptionV3") as scope:
                    logits, end_points = inception.inception_v3(batch_xs,
                                                                num_classes=2,
                                                                is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = inception.inception_v3(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=True)

            predictions = end_points['Predictions']
            predictions_validation = vend_points['Predictions']  # -- for inception model use Predictions

            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(batch_ys, 1))
            vcorrect_prediction = tf.equal(tf.argmax(predictions_validation, 1), tf.argmax(vbatch_ys, 1))

            # get mean of all entries in correct prediction, the higher the better
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_validation = tf.reduce_mean(tf.cast(vcorrect_prediction, tf.float32))

            logits = tf.reshape(logits, [bsize, 2])

            tf.losses.softmax_cross_entropy(batch_ys, logits)

            total_loss = tf.losses.get_total_loss()

            train_tensor = slim.learning.create_train_op(total_loss, optimizer)

            embedding_size = 1024
            embedding_input = end_points
            # print(tf.shape(embedding_input))
            embedding = tf.Variable(tf.zeros([bsize, embedding_size]), name="Embedding_Tensor")
            assignment = embedding.assign()
            writer = tf.summary.FileWriter(train_log_dir + '/', sess.graph)
            config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            embedding_config = config.embeddings.add()
            embedding_config.tensor_name = embedding.name
            embedding_config.sprite.image_path = train_log_dir + '/sprite.png'
            embedding_config.metadata_path = train_log_dir + '/metadata.tsv'
            print(embedding_config.sprite.image_path)
            print(embedding_config.metadata_path)
            # Specify the width and height of a single thumbnail.
            embedding_config.sprite.single_image_dim.extend([224, 224])
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

        def train_step_fn(sess, *args, **kwargs):
            total_loss, should_stop = train_step(sess, *args, **kwargs)
            accuracy = sess.run([train_step_fn.accuracy])
            if train_step_fn.step % 50 == 0:
                # sess.run(assignment)
                accuracy_validation = sess.run([train_step_fn.accuracy_validation])
                # print('Step %s - Loss: %.2f Validation Accuracy: %.2f%%' %
                #       (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))
                # saver.save(sess, os.path.join(train_log_dir, "model.ckpt"), train_step_fn.step)

            train_step_fn.step += 1
            return [total_loss, should_stop]

        train_step_fn.step = 0
        train_step_fn.accuracy = accuracy
        train_step_fn.accuracy_validation = accuracy_validation

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', total_loss)
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        slim.learning.train(train_tensor,
                            train_log_dir,
                            number_of_steps=num_steps,
                            summary_op=summary_op,
                            train_step_fn=train_step_fn,
                            save_summaries_secs=20)
        print('completed training')

        coord.request_stop()
        coord.join(threads)


def run_lenet_training(data_directory,
                       path_to_train_file,
                       path_to_val_file,
                       path_to_labels_file,
                       bsize,
                       num_steps,
                       train_log_dir,
                       optimizer):

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with graph.as_default():
        name_dict, nclass = gen_dict(data_directory, path_to_labels_file)

        label, image = getImage(path_to_train_file, nclass)
        vlabel, vimage = getImage(path_to_val_file, nclass)

        imageBatch, labelBatch = tf.train.shuffle_batch(
            [image, label], batch_size=bsize,
            capacity=2000,
            min_after_dequeue=1000)

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

            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            with tf.variable_scope("LeNet") as scope:
                logits, end_points = lenet.lenet(batch_xs,
                                                 num_classes=2,
                                                 is_training=True)
                scope.reuse_variables()
                vlogits, vend_points = lenet.lenet(vbatch_xs,
                                                   num_classes=2,
                                                   is_training=True)

            predictions = end_points['Predictions']
            predictions_validation = vend_points['Predictions']  # -- for inception model use Predictions

            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(batch_ys, 1))
            vcorrect_prediction = tf.equal(tf.argmax(predictions_validation, 1), tf.argmax(vbatch_ys, 1))

            # get mean of all entries in correct prediction, the higher the better
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_validation = tf.reduce_mean(tf.cast(vcorrect_prediction, tf.float32))

            logits = tf.reshape(logits, [bsize, 2])

            tf.losses.softmax_cross_entropy(batch_ys, logits)

            total_loss = tf.losses.get_total_loss()

            train_tensor = slim.learning.create_train_op(total_loss, optimizer)

            embedding_size = 1024
            embedding_input = end_points['fc3']
            # print(tf.shape(embedding_input))
            embedding = tf.Variable(tf.zeros([bsize, embedding_size]), name="Embedding_Tensor")
            assignment = embedding.assign(embedding_input)
            writer = tf.summary.FileWriter(train_log_dir + '/', sess.graph)
            config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            embedding_config = config.embeddings.add()
            embedding_config.tensor_name = embedding.name
            embedding_config.sprite.image_path = train_log_dir + '/sprite.png'
            embedding_config.metadata_path = train_log_dir + '/metadata.tsv'
            print(embedding_config.sprite.image_path)
            print(embedding_config.metadata_path)
            # Specify the width and height of a single thumbnail.
            embedding_config.sprite.single_image_dim.extend([224, 224])
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

        def train_step_fn(sess, *args, **kwargs):
            total_loss, should_stop = train_step(sess, *args, **kwargs)
            accuracy = sess.run([train_step_fn.accuracy])
            if train_step_fn.step % 50 == 0:
                sess.run(assignment)
                accuracy_validation = sess.run([train_step_fn.accuracy_validation])
                print('Step %s - Loss %f' % (str(train_step_fn.step), total_loss))
                # print('Step %s - Loss: %.2f Validation Accuracy: %.2f%%' %
                #       (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))

            train_step_fn.step += 1
            return [total_loss, should_stop]

        train_step_fn.step = 0
        train_step_fn.accuracy = accuracy
        train_step_fn.accuracy_validation = accuracy_validation
        for end_point in end_points:
            x = end_points[end_point]
            tf.summary.histogram('activations/' + end_point, x)
        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)
        # summaries.add(tf.summary.scalar('accuracy', accuracy))
        tf.summary.scalar('accuracy', accuracy)
        # summaries.add(tf.summary.scalar('loss', total_loss))
        tf.summary.scalar('loss', total_loss)
        summary_op = tf.summary.merge_all()

        slim.learning.train(train_tensor,
                            train_log_dir,
                            number_of_steps=num_steps,
                            summary_op=summary_op,
                            train_step_fn=train_step_fn,
                            save_summaries_secs=20)
        print('completed training')

        to_visualise = vbatch_xs
        to_visualise = vector_to_matrix(to_visualise, 32, 32)
        to_visualise = invert_grayscale(to_visualise)
        sprite_image = create_sprite_image(to_visualise)
        plt.imsave(train_log_dir + '/sprite.png', sprite_image, cmap='gray')
        print(train_log_dir + '/sprite.png')
        plt.imshow(sprite_image, cmap='gray')
        with open(train_log_dir + '/metadata.tsv', 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(batch_ys):
                f.write("%d\t%s\n" % (index, name_dict[tuple(label)]))
        print('completed sprite and metadata creation')

        coord.request_stop()
        coord.join(threads)


def run_resnet_v1_training(model_name,
                           data_directory,
                           path_to_train_file,
                           path_to_val_file,
                           path_to_labels_file,
                           bsize,
                           num_steps,
                           train_log_dir,
                           optimizer,
                           initial_checkpoint):

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with graph.as_default():
        name_dict, nclass = gen_dict(data_directory, path_to_labels_file)

        label, image = getImage(path_to_train_file, nclass)
        vlabel, vimage = getImage(path_to_val_file, nclass)

        imageBatch, labelBatch = tf.train.shuffle_batch(
            [image, label], batch_size=bsize,
            capacity=2000,
            min_after_dequeue=1000)

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

            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            if model_name == 'resnet_v1_50':
                with tf.variable_scope("resnet_v1_50") as scope:
                    logits, end_points = resnet_v1.resnet_v1_50(batch_xs,
                                                                num_classes=2,
                                                                is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = resnet_v1.resnet_v1_50(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=True)
            elif model_name == 'resnet_v1_101':
                with tf.variable_scope("resnet_v1_101") as scope:
                    logits, end_points = resnet_v1.resnet_v1_101(batch_xs,
                                                                 num_classes=2,
                                                                 is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = resnet_v1.resnet_v1_101(vbatch_xs,
                                                                   num_classes=2,
                                                                   is_training=True)
            elif model_name == 'resnet_v1_152':
                with tf.variable_scope("resnet_v1_152") as scope:
                    logits, end_points = resnet_v1.resnet_v1_152(batch_xs,
                                                                 num_classes=2,
                                                                 is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = resnet_v1.resnet_v1_152(vbatch_xs,
                                                                   num_classes=2,
                                                                   is_training=True)
            elif model_name == 'resnet_v1_200':
                with tf.variable_scope("resnet_v1_200") as scope:
                    logits, end_points = resnet_v1.resnet_v1_200(batch_xs,
                                                                 num_classes=2,
                                                                 is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = resnet_v1.resnet_v1_200(vbatch_xs,
                                                                   num_classes=2,
                                                                   is_training=True)

            predictions = end_points['predictions']
            predictions_validation = vend_points['predictions']  # -- for resnet model use predictions

            predictions = tf.squeeze(predictions)
            predictions_validation = tf.squeeze(predictions_validation)

            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(batch_ys, 1))
            vcorrect_prediction = tf.equal(tf.argmax(predictions_validation, 1), tf.argmax(vbatch_ys, 1))

            # get mean of all entries in correct prediction, the higher the better
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_validation = tf.reduce_mean(tf.cast(vcorrect_prediction, tf.float32))

            logits = tf.reshape(logits, [bsize, 2])

            tf.losses.softmax_cross_entropy(batch_ys, logits)

            total_loss = tf.losses.get_total_loss()

            train_tensor = slim.learning.create_train_op(total_loss, optimizer)

        def train_step_fn(sess, *args, **kwargs):
            total_loss, should_stop = train_step(sess, *args, **kwargs)
            accuracy = sess.run([train_step_fn.accuracy])
            if train_step_fn.step % 50 == 0:
                # sess.run(assignment)
                accuracy_validation = sess.run([train_step_fn.accuracy_validation])
                # print('Step %s - Loss: %.2f Validation Accuracy: %.2f%%' %
                #       (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))
                # saver.save(sess, os.path.join(train_log_dir, "model.ckpt"), train_step_fn.step)

            train_step_fn.step += 1
            return [total_loss, should_stop]

        train_step_fn.step = 0
        train_step_fn.accuracy = accuracy
        train_step_fn.accuracy_validation = accuracy_validation

        summaries.add(tf.summary.scalar('accuracy', accuracy))
        # tf.summary.scalar('accuracy', accuracy)
        summaries.add(tf.summary.scalar('loss', total_loss))
        # tf.summary.scalar('loss', total_loss)
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        summary_op = tf.summary.merge_all()

        slim.learning.train(train_tensor,
                            train_log_dir,
                            number_of_steps=num_steps,
                            summary_op=summary_op,
                            train_step_fn=train_step_fn,
                            save_summaries_secs=20)
        print('completed training')

        coord.request_stop()
        coord.join(threads)


def run_resnet_v2_training(model_name,
                           data_directory,
                           path_to_train_file,
                           path_to_val_file,
                           path_to_labels_file,
                           bsize,
                           num_steps,
                           train_log_dir,
                           optimizer,
                           initial_checkpoint):

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with graph.as_default():
        name_dict, nclass = gen_dict(data_directory, path_to_labels_file)

        label, image = getImage(path_to_train_file, nclass)
        vlabel, vimage = getImage(path_to_val_file, nclass)

        imageBatch, labelBatch = tf.train.shuffle_batch(
            [image, label], batch_size=bsize,
            capacity=2000,
            min_after_dequeue=1000)

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

            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            if model_name == 'resnet_v2_50':
                with tf.variable_scope("resnet_v2_50") as scope:
                    logits, end_points = resnet_v2.resnet_v2_50(batch_xs,
                                                                num_classes=2,
                                                                is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = resnet_v2.resnet_v2_50(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=True)
            elif model_name == 'resnet_v2_101':
                with tf.variable_scope("resnet_v2_101") as scope:
                    logits, end_points = resnet_v2.resnet_v2_101(batch_xs,
                                                                 num_classes=2,
                                                                 is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = resnet_v2.resnet_v2_101(vbatch_xs,
                                                                   num_classes=2,
                                                                   is_training=True)
            elif model_name == 'resnet_v2_152':
                with tf.variable_scope("resnet_v2_152") as scope:
                    logits, end_points = resnet_v2.resnet_v2_152(batch_xs,
                                                                 num_classes=2,
                                                                 is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = resnet_v2.resnet_v2_152(vbatch_xs,
                                                                   num_classes=2,
                                                                   is_training=True)
            elif model_name == 'resnet_v2_200':
                with tf.variable_scope("resnet_v2_200") as scope:
                    logits, end_points = resnet_v2.resnet_v2_200(batch_xs,
                                                                 num_classes=2,
                                                                 is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = resnet_v2.resnet_v2_200(vbatch_xs,
                                                                   num_classes=2,
                                                                   is_training=True)

            predictions = end_points['predictions']
            predictions_validation = vend_points['predictions']  # -- for resnet model use predictions

            predictions = tf.squeeze(predictions)
            predictions_validation = tf.squeeze(predictions_validation)

            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(batch_ys, 1))
            vcorrect_prediction = tf.equal(tf.argmax(predictions_validation, 1), tf.argmax(vbatch_ys, 1))

            # get mean of all entries in correct prediction, the higher the better
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_validation = tf.reduce_mean(tf.cast(vcorrect_prediction, tf.float32))

            logits = tf.reshape(logits, [bsize, 2])

            tf.losses.softmax_cross_entropy(batch_ys, logits)

            total_loss = tf.losses.get_total_loss()

            train_tensor = slim.learning.create_train_op(total_loss, optimizer)

        def train_step_fn(sess, *args, **kwargs):
            total_loss, should_stop = train_step(sess, *args, **kwargs)
            accuracy = sess.run([train_step_fn.accuracy])
            if train_step_fn.step % 50 == 0:
                # sess.run(assignment)
                accuracy_validation = sess.run([train_step_fn.accuracy_validation])
                # print('Step %s - Loss: %.2f Validation Accuracy: %.2f%%' %
                #       (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))

            train_step_fn.step += 1
            return [total_loss, should_stop]

        train_step_fn.step = 0
        train_step_fn.accuracy = accuracy
        train_step_fn.accuracy_validation = accuracy_validation

        summaries.add(tf.summary.scalar('accuracy', accuracy))
        # tf.summary.scalar('accuracy', accuracy)
        summaries.add(tf.summary.scalar('loss', total_loss))
        # tf.summary.scalar('loss', total_loss)
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        slim.learning.train(train_tensor,
                            train_log_dir,
                            number_of_steps=num_steps,
                            summary_op=summary_op,
                            train_step_fn=train_step_fn,
                            save_summaries_secs=20)
        print('completed training')

        coord.request_stop()
        coord.join(threads)


def run_vgg_training(model_name,
                     data_directory,
                     path_to_train_file,
                     path_to_val_file,
                     path_to_labels_file,
                     bsize,
                     num_steps,
                     train_log_dir,
                     optimizer,
                     initial_checkpoint):

    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with graph.as_default():
        name_dict, nclass = gen_dict(data_directory, path_to_labels_file)

        label, image = getImage(path_to_train_file, nclass)
        vlabel, vimage = getImage(path_to_val_file, nclass)

        imageBatch, labelBatch = tf.train.shuffle_batch(
            [image, label], batch_size=bsize,
            capacity=2000,
            min_after_dequeue=1000)

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

            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            if model_name == 'vgg_16':
                with tf.variable_scope("vgg_16") as scope:
                    logits, end_points = vgg.vgg_16(batch_xs,
                                                                num_classes=2,
                                                                is_training=True)
                    scope.reuse_variables()
                    vlogits, vend_points = vgg.vgg_16(vbatch_xs,
                                                                  num_classes=2,
                                                                  is_training=True)
            elif model_name == 'vgg_19':
                with tf.variable_scope("vgg_19") as scope:
                    logits, end_points = vgg.vgg_19(batch_xs,
                                                    num_classes=2,
                                                    is_training=True)
                    scope.reuse_variables()

                    vlogits, vend_points = vgg.vgg_19(vbatch_xs,
                                                      num_classes=2,
                                                      is_training=True)

            # predictions = end_points['Predictions']
            # predictions_validation = vend_points['Predictions']  # -- for inception model use Predictions

            predictions = tf.nn.softmax(logits)
            predictions_validation = tf.nn.softmax(vlogits)

            correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(batch_ys, 1))
            vcorrect_prediction = tf.equal(tf.argmax(predictions_validation, 1), tf.argmax(vbatch_ys, 1))

            # get mean of all entries in correct prediction, the higher the better
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_validation = tf.reduce_mean(tf.cast(vcorrect_prediction, tf.float32))

            logits = tf.reshape(logits, [bsize, 2])

            tf.losses.softmax_cross_entropy(batch_ys, logits)

            total_loss = tf.losses.get_total_loss()

            train_tensor = slim.learning.create_train_op(total_loss, optimizer)

        def train_step_fn(sess, *args, **kwargs):
            total_loss, should_stop = train_step(sess, *args, **kwargs)
            accuracy = sess.run([train_step_fn.accuracy])
            if train_step_fn.step % 50 == 0:
                # sess.run(assignment)
                accuracy_validation = sess.run([train_step_fn.accuracy_validation])
                # print('Step %s - Loss: %.2f Validation Accuracy: %.2f%%' %
                #       (str(train_step_fn.step).rjust(6, '0'), total_loss, accuracy * 100))
                # saver.save(sess, os.path.join(train_log_dir, "model.ckpt"), train_step_fn.step)

            train_step_fn.step += 1
            return [total_loss, should_stop]

        train_step_fn.step = 0
        train_step_fn.accuracy = accuracy
        train_step_fn.accuracy_validation = accuracy_validation

        summaries.add(tf.summary.scalar('accuracy', accuracy))
        # tf.summary.scalar('accuracy', accuracy)
        summaries.add(tf.summary.scalar('loss', total_loss))
        # tf.summary.scalar('loss', total_loss)
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        slim.learning.train(train_tensor,
                            train_log_dir,
                            number_of_steps=num_steps,
                            summary_op=summary_op,
                            train_step_fn=train_step_fn,
                            save_summaries_secs=20)
        print('completed training')

        coord.request_stop()
        coord.join(threads)
