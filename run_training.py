import tensorflow as tf
from model_functions import *


def run_model_training(model_name,
                       bsize,
                       train_log_dir,
                       data_directory,
                       initial_checkpoint,
                       num_steps,
                       optimizer):

    path_to_train_file = data_directory + '/train-00000-of-00001'
    path_to_val_file = data_directory + '/validation-00000-of-00001'
    path_to_labels_file = data_directory + '/labels.txt'

    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MakeDirs(train_log_dir)

    if 'inception' in model_name:
        if initial_checkpoint == '':
            print('Running ' + model_name + ' model using no pre-existing checkpoint')
        else:
            print('Running ' + model_name + ' model using checkpoint at ' + initial_checkpoint)
        run_inception_training(model_name,
                               data_directory,
                               path_to_train_file,
                               path_to_val_file,
                               path_to_labels_file,
                               bsize,
                               num_steps,
                               train_log_dir,
                               optimizer,
                               initial_checkpoint)
    if 'lenet' in model_name:
        if initial_checkpoint == '':
            print('Running ' + model_name + ' model using no pre-existing checkpoint')
        else:
            print('Running ' + model_name + ' model using checkpoint at ' + initial_checkpoint)
        run_lenet_training(data_directory,
                           path_to_train_file,
                           path_to_val_file,
                           path_to_labels_file,
                           bsize,
                           num_steps,
                           train_log_dir,
                           optimizer)

    if 'resnet_v1' in model_name:
        if initial_checkpoint == '':
            print('Running ' + model_name + ' model using no pre-existing checkpoint')
        else:
            print('Running ' + model_name + ' model using checkpoint at ' + initial_checkpoint)
        run_resnet_v1_training(model_name,
                               data_directory,
                               path_to_train_file,
                               path_to_val_file,
                               path_to_labels_file,
                               bsize,
                               num_steps,
                               train_log_dir,
                               optimizer,
                               initial_checkpoint)

    if 'resnet_v2' in model_name:
        if initial_checkpoint == '':
            print('Running ' + model_name + ' model using no pre-existing checkpoint')
        else:
            print('Running ' + model_name + ' model using checkpoint at ' + initial_checkpoint)
        run_resnet_v2_training(model_name,
                               data_directory,
                               path_to_train_file,
                               path_to_val_file,
                               path_to_labels_file,
                               bsize,
                               num_steps,
                               train_log_dir,
                               optimizer,
                               initial_checkpoint)

    if 'vgg' in model_name:
        if initial_checkpoint == '':
            print('Running ' + model_name + ' model using no pre-existing checkpoint')
        else:
            print('Running ' + model_name + ' model using checkpoint at ' + initial_checkpoint)
        run_vgg_training(model_name,
                         data_directory,
                         path_to_train_file,
                         path_to_val_file,
                         path_to_labels_file,
                         bsize,
                         num_steps,
                         train_log_dir,
                         optimizer,
                         initial_checkpoint)
