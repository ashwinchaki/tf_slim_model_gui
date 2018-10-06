import tensorflow as tf
import sys
import numpy as np
import matplotlib
import random

def gen_dict(data_dir, labels_file):
    # type: (object, object, object) -> object
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(
        labels_file, 'r').readlines()]

    dict = {}
    nlabels = len(unique_labels)

    for i, label in enumerate(unique_labels):
        x = np.zeros(nlabels)
        x[i] = 1
        dict[tuple(x)] = label

    return dict, nlabels


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix(images, height, width):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(images, (-1, height, width))


def invert_grayscale(images):
    """ Makes black white, and white black """
    return 1 - images


# Function to tell TensorFlow how to read a single image from input file
def getImage(filename, nClass):
    # convert filenames to a queue for an input pipeline.
    filenameQ = tf.train.string_input_producer([filename], num_epochs=None)

    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example
    key, fullExample = recordReader.read(filenameQ)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })

    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    image_buffer = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg', [image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # cast image into a single array, where each element corresponds to the greyscale
    # value of a single pixel.
    # the "1-.." part inverts the image, so that the background is black.

    image = tf.reshape(1 - tf.image.rgb_to_grayscale(image), [height * width])
    image = tf.reshape(image, [height,width,1])
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)

    # re-define label as a "one-hot" vector
    # it will be [0,1] or [1,0] here.
    # This approach can easily be extended to more classes.
    label = tf.stack(tf.one_hot(label - 1, nClass))

    return label, image

def get_current_optimizer(name,learning_rate):
    if name == 'GradientDescent':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif name == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif name == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    elif name == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif name == 'Ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    elif name == 'ProximalGradientDescent':
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate)
    elif name == 'ProximalAdagrad':
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate)
    elif name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    return optimizer
