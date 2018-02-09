import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    node_names = ['image_input:0', 'keep_prob:0', 'layer3_out:0', 'layer4_out:0','layer7_out:0']
    nodes = [graph.get_tensor_by_name(name) for name in node_names]

    return nodes
tests.test_load_vgg(load_vgg, tf)

def conv_1x1(layer, num_classes, name=None):
    output = tf.layers.conv2d(layer,
                              num_classes,
                              kernel_size=1,
                              padding='same',
                              name=name,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return output

def upsample(layer, num_classes, kernel_size, stride, name=None):
    output = tf.layers.conv2d_transpose(layer,
                                        num_classes,
                                        kernel_size=kernel_size,
                                        strides=stride,
                                        padding='same',
                                        name=name,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    return output


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    with tf.variable_scope("fcn"):
        # replace fully connected output of layer7 with 1x1 convolution
        layer7_conv_1x1 = conv_1x1(vgg_layer7_out, num_classes, 'layer7_conv_1x1')

        # scale output by 2 (kernel size=4, stride=2)
        layer7_out_2 = upsample(layer7_conv_1x1, num_classes, 4, (2, 2), 'layer7_out_2')

        # layer4 output 1x1 convolution
        layer4_conv_1x1 = conv_1x1(vgg_layer4_out, num_classes, 'layer4_conv_1x1')

        tf.add(layer7_out_2, layer4_conv_1x1)

        # upscale layer7_out_2 by 2 (kernel size=4, stride=2)
        layer7_out_4 = upsample(layer7_out_2, num_classes, 4, (2, 2), 'layer7_out_4')

        layer3_conv_1x1 = conv_1x1(vgg_layer3_out, num_classes, 'layer3_conv_1x1')

        tf.add(layer7_out_4, layer3_conv_1x1)

        # upscale layer7_out_4 by 2 (kernel size=16, stride=8)
        layer7_out_8 = upsample(layer7_out_4, num_classes, 16, (8, 8), 'layer7_out_8')

    return layer7_out_8
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    training_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)


def augment_batch(batch):
    return np.array([image_random_brightness(image) for image in batch], dtype=np.uint8)

def image_random_brightness(image, chance=.5):
    if np.random.random() > chance:
        return image
    image = np.array(image, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image = np.clip(image*random_bright, 0, 255)
    image = np.array(image, dtype=np.uint8)
    return image

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    if save_model:
        saver = tf.train.Saver()
        model_filename = './model.ckpt'
        best_loss = None

    print('Training...')

    for epoch in range(epochs):
        n_data = 0
        loss = 0
        for batch_images, batch_labels in get_batches_fn(batch_size):
            # batch_images = augment_batch(batch_images)
            _, batch_loss = sess.run([train_op, cross_entropy_loss],
                                      feed_dict={input_image : batch_images,
                                                 correct_label: batch_labels,
                                                 keep_prob: 0.5,
                                                 learning_rate: 0.0005})
            n_batch = batch_images.shape[0]
            loss += (batch_loss * n_batch)
            n_data += n_batch

        loss /= n_data

        print('epoch: {} - loss:{:4f}'.format(epoch, loss))

        if save_model and (best_loss is None or loss < best_loss):
            saver.save(sess, model_filename)
            best_loss = loss

save_model = False
tests.test_train_nn(train_nn)
save_model = True

def run():
    epochs = 50
    batch_size = 12
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes], name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits, training_operation, cross_entropy_loss = optimize(layer_output,
                                                                  correct_label,
                                                                  learning_rate,
                                                                  num_classes)

        sess.run(tf.global_variables_initializer())

        train_nn(sess, epochs, batch_size, get_batches_fn, training_operation,
                 cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
