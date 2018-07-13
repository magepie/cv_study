import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from nets import nets_factory
from preprocessing import preprocessing_factory
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
import os
import time
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import sys

slim = tf.contrib.slim

# two dataset directories
dataset_dir_rgb = 'tmp/dataset_parallel_trf/rgb'
dataset_dir_depth = 'tmp/dataset_parallel_trf/depth'
# correspondingly two labels files
labels_file_rgb = 'tmp/dataset_parallel_trf/rgb/labels.txt'
labels_file_depth = 'tmp/dataset_parallel_trf/depth/labels.txt'


log_dir = 'tmp/log/both_no_cp'
checkpoint_file_rgb = 'tmp/log/rgb_no_cp/model'
checkpoint_file_depth = 'tmp/log/depth_no_cp/model'
image_size = 256
num_classes = 5

labels = open(labels_file_rgb, 'r')
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1]
    labels_to_name[int(label)] = string_name

file_pattern = 'objects_%s_*.tfrecord'

items_to_descriptions = {
    'image': 'A 3-channel RGB coloured product image',
    'label': 'A label that from 4 labels'
}

num_epochs = 3
batch_size = 4
initial_learning_rate = 0.001
learning_rate_decay_factor = 0.75
num_epochs_before_decay = 1

def variable_summaries(var, scope):
  with tf.name_scope(scope):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def get_split(split_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='objects'):
    if split_name not in ['train', 'validation']:
        raise ValueError(
            'The split_name %s is not recognized. Please input either train or validation as the split_name' % (
            split_name))

    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    num_samples = 0
    file_pattern_for_counting = 'objects' + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if
                          file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    test = num_samples

    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_name_dict = labels_to_name

    dataset = slim.dataset.Dataset(
        data_sources=file_pattern_path,
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        num_classes=num_classes,
        labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset

def load_batch(dataset, batch_size, is_training=True):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    # First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=24 + 3 * batch_size,
        common_queue_min=24,
        shuffle=False)

    # # Obtain the raw image using the get method
    image, label = data_provider.get(['image', 'label'])

    # # Perform the correct preprocessing for this image depending if it is training or evaluating
    image_preprocessing_fn = preprocessing_factory.get_preprocessing('inception_v3',is_training=True)

    train_image_size = 256
    image = image_preprocessing_fn(image, train_image_size, train_image_size)

    # # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=5 * batch_size)

    return images, labels
    
def run():
    end_points = {}
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level
        
        ########################################################
        # Get RGB dataset and the Imagenet trained on RGB images
        ########################################################
        
        # First create the dataset and load one batch
        dataset_rgb = get_split('train', dataset_dir_rgb, file_pattern=file_pattern)
        images_rgb, labels_rgb = load_batch(dataset_rgb, batch_size=batch_size)

        num_batches_per_epoch = int(dataset_rgb.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        with tf.variable_scope("net_rgb"):
            # Create the model inference
            with slim.arg_scope(inception_v3_arg_scope()):
                logits_rgb, end_points_rgb = inception_v3(images_rgb, num_classes=dataset_rgb.num_classes, is_training=True)

        net1_varlist = {v.name.lstrip("net_rgb/")[:-2]: v
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net_rgb/")}
        
        ########################################################
        # Get depth dataset and the Imagenet trained on depth images
        ########################################################
        
        # First create the dataset and load one batch
        dataset_depth = get_split('train', dataset_dir_depth, file_pattern=file_pattern)
        images_depth, labels_depth = load_batch(dataset_depth, batch_size=batch_size)

        num_batches_per_epoch = int(dataset_rgb.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        # Create the model inference
        with tf.variable_scope("net_depth"):
            with slim.arg_scope(inception_v3_arg_scope()):
                logits_depth, end_points_depth = inception_v3(images_depth, num_classes=dataset_rgb.num_classes, is_training=True)

        net2_varlist = {v.name.lstrip("net_depth/")[:-2]: v
                for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="net_depth/")}        
        ########################################################
        # Combine the models with the concatenation operation
        # and add an FC layer on top
        ########################################################
        
        #       
        with tf.variable_scope("concat_dense"):        
            W_master = tf.Variable(tf.random_uniform([10, 5], -0.01, 0.01), name = "weights_concat")
            b_master = tf.Variable(tf.zeros([5]), name = "bias_concat")
            
            h_master = tf.matmul(tf.concat((logits_rgb, logits_depth), axis=1), W_master) + b_master
            
            variable_summaries(W_master, "concat")
            
            logits2 = tf.layers.dense(inputs=h_master, units=(num_classes * 2), name="dense_concat1")
            
            logits = tf.layers.dense(inputs=logits2, units=num_classes, name="dense_concat0")
        
        end_points['Logits'] = logits
        end_points['Predictions'] = slim.softmax(logits, scope='Predictions')
        

        one_hot_labels = slim.one_hot_encoding(labels_rgb, dataset_rgb.num_classes)


        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        tf.summary.scalar('cross_entropy', loss)
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)


        global_step = get_or_create_global_step()
        

        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "concat_dense")
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train = train_vars)

        # Metrics
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels_rgb)
        metrics_op = tf.group(accuracy_update, probabilities)

        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()
        

        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        saver_rgb = tf.train.Saver(var_list=net1_varlist)
        saver_depth = tf.train.Saver(var_list=net2_varlist)

        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, init_fn=None)

        with sv.managed_session() as sess:
            print (num_steps_per_epoch)
            print (num_epochs)
            rgb_latest = tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'tmp/log/rgb_no_cp'))
            depth_latest = tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'tmp/log/depth_no_cp'))
            
            saver_rgb.restore(sess, rgb_latest)
            saver_depth.restore(sess, depth_latest)

            
            for step in range(num_steps_per_epoch * num_epochs):
               
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)

                if step % 10 == 0:
                    train_writer = tf.summary.FileWriter('tmp/vis/both_no_cp', sess.graph)
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    train_writer.add_summary(summaries, step)
                    train_writer.close()
                # If not, simply run the training step
                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    

            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
            
if __name__ == '__main__':
    run()