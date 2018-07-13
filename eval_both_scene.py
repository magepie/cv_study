import tensorflow as tf
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from preprocessing import preprocessing_factory
from sklearn.metrics import confusion_matrix
import os
import time

slim = tf.contrib.slim

log_dir = 'tmp/log/both_with_cp'
log_eval = 'tmp/log_eval_scenes/sc2/both'

dataset_dir_rgb = 'tmp/scenes_trf/sc0041/rgb'
dataset_dir_depth = 'tmp/scenes_trf/sc0041/depth'
# correspondingly two labels files
labels_file_rgb = 'tmp/scenes_trf/sc0041/rgb/labels.txt'
labels_file_depth = 'tmp/scenes_trf/sc0041/depth/labels.txt'

batch_size = 10

num_epochs = 1

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

checkpoint_file = tf.train.latest_checkpoint(log_dir)

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
    image_preprocessing_fn = preprocessing_factory.get_preprocessing('inception_v3',is_training=False)

    train_image_size = 256
    image = image_preprocessing_fn(image, train_image_size, train_image_size)

    # # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size)

    return images, labels


def run():
    end_points = {}
    if not os.path.exists(log_eval):
        os.mkdir(log_eval)
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level
        
        ########################################################
        # Get RGB dataset and the Imagenet trained on RGB images
        ########################################################
        
        # First create the dataset and load one batch
        dataset_rgb = get_split('train', dataset_dir_rgb, file_pattern=file_pattern)
        images_rgb, labels_rgb = load_batch(dataset_rgb, batch_size=batch_size)

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset_rgb.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed

        with tf.variable_scope("net_rgb"):
            # Create the model inference
            with slim.arg_scope(inception_v3_arg_scope()):
                logits_rgb, end_points_rgb = inception_v3(images_rgb, num_classes=dataset_rgb.num_classes, is_training=True)
        
        ########################################################
        # Get depth dataset and the Imagenet trained on depth images
        ########################################################
        
        # First create the dataset and load one batch
        dataset_depth = get_split('train', dataset_dir_depth, file_pattern=file_pattern)
        images_depth, labels_depth = load_batch(dataset_depth, batch_size=batch_size)

        # Create the model inference
        with tf.variable_scope("net_depth"):
            with slim.arg_scope(inception_v3_arg_scope()):
                logits_depth, end_points_depth = inception_v3(images_depth, num_classes=dataset_rgb.num_classes, is_training=True)

        ########################################################
        # Combine the models with the concatenation operation
        # and add an FC layer on top
        ########################################################
        
        #    
        with tf.variable_scope("concat_dense"):           
            W_master = tf.Variable(tf.random_uniform([10, 5], -0.01, 0.01), name = "weights_concat")
            b_master = tf.Variable(tf.zeros([5]), name = "bias_concat")
            
            h_master = tf.matmul(tf.concat((logits_rgb, logits_depth), axis=1), W_master) + b_master
            
            logits2 = tf.layers.dense(inputs=h_master, units=(num_classes * 2), name="dense_concat1")
            
            logits = tf.layers.dense(inputs=logits2, units=num_classes, name="dense_concat0")
        
        end_points['Logits'] = logits
        end_points['Predictions'] = slim.softmax(logits, scope='Predictions')
        
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)
            
        ####################################################
        # EVALUATION
        ####################################################

        predictions = tf.argmax(end_points['Predictions'], 1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels_rgb)
        metrics_op = tf.group(accuracy_update)

        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1)
        
        counter = 0
        counter_delta = 0
        delta = 0.01
        selected_item_idx = 1

        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value = sess.run([metrics_op, global_step_op, accuracy])
            time_elapsed = time.time() - start_time
            
            pred = sess.run(predictions)
            
            #sess.run(print_op)
            
            values = sess.run(end_points['Predictions'])
            #logging.info('Probabilty values: ', values)
            
            #delta = 0.1
            nonlocal counter
            nonlocal counter_delta
            for i in range(10):
                #if (selected_item_idx ==  pred[i]):
                if (selected_item_idx ==  pred[i]):
                    counter = counter + 1
                elif ((values[i][pred[i]] - values[i][selected_item_idx]) <= delta):
                    counter_delta = counter_delta + 1

            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value,
                         time_elapsed)

            return accuracy_value

        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(logdir=log_eval, summary_op=None, saver=None, init_fn=restore_fn)
        
        selected_item_idx = 0
        
        with sv.managed_session() as sess:
            counter = 0
            num_steps_per_epoch = int(num_steps_per_epoch)
            for step in range(num_steps_per_epoch * num_epochs):
                sess.run(sv.global_step)
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))

                if step % 10 == 0:
                    eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)


                else:
                    eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)

            #logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))
            logging.info('Final counter: %d', counter)
            logging.info('Final delta counter: %d', counter_delta)
            
            images_rgb, images_depth, labels, predictions = sess.run([images_rgb, images_depth, labels_rgb, predictions])
           
            
            # for i in range(10):
                # label, prediction = labels[i], predictions[i]
                # prediction_name, label_name = dataset_rgb.labels_to_name[prediction], dataset_rgb.labels_to_name[label]
                # text = 'Prediction: %s \n Ground Truth: %s' % (prediction_name, label_name)
                # print(text)
            # logging.info(
                # 'Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
                
if __name__ == '__main__':
    run()