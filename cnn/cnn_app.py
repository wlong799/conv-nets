# coding=utf-8
"""Obtains configuration and runs the model in the appropriate mode."""
import argparse
import tensorflow as tf
import cnn


def _data_type(string):
    if string == 'float16':
        return tf.float16
    elif string == 'float32':
        return tf.float32
    elif string == 'float64':
        return tf.float64
    else:
        raise ValueError("Invalid data type specified.")


def _get_config():
    parser = argparse.ArgumentParser(
        description='Train or evaluate a convolutional neural network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(
        title='phase', dest='phase', help='which phase to run model in')
    train_parser = subparsers.add_parser('train', help='train model')
    valid_parser = subparsers.add_parser(
        'valid', help='run model validation from recent checkpoint')
    test_parser = subparsers.add_parser(
        'test', help='test model from recent checkpoint')
    subparsers.required = True

    parser.add_argument(
        'examples_per_epoch', type=int,
        help='total number of examples in subset of data being used')
    parser.add_argument(
        '--image_height', type=int, help='height of processed image')
    parser.add_argument(
        '--image_width', type=int, help='width of processed image')
    parser.add_argument(
        'num_classes', type=int,
        help='total number of distinct classes in data')

    file_locations = parser.add_argument_group(title='file locations')
    file_locations.add_argument(
        '--data_dir', default='data/',
        help='directory where datasets are stored')
    file_locations.add_argument(
        '--checkpoints_dir', default='checkpoints/',
        help='directory where Tensorflow graph checkpoints are stored')
    file_locations.add_argument(
        '--summaries_dir', default='summaries/',
        help='directory where TensorBoard summary logs are saved')

    model_params = parser.add_argument_group(title='model parameters')
    model_params.add_argument(
        '--model_type', default='simple', choices=['simple'],
        help='which model architecture to use')
    model_params.add_argument(
        '--padding', default='SAME', choices=['SAME', 'VALID'],
        help='which padding method to use for model')

    data_processing = parser.add_argument_group(title='data processing')
    data_processing.add_argument(
        '--batch_size', default=128, type=int,
        help='number of examples per batch')
    data_processing.add_argument(
        '--toggle_distortions', action='store_false', dest='distort_images',
        help='distort images during processing')
    data_processing.add_argument(
        '--num_threads', default=32, type=int,
        help='number of threads to use for preprocessing')
    data_processing.add_argument(
        '--min_example_fraction', default=0.5, type=float,
        help='min fraction of examples in queue during processing')

    data_rep = parser.add_argument_group(title='data representation')
    data_rep.add_argument(
        '--data_format', default='NHWC', choices=['NHWC', 'NCHW'],
        help='which image batch data format to use')
    data_rep.add_argument('--data_type', default='float32', type=_data_type,
                          choices=['float16', 'float32', 'float64'],
                          help='which floating type to store data as')

    train_params = train_parser.add_argument_group(title='training parameters')
    train_params.add_argument(
        '--learning_rate', default=0.1, type=float,
        help='learning rate of model for training')

    log_params = train_parser.add_argument_group(title='logging')
    log_params.add_argument(
        '--print_log_steps', default=10, type=int,
        help='how often (in steps) log information should be printed')
    log_params.add_argument(
        '--save_checkpoint_secs', default=600, type=int,
        help='how often (in seconds) checkpoint of model should be saved')
    log_params.add_argument(
        '--save_summaries_steps', default=100, type=int,
        help='how often (in steps) a TensorBoard summary should be created')

    return parser.parse_args()


def run():
    """Parses arguments to configure model and runs the appropriate step."""
    config = _get_config()
    if config.phase == 'train':
        cnn.train.train(
            config.model_type, config.image_height, config.image_width,
            config.num_classes, config.data_dir, config.checkpoints_dir,
            config.summaries_dir, config.batch_size, config.distort_images,
            config.num_threads,
            config.min_example_fraction * config.examples_per_epoch,
            config.learning_rate, config.print_log_steps,
            config.save_checkpoint_secs, config.save_summaries_steps)
    else:
        cnn.eval.evaluate(
            config.model_type, config.image_height, config.image_width,
            config.num_classes, config.data_dir, config.checkpoints_dir,
            config.examples_per_epoch, config.data_format, config.batch_size,
            config.num_threads)
