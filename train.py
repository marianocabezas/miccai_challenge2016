from optparse import OptionParser
import matplotlib
import numpy as np
from lasagne.layers import InputLayer, ReshapeLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from layers import Unpooling3D
from lasagne import updates
from data_creation import load_encoder_data
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import (
    SaveTrainingHistory, PlotTrainingHistory,
    EarlyStopping
)
matplotlib.use('Agg')


def create_encoder(input_shape, convo_size, pool_size):

    save_weights = SaveWeights('./model_weights.pkl', only_best=True, pickle=False)
    save_training_history = SaveTrainingHistory('./model_history.pkl')
    plot_training_history = PlotTrainingHistory('./training_history.png')
    early_stopping = EarlyStopping(patience=100)

    encoder = NeuralNet(
        layers=[
            (InputLayer, {'name': 'input', 'shape': input_shape}),
            (MaxPool3DDNNLayer, {'name': 'downsample', 'pool_size': pool_size}),
            (Conv3DDNNLayer, {'name': 'conv1', 'num_filters': 32, 'filter_size': (convo_size, convo_size, convo_size), 'pad': 'valid'}),
            (MaxPool3DDNNLayer, {'name': 'pool', 'pool_size': pool_size}),

            (Conv3DDNNLayer, {'name': 'conv2', 'num_filters': 32, 'filter_size': (convo_size, convo_size, convo_size), 'pad': 'valid'}),
            (Conv3DDNNLayer, {'name': 'deconv2', 'num_filters': 32, 'filter_size': (convo_size, convo_size, convo_size), 'pad': 'full'}),

            (Unpooling3D, {'name': 'unpool', 'pool_size': pool_size}),
            (Conv3DDNNLayer, {'name': 'deconv1', 'num_filters': 5, 'filter_size': (convo_size, convo_size, convo_size), 'pad': 'full'}),
            (Unpooling3D, {'name': 'upsample', 'pool_size': pool_size}),
            (ReshapeLayer, {'name': 'resampling', 'shape': (([0], -1))})

        ],

        regression=True,

        update=updates.adadelta,

        on_epoch_finished=[
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping
        ],

        verbose=11,
        max_epochs=100
    )

    return encoder


if __name__ == '__main__':
    # parse command line options
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option('-f', '--folder', dest='folder',
                      default='/home/mariano/DATA/Challenge/',
                      help="read data from FOLDER")
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose', default=False)
    parser.add_option('-c', '--convolution-size',
                      action='store', dest='convo_size', type='int', nargs=1, default=15)
    parser.add_option('-p', '--pool-size',
                      action='store', dest='pool_size', type='int', nargs=1, default=2)
    parser.add_option('-t', '--test-size',
                      action='store', dest='test_size', type='float', nargs=1, default=2)
    (options, args) = parser.parse_args()

    X_train, X_test, y_train, y_test = load_encoder_data(test_size=options.test_size, dir_name=options.folder)
    np.save(options.folder + 'test_encoder.npy', X_test)
    net = create_encoder(X_train.shape, options.convo_size, options.pool_size)
    net.fit(X_train, y_train.astype(np.float32))

    # Load the best weights from pickled model
    net.load_params_from('./model_weights.pkl')

    score = net.score(X_test, y_test)
    print 'Final score %.4f' % score