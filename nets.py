from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import (
    SaveTrainingHistory, PlotTrainingHistory,
    EarlyStopping
)
from lasagne.layers import InputLayer, ReshapeLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from layers import Unpooling3D
from lasagne import updates

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