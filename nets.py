from operator import mul
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import (
    SaveTrainingHistory, PlotTrainingHistory,
    EarlyStopping
)
from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer, Pool3DDNNLayer
from layers import Unpooling3D
from lasagne import updates
from lasagne import nonlinearities


def create_encoder_string(forward_layers, input_shape, convo_size, pool_size, dir_name, number_filters):
    # Index used to numerate the layers
    # While defining this object is not necessary, it helps encapsulate
    # the increment and decrement of the indices corresponding to the layers.
    # Since this object will only be used here, we decided to limit its scope to this function.
    class Index:
        def __init__(self):
            self.i = 1

        def inc(self):
            self.i += 1
            return self.i - 1

        def dec(self):
            self.i -= 1
            return self.i

    c_index = Index()
    p_index = Index()

    # These are all the possible layers for our autoencoders
    possible_layers = {
        'i': '(InputLayer, {\'name\': \'input\',\'shape\': input_shape})',
        'c': '(Conv3DDNNLayer, {'
             '\'name\': \'conv%d\' % (c_index.inc()),'
             '\'num_filters\': number_filters,'
             '\'filter_size\': (convo_size, convo_size, convo_size),'
             '\'pad\': \'valid\'})',
        'a': '(Pool3DDNNLayer, {'
             '\'name\': \'pool%d\' % (p_index.inc()),'
             '\'pool_size\': pool_size,'
             '\'mode\': \'average_inc_pad\'})',
        'm': '(MaxPool3DDNNLayer, {'
             '\'name\': \'pool%d\' % (p_index.inc()),'
             '\'pool_size\': pool_size})',
        'u': '(Unpooling3D, {'
             '\'name\': \'unpool%d\' % (p_index.dec()),'
             '\'pool_size\': pool_size})',
        'd': '(Conv3DDNNLayer, {'
             '\'name\': \'deconv%d\' % (c_index.dec()),'
             '\'num_filters\': number_filters,'
             '\'filter_size\': (convo_size, convo_size, convo_size),'
             '\'pad\': \'full\'})',
        'f': '(Conv3DDNNLayer, {'
             '\'name\': \'final\','
             '\'num_filters\': input_shape[1],'
             '\'filter_size\': (convo_size, convo_size, convo_size),'
             '\'pad\': \'full\'})',
        'r': '(ReshapeLayer, {\'name\': \'reshape\', \'shape\': ([0], -1)})'
    }

    # We create the backwards patt of the encoder from the forward path
    # We need to mirror the configuration of the layers and change the pooling operators with unpooling,
    # and the convolutions with deconvolutions (convolutions with diferent padding). This definitions
    # match the values of the possible_layers dictionary
    back_layers = ''.join(['d' if l is 'c' else 'u' for l in forward_layers[::-1]])
    last_conv = back_layers.rfind('d')
    back_layers = back_layers[:last_conv] + 'f' + back_layers[last_conv+1:]
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'i' + forward_layers + back_layers + 'r'

    save_weights = SaveWeights(dir_name + 'model_weights.pkl', only_best=True, pickle=False)
    save_training_history = SaveTrainingHistory(dir_name + 'model_history.pkl')
    plot_training_history = PlotTrainingHistory(dir_name + 'training_history.png')
    early_stopping = EarlyStopping(patience=100)

    encoder = NeuralNet(
        layers=[eval(possible_layers[l]) for l in final_layers],

        regression=True,

        update=updates.adadelta,
        # update=updates.adam,
        # update_learning_rate=1e-3,

        on_epoch_finished=[
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping
        ],

        verbose=11,
        max_epochs=200
    )

    return encoder


def create_encoder(input_shape, convo_size, pool_size, dir_name, number_filters):

    save_weights = SaveWeights(dir_name + 'model_weights.pkl', only_best=True, pickle=False)
    save_training_history = SaveTrainingHistory(dir_name + 'model_history.pkl')
    plot_training_history = PlotTrainingHistory(dir_name + 'training_history.png')
    early_stopping = EarlyStopping(patience=100)

    encoder = NeuralNet(
        layers=[
            (InputLayer, {'name': 'input', 'shape': input_shape}),
            # (MaxPool3DDNNLayer, {'name': 'downsample', 'pool_size': pool_size}),
            (Conv3DDNNLayer, {
                'name': 'conv1',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'valid'
            }),
            (Conv3DDNNLayer, {
                'name': 'conv2',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'valid'
            }),
            (Pool3DDNNLayer, {
                'name': 'pool1',
                'pool_size': pool_size,
                'mode': 'average_inc_pad'}),
            (Conv3DDNNLayer, {
                'name': 'conv3',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'valid'
            }),


            (Conv3DDNNLayer, {
                'name': 'deconv3',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'full'
            }),
            (Unpooling3D, {'name': 'unpool1', 'pool_size': pool_size}),
            (Conv3DDNNLayer, {
                'name': 'deconv2',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'full'
            }),
            (Conv3DDNNLayer, {
                'name': 'deconv1',
                'num_filters': input_shape[1],
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'full'
            }),
            # (Unpooling3D, {'name': 'upsample', 'pool_size': pool_size}),
            (ReshapeLayer, {'name': 'resampling', 'shape': ([0], -1)})

        ],

        regression=True,

        update=updates.adadelta,
        # update=updates.adam,
        # update_learning_rate=1e-3,

        on_epoch_finished=[
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping
        ],

        verbose=11,
        max_epochs=200
    )

    return encoder


def create_unet(input_shape, convo_size, pool_size, dir_name, number_filters):

    save_weights = SaveWeights(dir_name + 'model_weights.pkl', only_best=True, pickle=False)
    save_training_history = SaveTrainingHistory(dir_name + 'model_history.pkl')
    plot_training_history = PlotTrainingHistory(dir_name + 'training_history.png')
    early_stopping = EarlyStopping(patience=100)

    unet = NeuralNet(
        layers=[
            (InputLayer, {'name': 'input', 'shape': input_shape}),
            (Conv3DDNNLayer, {
                'name': 'conv1',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'valid'
            }),
            (Conv3DDNNLayer, {
                'name': 'conv2',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'valid'
            }),
            (MaxPool3DDNNLayer, {'name': 'pool', 'pool_size': pool_size}),

            (Conv3DDNNLayer, {
                'name': 'conv3',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'valid'
            }),
            (Conv3DDNNLayer, {
                'name': 'deconv3',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'full'
            }),

            (Unpooling3D, {'name': 'unpool', 'pool_size': pool_size}),
            (Conv3DDNNLayer, {
                'name': 'deconv2',
                'num_filters': number_filters,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'full'
            }),
            (Conv3DDNNLayer, {
                'name': 'deconv1',
                'num_filters': 1,
                'filter_size': (convo_size, convo_size, convo_size),
                'pad': 'full'
            }),
            (ReshapeLayer, {'name': 'resampling', 'shape': ([0], -1)}),

            (DenseLayer, {'name':'out', 'num_units': reduce(mul, input_shape[2:], 1), 'nonlinearity': nonlinearities.softmax}),

        ],

        regression=False,

        update=updates.adadelta,

        on_epoch_finished=[
            save_weights,
            save_training_history,
            plot_training_history,
            early_stopping
        ],

        verbose=11,
        max_epochs=200
    )

    return unet
