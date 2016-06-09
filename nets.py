import os
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


def get_epoch_finished(dir_name, patience=100):
    return [
        SaveWeights(dir_name + 'model_weights.pkl', only_best=True, pickle=False),
        SaveTrainingHistory(dir_name + 'model_history.pkl'),
        PlotTrainingHistory(dir_name + 'training_history.png'),
        EarlyStopping(patience=patience)
    ]


def get_back_pathway(forward_pathway):
    # We create the backwards path of the encoder from the forward path
    # We need to mirror the configuration of the layers and change the pooling operators with unpooling,
    # and the convolutions with deconvolutions (convolutions with diferent padding). This definitions
    # match the values of the possible_layers dictionary
    back_pathway = ''.join(['d' if l is 'c' else 'u' for l in forward_pathway[::-1]])
    last_conv = back_pathway.rfind('d')
    back_pathway = back_pathway[:last_conv] + 'f' + back_pathway[last_conv + 1:]

    return back_pathway


def get_layers_string(net_layers, input_shape, convo_size, pool_size, number_filters):
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
        'i': '(InputLayer, {'
             '\'name\': \'\033[34minput\033[0m\','
             '\'shape\': input_shape})',
        'c': '(Conv3DDNNLayer, {'
             '\'name\': \'\033[33mconv%d\033[0m\' % (c_index.inc()),'
             '\'num_filters\': number_filters,'
             '\'filter_size\': (convo_size, convo_size, convo_size),'
             '\'pad\': \'valid\'})',
        'a': '(Pool3DDNNLayer, {'
             '\'name\': \'\033[36mavg_pool%d\033[0m\' % (p_index.inc()),'
             '\'pool_size\': pool_size,'
             '\'mode\': \'average_inc_pad\'})',
        'm': '(MaxPool3DDNNLayer, {'
             '\'name\': \'\033[36mmax_pool%d\033[0m\' % (p_index.inc()),'
             '\'pool_size\': pool_size})',
        'u': '(Unpooling3D, {'
             '\'name\': \'\033[46munpool%d\033[0m\' % (p_index.dec()),'
             '\'pool_size\': pool_size})',
        'd': '(Conv3DDNNLayer, {'
             '\'name\': \'\033[43mdeconv%d\033[0m\' % (c_index.dec()),'
             '\'num_filters\': number_filters,'
             '\'filter_size\': (convo_size, convo_size, convo_size),'
             '\'pad\': \'full\'})',
        'f': '(Conv3DDNNLayer, {'
             '\'name\': \'\033[43mfinal\033[0m\','
             '\'num_filters\': input_shape[1],'
             '\'filter_size\': (convo_size, convo_size, convo_size),'
             '\'pad\': \'full\'})',
        'r': '(ReshapeLayer, {'
             '\'name\': \'\033[44mreshape\033[0m\','
             ' \'shape\': ([0], -1)})',
        's': '(DenseLayer, {'
             '\'name\':\'\033[44mout\033[0m\','
             ' \'num_units\': reduce(mul, input_shape[2:], 1),'
             ' \'nonlinearity\': nonlinearities.softmax})'
    }

    return [eval(possible_layers[l]) for l in net_layers]


def create_encoder3D_string(forward_path, input_shape, convo_size, pool_size, number_filters, dir_name):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'i' + forward_path + get_back_pathway(forward_path) + 'r'

    encoder = NeuralNet(
        layers=get_layers_string(final_layers, input_shape, convo_size, pool_size, number_filters),

        regression=True,

        update=updates.adadelta,
        # update=updates.adam,
        # update_learning_rate=1e-3,

        on_epoch_finished=get_epoch_finished(os.path.join(dir_name, 'encoder')),

        verbose=11,
        max_epochs=200
    )

    return encoder


def create_unet3D_string(forward_path, input_shape, convo_size, pool_size, number_filters, dir_name):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'i' + forward_path + get_back_pathway(forward_path) + 'r' + 's'

    encoder = NeuralNet(
        layers=get_layers_string(final_layers, input_shape, convo_size, pool_size, number_filters),

        regression=True,

        # update=updates.adadelta,
        update=updates.adam,
        update_learning_rate=1e-3,

        on_epoch_finished=get_epoch_finished(os.path.join(dir_name, 'unet')),

        verbose=11,
        max_epochs=200
    )

    return encoder
