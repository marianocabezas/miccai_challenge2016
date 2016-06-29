import os
from theano import tensor
from operator import mul
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import (
    SaveTrainingHistory, PlotTrainingHistory,
    EarlyStopping
)
from lasagne import objectives
from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, DropoutLayer, ElemwiseSumLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer, Pool3DDNNLayer
from layers import Unpooling3D
from lasagne import updates
from lasagne import nonlinearities


def probabilistic_dsc_objective(predictions, targets):
    top = 2.0 * tensor.sum(predictions * targets, axis=1)
    bottom = tensor.sum(predictions, axis=1) + tensor.sum(targets, axis=1)
    return 1.0 - (top / bottom)


def get_epoch_finished(dir_name, patience):
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
    previous_layer = InputLayer(name='\033[30minput\033[0m', shape=input_shape)
    convolutions = dict()
    c_index = 1
    p_index = 1
    c_size = (convo_size, convo_size, convo_size)
    for layer in net_layers[1:]:
        if layer == 'c':
            conv_layer = Conv3DDNNLayer(
                incoming=previous_layer,
                name='\033[34mconv%d\033[0m' % c_index,
                num_filters=number_filters,
                filter_size=c_size,
                pad='valid')
            convolutions['conv%d' % c_index] = conv_layer
            previous_layer = conv_layer
            c_index += 1
        elif layer == 'a':
            previous_layer = Pool3DDNNLayer(
                incoming=previous_layer,
                name='\033[31mavg_pool%d\033[0m' % p_index,
                pool_size=pool_size,
                mode='average_inc_pad'
            )
            p_index += 1
        elif layer == 'm':
            previous_layer = MaxPool3DDNNLayer(
                incoming=previous_layer,
                name='\033[31mmax_pool%d\033[0m' % p_index,
                pool_size=pool_size
            )
            p_index += 1
        elif layer == 'u':
            p_index -= 1
            previous_layer = Unpooling3D(
                incoming=previous_layer,
                name='\033[35munpool%d\033[0m' % p_index,
                pool_size=pool_size
            )
        elif layer == 's':
            previous_layer = ElemwiseSumLayer(
                incomings=[convolutions['conv%d' % (c_index - 1)], previous_layer],
                name='short%d' % (c_index - 1)
            )
        elif layer == 'd':
            c_index -= 1
            previous_layer = Conv3DDNNLayer(
                incoming=previous_layer,
                name='\033[36mdeconv%d\033[0m' % c_index,
                num_filters=number_filters,
                filter_size=c_size,
                pad='full'
            )
        elif layer == 'o':
            previous_layer = DropoutLayer(
                incoming=previous_layer,
                name='drop%d' % (c_index - 1),
                p=0.5
            )
        elif layer == 'f':
            c_index -= 1
            previous_layer = Conv3DDNNLayer(
                incoming=previous_layer,
                name='\033[36mfinal\033[0m',
                num_filters=input_shape[1],
                filter_size=c_size,
                pad='full'
            )
        elif layer == 'r':
            previous_layer = ReshapeLayer(
                incoming=previous_layer,
                name='\033[32mreshape\033[0m',
                shape=([0], -1)
            )
        elif layer == 'S':
            previous_layer = DenseLayer(
                incoming=previous_layer,
                name='\033[32m3d_out\033[0m',
                num_units=reduce(mul, input_shape[2:], 1),
                nonlinearity=nonlinearities.softmax
            )
        elif layer == 'C':
            previous_layer = DenseLayer(
                incoming=previous_layer,
                name='\033[32mclass_out\033[0m',
                num_units=2,
                nonlinearity=nonlinearities.softmax
            )

    return previous_layer


def create_classifier_net(layers, input_shape, convo_size, pool_size, number_filters, patience, dir_name):
    return NeuralNet(

        layers=get_layers_string(layers, input_shape, convo_size, pool_size, number_filters),

        regression=False,
        objective_loss_function=objectives.categorical_crossentropy,

        # update=updates.adadelta,
        update=updates.adam,
        update_learning_rate=1e-3,

        on_epoch_finished=get_epoch_finished(os.path.join(dir_name, 'patches'), patience),

        verbose=11,
        max_epochs=200
    )


def create_cnn3d_det_string(
        cnn_path,
        input_shape,
        convo_size,
        pool_size,
        number_filters,
        patience,
        dir_name
):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'i' + cnn_path + 'r' + 'C'

    return create_classifier_net(final_layers, input_shape, convo_size, pool_size, number_filters, patience, dir_name)


def create_unet3d_seg_string(
        forward_path,
        input_shape,
        convo_size,
        pool_size,
        number_filters,
        patience,
        dir_name
):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'i' + forward_path + get_back_pathway(forward_path) + 'r' + 'S'

    return create_classifier_net(final_layers, input_shape, convo_size, pool_size, number_filters, patience, dir_name)


def create_unet3d_det_string(
        forward_path,
        input_shape,
        convo_size,
        pool_size,
        number_filters,
        patience,
        dir_name
):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    final_layers = 'i' + forward_path + get_back_pathway(forward_path) + 'r' + 'C'

    return create_classifier_net(final_layers, input_shape, convo_size, pool_size, number_filters, patience, dir_name)


def create_unet3d_shortcuts_det_string(
        forward_path,
        input_shape,
        convo_size,
        pool_size,
        number_filters,
        patience, dir_name
):
    # We create the final string defining the net with the necessary input and reshape layers
    # We assume that the user will never put these parameters as part of the net definition when
    # calling the main python function
    back_pathway = get_back_pathway(forward_path).replace('d', 'sd').replace('f', 'sf')
    final_layers = ('i' + forward_path + back_pathway + 'r' + 'C').replace('csd', 'cd')

    return create_classifier_net(final_layers, input_shape, convo_size, pool_size, number_filters, patience, dir_name)


def create_encoder3d_string(
        forward_path,
        input_shape,
        convo_size,
        pool_size,
        number_filters,
        patience,
        dir_name
):
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

        on_epoch_finished=get_epoch_finished(os.path.join(dir_name, 'encoder'), patience),

        verbose=11,
        max_epochs=200
    )

    return encoder
