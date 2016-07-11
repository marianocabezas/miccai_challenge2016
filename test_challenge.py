from __future__ import print_function
import sys
import argparse
from time import strftime
from nibabel import load as load_nii
import numpy as np
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne import nonlinearities, objectives, updates
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import NeuralNet, BatchIterator
from data_creation import load_patch_batch_percent
from data_creation import sum_patches_to_image


def color_codes():
    codes = {'g': '\033[32m',
             'c': '\033[36m',
             'bg': '\033[32;1m',
             'b': '\033[1m',
             'nc': '\033[0m',
             'gc': '\033[32m, \033[0m'
             }
    return codes


def main():
    # Parse command line options
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('--flair', action='store', dest='flair', default='FLAIR_preprocessed.nii.gz')
    parser.add_argument('--pd', action='store', dest='pd', default='DP_preprocessed.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='T2_preprocessed.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='T1_preprocessed.nii.gz')
    parser.add_argument('--output', action='store', dest='output', default='output.nii.gz')
    parser.add_argument('--no-docker', action='store_false', dest='docker', default=True)

    c = color_codes()
    patch_size = (15, 15, 15)
    options = vars(parser.parse_args())
    batch_size = 500000

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] + '<Loading the net>' + c['nc'])
    net_name = '/usr/local/nets/deep-challenge2016.final.model_weights.pkl' if options['docker'] \
        else '/home/sergivalverde/w/CNN/code/CNN1/miccai_challenge2016/deep-challenge2016.final.model_weights.pkl'
    net = NeuralNet(
        layers=[
            (InputLayer, dict(name='in', shape=(None, 4, 15, 15, 15))),
            (Conv3DDNNLayer, dict(name='conv1_1', num_filters=32, filter_size=(5, 5, 5), pad='same')),
            (Pool3DDNNLayer, dict(name='avgpool_1', pool_size=2, stride=2, mode='average_inc_pad')),
            (Conv3DDNNLayer, dict(name='conv2_1', num_filters=64, filter_size=(5, 5, 5), pad='same')),
            (Pool3DDNNLayer, dict(name='avgpool_2', pool_size=2, stride=2, mode='average_inc_pad')),
            (DropoutLayer, dict(name='l2drop', p=0.5)),
            (DenseLayer, dict(name='l1', num_units=256)),
            (DenseLayer, dict(name='out', num_units=2, nonlinearity=nonlinearities.softmax)),
        ],
        objective_loss_function=objectives.categorical_crossentropy,
        update=updates.adam,
        update_learning_rate=0.0001,
        batch_iterator_train=BatchIterator(batch_size=4096),
        verbose=10,
        max_epochs=2000,
        train_split=TrainSplit(eval_size=0.25),
        custom_scores=[('dsc', lambda x, y: 2 * np.sum(x * y[:, 1]) / np.sum((x + y[:, 1])))],
    )
    net.load_params_from(net_name)

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] + '<Creating the probability map>' + c['nc'])
    names = np.array([options['flair'], options['pd'], options['t2'], options['t1']])
    image_nii = load_nii(options['flair'])
    image = np.zeros_like(image_nii.get_data())
    print('-- Output shape = (' + ','.join([str(length) for length in image.shape]) + ')')
    #print('0% of data tested', end='\r')
    sys.stdout.flush()
    for batch, centers, percent in load_patch_batch_percent(names, batch_size, patch_size):
        print('-- centers shape = (' + ','.join([str(len(centers)), str(len(centers[0]))]) + ')')
        print('-- image shape = (' + ','.join([str(length) for length in image.shape]) + ')')
        print('-- batch shape = (' + ','.join([str(length) for length in batch.shape]) + ')')
        y_pred = net.predict_proba(batch)
        print('-- y_pred shape = (' + ','.join([str(length) for length in y_pred.shape]) + ')')
        #print('%f% of data tested' % percent, end='\r')
        sys.stdout.flush()
        image += sum_patches_to_image(y_pred, centers, image)

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] +
          '<Saving to file' + c['b'] + options['output'] + c['nc'] + c['g'] + '>' + c['nc'])
    image_nii.get_data()[:] = image
    image_nii.to_filename(options['output'])


if __name__ == '__main__':
    main()
