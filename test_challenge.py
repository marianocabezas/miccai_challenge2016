from __future__ import print_function
import os
import sys
import argparse
from time import strftime
from nibabel import load as load_nii
import numpy as np
from scipy import ndimage
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne import nonlinearities, objectives, updates
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import NeuralNet, BatchIterator
from cnn.data_creation import load_patch_batch_percent


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
    batch_size = 100000
    min_size = 30

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] +
          '<Loading the net ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
    net_name = '/usr/local/nets/deep-challenge2016.init.model_weights.pkl' if options['docker'] \
        else './deep-challenge2016.init.model_weights.pkl'
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
        verbose=10,
        max_epochs=50,
        train_split=TrainSplit(eval_size=0.25),
        custom_scores=[('dsc', lambda p, t: 2 * np.sum(p * t[:, 1]) / np.sum((p + t[:, 1])))],
    )
    net.load_params_from(net_name)

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] +
          '<Creating the probability map ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
    names = np.array([options['flair'], options['pd'], options['t2'], options['t1']])
    image_nii = load_nii(options['flair'])
    image1 = np.zeros_like(image_nii.get_data())
    print('0% of data tested', end='\r')
    sys.stdout.flush()
    for batch, centers, percent in load_patch_batch_percent(names, batch_size, patch_size):
        y_pred = net.predict_proba(batch)
        print('%f%% of data tested' % percent, end='\r')
        sys.stdout.flush()
        [x, y, z] = np.stack(centers, axis=1)
        image1[x, y, z] = y_pred[:, 1]

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] +
          '<Loading the net ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
    net_name = '/usr/local/nets/deep-challenge2016.final.model_weights.pkl' if options['docker'] \
        else './deep-challenge2016.final.model_weights.pkl'
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
        custom_scores=[('dsc', lambda t, p: 2 * np.sum(t * p[:, 1]) / np.sum((t + p[:, 1])))],
    )
    net.load_params_from(net_name)

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] +
          '<Creating the probability map ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
    image2 = np.zeros_like(image_nii.get_data())
    print('0% of data tested', end='\r')
    sys.stdout.flush()
    for batch, centers, percent in load_patch_batch_percent(names, batch_size, patch_size):
        y_pred = net.predict_proba(batch)
        print('%f%% of data tested' % percent, end='\r')
        sys.stdout.flush()
        [x, y, z] = np.stack(centers, axis=1)
        image2[x, y, z] = y_pred[:, 1]

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] +
          '<Saving to file ' + c['b'] + options['output'] + c['nc'] + c['g'] + '>' + c['nc'])
    image = (image1 * image2) > 0.5

    # filter candidates < min_size
    labels, num_labels = ndimage.label(image)
    lesion_list = np.unique(labels)
    num_elements_by_lesion = ndimage.labeled_comprehension(image, labels, lesion_list, np.sum, float, 0)
    filt_min_size = num_elements_by_lesion >= min_size
    lesion_list = lesion_list[filt_min_size]
    image = reduce(np.logical_or, map(lambda lab: lab == labels, lesion_list))

    image_nii.get_data()[:] = np.roll(np.roll(image, 1, axis=0), 1, axis=1)
    path = '/'.join(options['t1'].rsplit('/')[:-1])
    outputname = options['output'].rsplit('/')[-1]
    image_nii.to_filename(os.path.join(path, outputname))

    if not options['docker']:
        path = '/'.join(options['output'].rsplit('/')[:-1])
        case = options['output'].rsplit('/')[-1]
        gt = load_nii(os.path.join(path, 'Consensus.nii.gz')).get_data().astype(dtype=np.bool)
        dsc = np.sum(2.0 * np.logical_and(gt, image)) / (np.sum(gt) + np.sum(image))
        print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + c['g'] +
              '<DSC value for ' + c['c'] + case + c['g'] + ' = ' + c['b'] + str(dsc) + c['nc'] + c['g'] + '>' + c['nc'])


if __name__ == '__main__':
    main()
