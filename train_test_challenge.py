from __future__ import print_function
import argparse
import os
import sys
from time import strftime
import numpy as np
from cnn.data_creation import load_patch_batch_percent
from cnn.data_creation import load_patch_vectors_by_name_pr, load_patch_vectors_by_name
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne import nonlinearities, objectives, updates
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import EarlyStopping
from nibabel import load as load_nii


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

    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/home/sergivalverde/w/CNN/images/CH16')
    parser.add_argument('--flair', action='store', dest='flair', default='FLAIR_preprocessed.nii.gz')
    parser.add_argument('--pd', action='store', dest='pd', default='DP_preprocessed.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='T2_preprocessed.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='T1_preprocessed.nii.gz')
    parser.add_argument('--mask', action='store', dest='mask', default='Consensus.nii.gz')
    options = vars(parser.parse_args())

    c = color_codes()
    patch_size = (15, 15, 15)
    batch_size = 100000
    # Create the data
    patients = [f for f in sorted(os.listdir(options['dir_name']))
                if os.path.isdir(os.path.join(options['dir_name'], f))]
    flair_names = [os.path.join(options['dir_name'], patient, options['flair']) for patient in patients]
    pd_names = [os.path.join(options['dir_name'], patient, options['pd']) for patient in patients]
    t2_names = [os.path.join(options['dir_name'], patient, options['t2']) for patient in patients]
    t1_names = [os.path.join(options['dir_name'], patient, options['t1']) for patient in patients]
    names = np.stack([name for name in [flair_names, pd_names, t2_names, t1_names]])
    seed = np.random.randint(np.iinfo(np.int32).max)

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting leave-one-out' + c['nc'])

    for i in range(0, 15):
        case = names[0, i].rsplit('/')[-2]
        path = '/'.join(names[0, i].rsplit('/')[:-1])
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['nc'] + 'Patient ' + c['b'] + case + c['nc'])
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Running iteration ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
        net_name = os.path.join(path, 'deep-challenge2016.init.')
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
            on_epoch_finished=[
                SaveWeights(net_name + 'model_weights.pkl', only_best=True, pickle=False),
                EarlyStopping(patience=10)
            ],
            verbose=10,
            max_epochs=50,
            train_split=TrainSplit(eval_size=0.25),
            custom_scores=[('dsc', lambda p, t: 2 * np.sum(p * t[:, 1]) / np.sum((p + t[:, 1])))],
        )
        flair_name = os.path.join(path, options['flair'])
        pd_name = os.path.join(path, options['pd'])
        t2_name = os.path.join(path, options['t2'])
        t1_name = os.path.join(path, options['t1'])
        names_test = np.array([flair_name, pd_name, t2_name, t1_name])
        outputname1 = os.path.join(path, 'test' + str(i) + '.iter1.nii.gz')
        try:
            image_nii = load_nii(outputname1)
            image1 = image_nii.getdata()
            net.load_params_from(net_name + 'model_weights.pkl')
        except IOError:
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Loading the data for ' + c['b'] + 'iteration 1' + c['nc'])
            names_lou = np.concatenate([names[:, :i], names[:, i + 1:]], axis=1)
            paths = ['/'.join(name.rsplit('/')[:-1]) for name in names_lou[0, :]]
            mask_names = [os.path.join(p_path, 'Consensus.nii.gz') for p_path in paths]
            print('              Loading FLAIR images')
            flair, y_train = load_patch_vectors_by_name(names_lou[0, :], mask_names, patch_size)
            print('              Loading PD images')
            pd, _ = load_patch_vectors_by_name(names_lou[1, :], mask_names, patch_size)
            print('              Loading T2 images')
            t2, _ = load_patch_vectors_by_name(names_lou[2, :], mask_names, patch_size)
            print('              Loading T1 images')
            t1, _ = load_patch_vectors_by_name(names_lou[3, :], mask_names, patch_size)

            print('              Creating data vector')
            x_train = [np.stack(images, axis=1) for images in zip(*[flair, pd, t2, t1])]

            print('              Permuting the data')
            np.random.seed(seed)
            x_train = np.random.permutation(np.concatenate(x_train[:i] + x_train[i + 1:]).astype(dtype=np.float32))
            print('              Permuting the labels')
            np.random.seed(seed)
            y_train = np.random.permutation(np.concatenate(y_train[:i] + y_train[i + 1:]).astype(dtype=np.int32))
            y_train = y_train[:, y_train.shape[1] / 2 + 1, y_train.shape[2] / 2 + 1, y_train.shape[3] / 2 + 1]
            print('              Training vector shape ='
                  ' (' + ','.join([str(length) for length in x_train.shape]) + ')')
            print('              Training labels shape ='
                  ' (' + ','.join([str(length) for length in y_train.shape]) + ')')

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  'Training (' + c['b'] + 'initial' + c['nc'] + c['g'] + ')' + c['nc'])
            # We try to get the last weights to keep improving the net over and over
            net.fit(x_train, y_train)

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<Creating the probability map ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
            flair_name = os.path.join(path, options['flair'])
            image_nii = load_nii(flair_name)
            image1 = np.zeros_like(image_nii.get_data())
            print('              0% of data tested', end='\r')
            sys.stdout.flush()
            for batch, centers, percent in load_patch_batch_percent(names_test, batch_size, patch_size):
                y_pred = net.predict_proba(batch)
                print('              %f%% of data tested' % percent, end='\r')
                sys.stdout.flush()
                [x, y, z] = np.stack(centers, axis=1)
                image1[x, y, z] = y_pred[:, 1]

            image_nii.get_data()[:] = image1
            image_nii.to_filename(outputname1)
        ''' Here we get the seeds '''
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
              c['g'] + '<Looking for seeds for the final iteration>' + c['nc'])
        for patient in np.rollaxis(np.concatenate([names[:, :i], names[:, i+1:]], axis=1), 1):
            output_name = os.path.join('/'.join(patient[0].rsplit('/')[:-1]), 'test' + str(i) + '.iter1.nii.gz')
            try:
                load_nii(output_name)
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + '    Patient ' + patient[0].rsplit('/')[-2] + ' already done' + c['nc'])
            except IOError:
                print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                      c['g'] + '     Testing with patient ' + c['b'] + patient[0].rsplit('/')[-2] + c['nc'])
                image_nii = load_nii(patient[0])
                image = np.zeros_like(image_nii.get_data())
                print('    0% of data tested', end='\r')
                for batch, centers, percent in load_patch_batch_percent(patient, 100000, patch_size):
                    y_pred = net.predict_proba(batch)
                    print('    %f%% of data tested' % percent, end='\r')
                    [x, y, z] = np.stack(centers, axis=1)
                    image[x, y, z] = y_pred[:, 1]

                print(c['g'] + '              -- Saving image ' + c['b'] + output_name + c['nc'])
                image_nii.get_data()[:] = image
                image_nii.to_filename(output_name)

        ''' Here we perform the last iteration '''
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Running iteration ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
        outputname2 = os.path.join(path, 'test' + str(i) + '.iter2.nii.gz')
        try:
            image_nii = load_nii(outputname2)
            image2 = image_nii.getdata()
        except IOError:
            net_name = os.path.join(path, 'deep-challenge2016.final.')
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
                on_epoch_finished=[
                    SaveWeights(net_name + 'model_weights.pkl', only_best=True, pickle=False),
                    EarlyStopping(patience=50)
                ],
                batch_iterator_train=BatchIterator(batch_size=4096),
                verbose=10,
                max_epochs=2000,
                train_split=TrainSplit(eval_size=0.25),
                custom_scores=[('dsc', lambda p, t: 2 * np.sum(p * t[:, 1]) / np.sum((p + t[:, 1])))],
            )

            try:
                net.load_params_from(net_name + 'model_weights.pkl')
            except IOError:
                pass
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Loading the data for ' + c['b'] + 'iteration 2' + c['nc'])
            names_lou = np.concatenate([names[:, :i], names[:, i + 1:]], axis=1)
            paths = ['/'.join(name.rsplit('/')[:-1]) for name in names_lou[0, :]]
            roi_names = [os.path.join(p_path, 'test' + str(i) + '.iter1.nii.gz') for p_path in paths]
            mask_names = [os.path.join(p_path, 'Consensus.nii.gz') for p_path in paths]
            pr_maps = [load_nii(roi_name).get_data() for roi_name in roi_names]
            print('              Loading FLAIR images')
            flair, y_train = load_patch_vectors_by_name_pr(names_lou[0, :], mask_names, patch_size, pr_maps)
            print('              Loading PD images')
            pd, _ = load_patch_vectors_by_name_pr(names_lou[1, :], mask_names, patch_size, pr_maps)
            print('              Loading T2 images')
            t2, _ = load_patch_vectors_by_name_pr(names_lou[2, :], mask_names, patch_size, pr_maps)
            print('              Loading T1 images')
            t1, _ = load_patch_vectors_by_name_pr(names_lou[3, :], mask_names, patch_size, pr_maps)

            print('              Creating data vector')
            x_train = [np.stack(images, axis=1) for images in zip(*[flair, pd, t2, t1])]

            print('              Permuting the data')
            np.random.seed(seed)
            x_train = np.random.permutation(np.concatenate(x_train[:i] + x_train[i+1:]).astype(dtype=np.float32))
            print('              Permuting the labels')
            np.random.seed(seed)
            y_train = np.random.permutation(np.concatenate(y_train[:i] + y_train[i+1:]).astype(dtype=np.int32))
            y_train = y_train[:, y_train.shape[1] / 2 + 1, y_train.shape[2] / 2 + 1, y_train.shape[3] / 2 + 1]
            print('              Training vector shape = (' + ','.join([str(length) for length in x_train.shape]) + ')')
            print('              Training labels shape = (' + ','.join([str(length) for length in y_train.shape]) + ')')
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Training (' + c['b'] + 'final' + c['nc'] + c['g'] + ')' + c['nc'])
            net.fit(x_train, y_train)

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  '<Creating the probability map ' + c['b'] + '2' + c['nc'] + c['g'] + '>' + c['nc'])
            image_nii = load_nii(flair_name)
            image2 = np.zeros_like(image_nii.get_data())
            print('              0% of data tested', end='\r')
            sys.stdout.flush()
            for batch, centers, percent in load_patch_batch_percent(names_test, batch_size, patch_size):
                y_pred = net.predict_proba(batch)
                print('              %f%% of data tested' % percent, end='\r')
                sys.stdout.flush()
                [x, y, z] = np.stack(centers, axis=1)
                image2[x, y, z] = y_pred[:, 1]

            image_nii.get_data()[:] = image2
            image_nii.to_filename(os.path.join(path, 'test' + str(i) + '.iter2.nii.gz'))

        image = (image1 * image2) > 0.5
        seg = np.roll(np.roll(image, 1, axis=0), 1, axis=1)
        image_nii.get_data()[:] = seg
        image_nii.to_filename(os.path.join(path, 'test' + str(i) + '.final.nii.gz'))

        gt = load_nii(os.path.join(path, 'Consensus.nii.gz')).get_data().astype(dtype=np.bool)
        dsc = np.sum(2.0 * np.logical_and(gt, seg)) / (np.sum(gt) + np.sum(seg))
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<DSC value for ' + c['c'] + case + c['g'] + ' = ' + c['b'] + str(dsc) + c['nc'] + c['g'] + '>' + c['nc'])


if __name__ == '__main__':
    main()
