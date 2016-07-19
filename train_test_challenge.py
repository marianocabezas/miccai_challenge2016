from __future__ import print_function
import os
import sys
from time import strftime
import numpy as np
from data_creation import load_patches, leave_one_out, load_patch_batch_percent
from data_creation import load_patch_vectors_by_name, load_thresholded_images_by_name
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne import nonlinearities, objectives, updates
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.hooks import SaveTrainingHistory, PlotTrainingHistory, EarlyStopping
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
    c = color_codes()
    patch_size = (15, 15, 15)
    batch_size = 100000
    dir_name = '/home/sergivalverde/w/CNN/images/CH16'
    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' +
          c['g'] + 'Loading the data for the leave-one-out test' + c['nc'])
    # Create the data
    (x, y, names) = load_patches(
        dir_name=dir_name,
        use_flair=True,
        use_pd=True,
        use_t2=True,
        use_gado=False,
        use_t1=True,
        flair_name='FLAIR_preprocessed.nii.gz',
        pd_name='DP_preprocessed.nii.gz',
        t2_name='T2_preprocessed.nii.gz',
        gado_name=None,
        t1_name='T1_preprocessed.nii.gz',
        mask_name='Consensus.nii.gz',
        size=patch_size
    )
    seed = np.random.randint(np.iinfo(np.int32).max)

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ' + 'Starting leave-one-out' + c['nc'])

    for x_train, y_train, i in leave_one_out(x, y):
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
                SaveTrainingHistory(net_name + 'model_history.pkl'),
                PlotTrainingHistory(net_name + 'training_history.png'),
                EarlyStopping(patience=10)
            ],
            verbose=10,
            max_epochs=50,
            train_split=TrainSplit(eval_size=0.25),
            custom_scores=[('dsc', lambda p, t: 2 * np.sum(p * t[:, 1]) / np.sum((p + t[:, 1])))],
        )

        try:
            net.load_params_from(net_name + 'model_weights.pkl')
        except IOError:
            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
                  c['g'] + 'Loading the data for ' + c['b'] + 'iteration 1' + c['nc'])
            # Create the data
            print('              Permuting the data')
            np.random.seed(seed)
            x_train = np.random.permutation(np.concatenate(x_train).astype(dtype=np.float32))
            print('              Permuting the labels')
            np.random.seed(seed)
            y_train = np.random.permutation(np.concatenate(y_train).astype(dtype=np.int32))
            y_train = y_train[:, y_train.shape[1] / 2 + 1, y_train.shape[2] / 2 + 1, y_train.shape[3] / 2 + 1]
            print('              Training vector shape = (' + ','.join([str(length) for length in x_train.shape]) + ')')
            print('              Training labels shape = (' + ','.join([str(length) for length in y_train.shape]) + ')')

            print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
                  'Training (' + c['b'] + 'initial' + c['nc'] + c['g'] + ')' + c['nc'])
            # We try to get the last weights to keep improving the net over and over
            net.fit(x_train, y_train)

        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<Creating the probability map ' + c['b'] + '1' + c['nc'] + c['g'] + '>' + c['nc'])
        flair_name = os.path.join(path, 'FLAIR_preprocessed.nii.gz')
        pd_name = os.path.join(path, 'DP_preprocessed.nii.gz')
        t2_name = os.path.join(path, 'T2_preprocessed.nii.gz')
        t1_name = os.path.join(path, 'T1_preprocessed.nii.gz')
        names_test = np.array([flair_name, pd_name, t2_name, t1_name])
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

        ''' Here we get the seeds '''
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
              c['g'] + '<Looking for seeds>' + c['nc'])
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
                SaveTrainingHistory(net_name + 'model_history.pkl'),
                PlotTrainingHistory(net_name + 'training_history.png'),
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
        rois = load_thresholded_images_by_name(roi_names, threshold=0.5)
        print('              Loading FLAIR images')
        flair, y_train = load_patch_vectors_by_name(names_lou[0, :], mask_names, patch_size, rois)
        print('              Loading PD images')
        pd, _ = load_patch_vectors_by_name(names_lou[1, :], mask_names, patch_size, rois)
        print('              Loading T2 images')
        t2, _ = load_patch_vectors_by_name(names_lou[2, :], mask_names, patch_size, rois)
        print('              Loading T1 images')
        t1, _ = load_patch_vectors_by_name(names_lou[3, :], mask_names, patch_size, rois)

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
        image2 = np.zeros_like(image_nii.get_data())
        print('              0% of data tested', end='\r')
        sys.stdout.flush()
        for batch, centers, percent in load_patch_batch_percent(names, batch_size, patch_size):
            y_pred = net.predict_proba(batch)
            print('              %f%% of data tested' % percent, end='\r')
            sys.stdout.flush()
            [x, y, z] = np.stack(centers, axis=1)
            image2[x, y, z] = y_pred[:, 1]

        image = (image1 * image2) > 0.5
        image_nii.get_data()[:] = np.roll(np.roll(image, 1, axis=0), 1, axis=1)

        gt = load_nii(os.path.join(path, 'Consensus.nii.gz')).get_data().astype(dtype=np.bool)
        dsc = np.sum(2.0 * np.logical_and(gt, image)) / (np.sum(gt) + np.sum(image))
        print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] +
              '<DSC value for ' + c['c'] + case + c['g'] + ' = ' + c['b'] + str(dsc) + c['nc'] + c['g'] + '>' + c['nc'])


if __name__ == '__main__':
    main()
