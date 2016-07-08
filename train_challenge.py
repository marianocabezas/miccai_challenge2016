import os
import numpy as np
from data_creation import load_patches, load_patch_batch
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv3DDNNLayer, Pool3DDNNLayer
from lasagne import nonlinearities, objectives, updates
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import NeuralNet
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

    print c['g'] + 'Loading the data for the ' + c['b'] + 'Challenge' + c['nc'] + c['g'] + ' training' + c['nc']
    # Create the data
    (x, y, names) = load_patches(
        dir_name='/home/sergivalverde/w/CNN/images/CH16',
        use_flair=True,
        use_pd=True,
        use_t2=True,
        use_t1=True,
        use_gado=False,
        flair_name='FLAIR_preprocessed.nii.gz',
        pd_name='PD_preprocessed.nii.gz',
        t2_name='T2_preprocessed.nii.gz',
        gado_name=None,
        t1_name='T1_preprocessed.nii.gz',
        mask_name='Consensus.nii.gz',
        size=patch_size
    )

    seed = np.random.randint(np.iinfo(np.int32).max)
    print '-- Permuting the data'
    np.random.seed(seed)
    x_train = np.random.permutation(np.concatenate(x).astype(dtype=np.float32))
    print '-- Permuting the labels'
    np.random.seed(seed)
    y_train = np.random.permutation(np.concatenate(y).astype(dtype=np.int32))
    y_train = y_train[:, y_train.shape[1] / 2 + 1, y_train.shape[2] / 2 + 1, y_train.shape[3] / 2 + 1]
    print '-- Training vector shape = (' + ','.join([str(length) for length in x_train.shape]) + ')'
    print '-- Training labels shape = (' + ','.join([str(length) for length in y_train.shape]) + ')'

    print c['g'] + '-- Creating the ' + c['b'] + 'network' + c['nc']
    net_name = '/home/sergivalverde/w/CNN/code/CNN1/miccai_challenge2016/deep-challenge2016.init.'
    net = NeuralNet(
        layers= [
            (InputLayer, dict(name='in', shape=(None, 4, 15, 15))),
            (Conv3DDNNLayer, dict(name='conv1_1', num_filters=64, filter_size=(5,5), pad='same')),
            (Pool3DDNNLayer, dict(name='avgpool_1', pool_size=2, stride=2, mode='average_inc_pad')),
            (Conv3DDNNLayer, dict(name='conv2_1', num_filters=128, filter_size=(5, 5), pad='same')),
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
        verbose= 10,
        max_epochs=50,
        train_split=TrainSplit(eval_size=0.25),
        custom_scores=[('dsc', lambda x, y: 2 * np.sum(x * y[:, 1]) / np.sum((x + y[:, 1])))],
    )

    print c['g'] + '-- Training the ' + c['b'] + 'network' + c['nc']
    # We try to get the last weights to keep improving the net over and over
    try:
        net.load_params_from(net_name + 'model_weights.pkl')
    except:
        pass

    net.fit(x_train, y_train)

    print c['g'] + '-- Creating the test probability maps' + c['nc']
    paths = ['/'.join(name[0].rsplit('/')[:-1]) for name in names]
    roi_names = [os.path.join(path, 'test%d' % i) for path, i in zip(paths, range(15))]
    for patient, output_name in zip(names, roi_names):
        image_nii = load_nii(patient[0])
        image = np.zeros_like(image_nii.get_data())
        for batch, centers in load_patch_batch(patient, 100000, patch_size):
            y_pred = net.predict_proba(batch)
            [x, y, z] = np.stack(centers, axis=1)
            image[x, y, z] = y_pred[:, 1]

        image_nii.get_data()[:] = image
        image_nii.to_filename(output_name)

    (x, y, names) = load_patches(
        dir_name='/home/sergivalverde/w/CNN/images/CH16',
        use_flair=True,
        use_pd=True,
        use_t2=True,
        use_t1=True,
        use_gado=False,
        flair_name='FLAIR_preprocessed.nii.gz',
        pd_name='PD_preprocessed.nii.gz',
        t2_name='T2_preprocessed.nii.gz',
        gado_name=None,
        t1_name='T1_preprocessed.nii.gz',
        mask_name='Consensus.nii.gz',
        size=patch_size,
        roi_names=roi_names
    )

    net_name = '/home/sergivalverde/w/CNN/code/CNN1/miccai_challenge2016/deep-challenge2016.final.'
    net = NeuralNet(
        layers=[
            (InputLayer, dict(name='in', shape=(None, 4, 15, 15))),
            (Conv3DDNNLayer, dict(name='conv1_1', num_filters=64, filter_size=(5, 5), pad='same')),
            (Pool3DDNNLayer, dict(name='avgpool_1', pool_size=2, stride=2, mode='average_inc_pad')),
            (Conv3DDNNLayer, dict(name='conv2_1', num_filters=128, filter_size=(5, 5), pad='same')),
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
        ],
        verbose=10,
        max_epochs=2000,
        train_split=TrainSplit(eval_size=0.25),
        custom_scores=[('dsc', lambda x, y: 2 * np.sum(x * y[:, 1]) / np.sum((x + y[:, 1])))],
    )

    try:
        net.load_params_from(net_name + 'model_weights.pkl')
    except:
        pass

    net.fit(x_train, y_train)


if __name__ == '__main__':
    main()