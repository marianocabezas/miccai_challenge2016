import os
import matplotlib
import numpy as np
from scipy import ndimage as nd
from nibabel import load as load_nii
from sklearn.cross_validation import train_test_split
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, InverseLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.layers import Layer
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from nolearn.lasagne import NeuralNet
from nolearn.lasagne.handlers import SaveWeights
from nolearn_utils.iterators import (
    BufferedBatchIteratorMixin,
    ShuffleBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    make_iterator
)
from nolearn_utils.hooks import (
    SaveTrainingHistory, PlotTrainingHistory,
    EarlyStopping
)

matplotlib.use('Agg')


class Unpooling3D(Layer):
    def __init__(self, incoming, poolsize=2, ignore_border=True, **kwargs):
        super(Unpooling3D, self).__init__(incoming, **kwargs)
        self.poolsize = poolsize
        self.ignore_border = ignore_border

    def get_output_for(self, input, **kwargs):
        output = input.repeat(self.poolsize, axis=2).repeat(self.poolsize, axis=4).repeat(self.poolsize, axis=5)
        return output

    def get_output_shape_for(self, input_shape):
        return tuple(a*self.poolsize for a in input_shape[-2:])




def load_data(test_size=0.25, random_state=None, dir_name='/home/mariano/DATA/Challenge/',
              flair_name='FLAIR_preprocessed.nii.gz', pd_name='DP_preprocessed.nii.gz',
              t2_name='T2_preprocessed.nii.gz', gado_name='GADO_preprocessed.nii.gz',
              t1_name='T1_preprocessed.nii.gz', use_flair=True, use_pd=True, use_t2=True,
              use_gado=True, use_t1=True, rater=3
              ):

    patients = [file for file in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, file))]

    try:
        y = np.load('%slabel%d_vector.npy' % (dir_name, rater))
    except IOError:
        masks = [load_nii('%s/%s/ManualSegmentation_%d.nii.gz' % (dir_name, patient, rater)).get_data() for patient in
                 patients]
        min_shape = min([im.shape for im in masks])
        y = np.asarray(
            [nd.zoom((im - im.mean()) / im.std(),
                     [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                      float(min_shape[2]) / im.shape[2]]) for im in masks]
        ).astype(np.int32)
        np.save('%slabel%d_vector.npy' % (dir_name, rater), y)

    try:
        X = np.load(dir_name + 'image_vector.npy')
    except IOError:
        # Setting up the lists for all images
        n_patients = len(patients)
        flair = None
        pd = None
        t2 = None
        gado = None
        t1 = None

        # We load the image modalities for each patient according to the parameters
        if use_flair:
            images = [load_nii('%s/%s/%s' % (dir_name, patient, flair_name)).get_data() for patient in patients]
            min_shape = min([im.shape for im in images])
            flair = np.asarray(
                [nd.zoom((im - im.mean()) / im.std(),
                         [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                          float(min_shape[2]) / im.shape[2]]) for im in images]
            )
            flair.reshape(n_patients, 1, flair.shape[1], flair.shape[2], flair.shape[3]).astype(np.float32)
        if use_pd:
            images = [load_nii('%s/%s/%s' % (dir_name, patient, pd_name)).get_data() for patient in patients]
            min_shape = min([im.shape for im in images])
            pd = np.asarray(
                [nd.zoom((im - im.mean()) / im.std(),
                         [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                          float(min_shape[2]) / im.shape[2]]) for im in images]
            )
            pd.reshape(n_patients, 1, pd.shape[1], pd.shape[2], pd.shape[3]).astype(np.float32)
        if use_t2:
            images = [load_nii('%s/%s/%s' % (dir_name, patient, t2_name)).get_data() for patient in patients]
            min_shape = min([im.shape for im in images])
            t2 = np.asarray(
                [nd.zoom((im - im.mean()) / im.std(),
                         [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                          float(min_shape[2]) / im.shape[2]]) for im in images]
            )
            t2.reshape(n_patients, 1, t2.shape[1], t2.shape[2], t2.shape[3]).astype(np.float32)
        if use_gado:
            images = [load_nii('%s/%s/%s' % (dir_name, patient, gado_name)).get_data() for patient in patients]
            min_shape = min([im.shape for im in images])
            gado = np.asarray(
                [nd.zoom((im - im.mean()) / im.std(),
                         [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                          float(min_shape[2]) / im.shape[2]]) for im in images]
            )
            gado.reshape(n_patients, 1, gado.shape[1], gado.shape[2], gado.shape[3]).astype(np.float32)
        if use_t1:
            images = [load_nii('%s/%s/%s' % (dir_name, patient, t1_name)).get_data() for patient in patients]
            min_shape = min([im.shape for im in images])
            t1 = np.asarray(
                [nd.zoom((im - im.mean()) / im.std(),
                         [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                          float(min_shape[2]) / im.shape[2]]) for im in images]
            )
            t1.reshape(n_patients, 1, t1.shape[1], t1.shape[2], t1.shape[3]).astype(np.float32)

        X = np.stack([data for data in [flair, pd, t2, gado, t1] if data is not None], axis=1)
        np.save(dir_name + 'image_vector.npy', X)

    # y = df['label'].values.astype(np.int32)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


train_iterator_mixins = [
    ShuffleBatchIteratorMixin,
    AffineTransformBatchIteratorMixin,
    BufferedBatchIteratorMixin,
]
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)

test_iterator_mixins = [
    BufferedBatchIteratorMixin,
]
TestIterator = make_iterator('TestIterator', test_iterator_mixins)

train_iterator_kwargs = {
    'batch_size': 15,
    'buffer_size': 5,
    'affine_p': 0.5,
    'affine_scale_choices': np.linspace(0.75, 1.25, 5),
    'affine_translation_choices': np.arange(-5, 6, 1),
    'affine_rotation_choices': np.arange(-45, 50, 5),
}
train_iterator = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = {
    'batch_size': 15,
    'buffer_size': 5,
}
test_iterator = TestIterator(**test_iterator_kwargs)

save_weights = SaveWeights('./examples/mnist/model_weights.pkl', only_best=True, pickle=False)
save_training_history = SaveTrainingHistory('./examples/mnist/model_history.pkl')
plot_training_history = PlotTrainingHistory('./examples/mnist/training_history.png')
early_stopping = EarlyStopping(patience=100)

net = NeuralNet(
    layers=[
        (InputLayer, {'name': 'input', 'shape': (None, 5, None, None, None)}),

        (Conv3DDNNLayer, {'name': 'conv1', 'num_filters': 32, 'filter_size': (9, 9, 9)}),
        (MaxPool3DDNNLayer, {'name': 'pool', 'pool_size': 2}),

        (Conv3DDNNLayer, {'name': 'conv2', 'num_filters': 32, 'filter_size': (9, 9, 9)}),

        (Conv3DDNNLayer, {'name': 'conv2', 'num_filters': 9, 'filter_size': (9, 9, 32)}),
        (Unpooling3D, {'name': 'unpool', 'pool_size': 2}),
        (Conv3DDNNLayer, {'name': 'conv2', 'num_filters': 9, 'filter_size': (9, 9, 32)}),

    ],

    regression=True,
    objective_loss_function=objectives.categorical_crossentropy,

    update=updates.adadelta,

    batch_iterator_train=train_iterator,
    batch_iterator_test=test_iterator,

    on_epoch_finished=[
        save_weights,
        save_training_history,
        plot_training_history,
        early_stopping
    ],

    verbose=10,
    max_epochs=100
)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data(test_size=0.25, random_state=42)
    net.fit(X_train, y_train)

    # Load the best weights from pickled model
    net.load_params_from('./examples/mnist/model_weights.pkl')

    score = net.score(X_test, y_test)
    print 'Final score %.4f' % score
