import os
import matplotlib
import numpy as np
from scipy import ndimage as nd
from nibabel import load as load_nii
from sklearn.cross_validation import train_test_split
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, FeaturePoolLayer, InverseLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
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


def load_data(test_size=0.25, random_state=None, dir_name='~/DATA/Challenge',
              flair_name='FLAIR_preprocessed.nii.gz', pd_name='DP_preprocessed.nii.gz',
              t2_name='T2_preprocessed.nii.gz', gado_name='GADO_preprocessed.nii.gz',
              t1_name='T1_preprocessed.nii.gz', use_flair=True, use_pd=True, use_t2=True,
              use_gado=True, use_t1=True
              ):
    # Setting up the lists for all images
    patients = [file for file in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, file))]
    n_patients = len(patients)
    flair = None
    pd = None
    t2 = None
    gado = None
    t1 = None

    # We load the image modalities for each patient according to the parameters
    if use_flair:
        flair = [load_nii('%s/%s/%s' % (dir_name, patient, flair_name)).get_data() for patient in patients]
    if use_pd:
        pd = np.asarray([load_nii('%s/%s/%s' % (dir_name, patient, pd_name)).get_data() for patient in patients])
    if use_t2:
        t2 = np.asarray([load_nii('%s/%s/%s' % (dir_name, patient, t2_name)).get_data() for patient in patients])
    if use_gado:
        gado = np.asarray([load_nii('%s/%s/%s' % (dir_name, patient, gado_name)).get_data() for patient in patients])
    if use_t1:
        t1 = np.asarray([load_nii('%s/%s/%s' % (dir_name, patient, t1_name)).get_data() for patient in patients])

    # The size of the final numpy array should be: n_patients x n_images x size_of_image
    if use_flair:
        min_shape = min([im.shape for im in flair])
        flair = np.asarray([nd.zoom(im, [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                                         float(min_shape[2]) / im.shape[2]]) for im in flair])
        flair.reshape(n_patients, 1, flair.shape[1], flair.shape[2], flair.shape[3]).astype(np.float32)
    if use_pd:
        min_shape = min([im.shape for im in pd])
        pd = np.asarray([nd.zoom(im, [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                                      float(min_shape[2]) / im.shape[2]]) for im in pd])
        pd.reshape(n_patients, 1, pd.shape[1], pd.shape[2], pd.shape[3]).astype(np.float32)
    if use_t2:
        min_shape = min([im.shape for im in t2])
        t2 = np.asarray([nd.zoom(im, [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                                      float(min_shape[2]) / im.shape[2]]) for im in t2])
        t2.reshape(n_patients, 1, t2.shape[1], t2.shape[2], t2.shape[3]).astype(np.float32)
    if use_gado:
        min_shape = min([im.shape for im in gado])
        gado = np.asarray([nd.zoom(im, [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                                        float(min_shape[2]) / im.shape[2]]) for im in gado])
        gado.reshape(n_patients, 1, gado.shape[1], gado.shape[2], gado.shape[3]).astype(np.float32)
    if use_t1:
        min_shape = min([im.shape for im in t1])
        t1 = np.asarray([nd.zoom(im, [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                                      float(min_shape[2]) / im.shape[2]]) for im in t1])
        t1.reshape(n_patients, 1, t1.shape[1], t1.shape[2], t1.shape[3]).astype(np.float32)
    images = filter(None, [flair, pd, t2, gado, t1])
    image_vector = np.stack(images, axis=1)
    n_images = len(images)
    x_len = image_vector.shape[2]
    y_len = image_vector.shape[3]
    z_len = image_vector.shape[4]

    df = pd.read_csv('examples/mnist/train.csv')
    X = df[df.columns[1:]].values.reshape(-1, 1, 28, 28).astype(np.float32)
    X = X / 255
    y = df['label'].values.astype(np.int32)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


image_size = 28
batch_size = 1024

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
    'batch_size': batch_size,
    'buffer_size': 5,
    'affine_p': 0.5,
    'affine_scale_choices': np.linspace(0.75, 1.25, 5),
    'affine_translation_choices': np.arange(-5, 6, 1),
    'affine_rotation_choices': np.arange(-45, 50, 5),
}
train_iterator = TrainIterator(**train_iterator_kwargs)

test_iterator_kwargs = {
    'batch_size': batch_size,
    'buffer_size': 5,
}
test_iterator = TestIterator(**test_iterator_kwargs)

save_weights = SaveWeights('./examples/mnist/model_weights.pkl', only_best=True, pickle=False)
save_training_history = SaveTrainingHistory('./examples/mnist/model_history.pkl')
plot_training_history = PlotTrainingHistory('./examples/mnist/training_history.png')
early_stopping = EarlyStopping(patience=100)

net = NeuralNet(
    layers=[
        (InputLayer, dict(name='input', shape=(None, 1, image_size, image_size))),
        (Conv3DDNNLayer, dict(name='conv1', num_filters=32, filter_size=(9, 9, 9), pad='same')),
        (MaxPool3DDNNLayer, dict(name='pool', pool_size=2)),

        (Conv3DDNNLayer, dict(name='conv2', num_filters=32, filter_size=(9, 9, 9), pad='same')),

        (InverseLayer, dict(name='deconv2', incoming='conv2', layer='conv2')),
        (InverseLayer, dict(name='unpool', incoming='deconv2', layer='pool')),
        (InverseLayer, dict(name='deconv1', incoming='unpool', layer='pool')),

    ],

    regression=False,
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
