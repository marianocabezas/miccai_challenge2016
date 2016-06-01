import os
import numpy as np
from scipy import ndimage as nd
from nibabel import load as load_nii
from sklearn.cross_validation import train_test_split


def load_and_vectorize(name, dir_name, datatype=np.float32):
    patients = [file for file in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, file))]
    images = [load_nii('%s/%s/%s' % (dir_name, patient, name)).get_data() for patient in patients]
    min_shape = min([im.shape for im in images])
    data = np.asarray(
        [nd.zoom((im - im.mean()) / im.std(),
                 [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                  float(min_shape[2]) / im.shape[2]]) for im in images]
    )
    data.reshape(data.shape[0], 1, data.shape[1], data.shape[2], data.shape[3]).astype(datatype)
    return data


def load_encoder_data(test_size=0.25, random_state=None, dir_name='/home/mariano/DATA/Challenge/',
                  flair_name='FLAIR_preprocessed.nii.gz', pd_name='DP_preprocessed.nii.gz',
                  t2_name='T2_preprocessed.nii.gz', gado_name='GADO_preprocessed.nii.gz',
                  t1_name='T1_preprocessed.nii.gz', use_flair=True, use_pd=True, use_t2=True,
                  use_gado=True, use_t1=True, rater=3
                  ):

    patients = [file for file in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, file))]

    try:
        X = np.load(dir_name + 'image_vector.npy')
    except IOError:
        # Setting up the lists for all images
        flair = None
        pd = None
        t2 = None
        gado = None
        t1 = None

        # We load the image modalities for each patient according to the parameters
        if use_flair:
            flair = load_and_vectorize(flair_name, dir_name)
        if use_pd:
            pd = load_and_vectorize(pd_name, dir_name)
        if use_t2:
            t2 = load_and_vectorize(t2_name, dir_name)
        if use_gado:
            gado = load_and_vectorize(gado_name, dir_name)
        if use_t1:
            t1 = load_and_vectorize(t1_name, dir_name)

        X = np.stack([data for data in [flair, pd, t2, gado, t1] if data is not None], axis=1)
        np.save(dir_name + 'image_vector_encoder.npy', X)

    y = np.reshape(X, [X.shape[0], -1])

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_data(test_size=0.25, random_state=None, dir_name='/home/mariano/DATA/Challenge/',
              flair_name='FLAIR_preprocessed.nii.gz', pd_name='DP_preprocessed.nii.gz',
              t2_name='T2_preprocessed.nii.gz', gado_name='GADO_preprocessed.nii.gz',
              t1_name='T1_preprocessed.nii.gz', use_flair=True, use_pd=True, use_t2=True,
              use_gado=True, use_t1=True, rater=3
              ):

    try:
        y = np.load('%slabel%d_vector.npy' % (dir_name, rater))
    except IOError:
        patients = [file for file in sorted(os.listdir(dir_name)) if os.path.isdir(os.path.join(dir_name, file))]
        masks = [load_nii('%s/%s/ManualSegmentation_%d.nii.gz' % (dir_name, patient, rater)).get_data() for patient in
                 patients]
        min_shape = min([im.shape for im in masks])
        y = np.asarray(
            [nd.zoom((im - im.mean()) / im.std(),
                     [float(min_shape[0]) / im.shape[0], float(min_shape[1]) / im.shape[1],
                      float(min_shape[2]) / im.shape[2]]) for im in masks]
        ).astype(np.uint8)
        np.save('%slabel%d_vector.npy' % (dir_name, rater), y)

    try:
        X = np.load(dir_name + 'image_vector.npy')
    except IOError:
        # Setting up the lists for all images
        flair = None
        pd = None
        t2 = None
        gado = None
        t1 = None

        # We load the image modalities for each patient according to the parameters
        if use_flair:
            flair = load_and_vectorize(flair_name, dir_name)
        if use_pd:
            pd = load_and_vectorize(pd_name, dir_name)
        if use_t2:
            t2 = load_and_vectorize(t2_name, dir_name)
        if use_gado:
            gado = load_and_vectorize(gado_name, dir_name)
        if use_t1:
            t1 = load_and_vectorize(t1_name, dir_name)

        X = np.stack([data for data in [flair, pd, t2, gado, t1] if data is not None], axis=1)
        np.save(dir_name + 'image_vector_encoder.npy', X)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)