import os
import argparse
import numpy as np
import cPickle
from data_creation import load_unet_data, load_encoder_data, load_patches
from data_creation import reshape_save_nifti_to_dir, reshape_save_nifti
from data_creation import get_sufix
from nets import create_unet3d_string, create_encoder3d_string, create_patches3d_string


def main():
    # Parse command line options
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='folder', default='/home/mariano/DATA/Challenge/')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False)
    parser.add_argument('-c', '--convolution-size', action='store', dest='convo_size', type=int, default=3)
    parser.add_argument('-p', '--pool-size', action='store', dest='pool_size', type=int, default=2)
    parser.add_argument('-t', '--test-size', action='store', dest='test_size', type=float, default=0.25)
    parser.add_argument('-n', '--number-filters', action='store', dest='number_filters', type=int, default=4)
    parser.add_argument('-l', '--forward-layers', action='store', dest='layers', default='cac')
    parser.add_argument('-w', '--win-size', action='store', dest='patch_size', type=int, nargs=3, default=(9, 9, 9))
    parser.add_argument('-i', '--image-size', action='store', dest='min_shape', type=int, nargs=3, default=None)
    parser.add_argument('--use-gado', action='store_true', dest='use_gado', default=False)
    parser.add_argument('--no-gado', action='store_false', dest='use_gado', default=False)
    parser.add_argument('--gado', action='store', dest='gado', default='GADO_preprocessed.nii.gz')
    parser.add_argument('--use-flair', action='store_true', dest='use_flair', default=True)
    parser.add_argument('--no-flair', action='store_false', dest='use_flair', default=True)
    parser.add_argument('--flair', action='store', dest='flair', default='FLAIR_preprocessed.nii.gz')
    parser.add_argument('--use-pd', action='store_true', dest='use_pd', default=True)
    parser.add_argument('--no-pd', action='store_false', dest='use_pd', default=True)
    parser.add_argument('--pd', action='store', dest='pd', default='DP_preprocessed.nii.gz')
    parser.add_argument('--use-t2', action='store_true', dest='use_t2', default=True)
    parser.add_argument('--no-t2', action='store_false', dest='use_t2', default=True)
    parser.add_argument('--t2', action='store', dest='t2', default='T2_preprocessed.nii.gz')
    parser.add_argument('--use-t1', action='store_true', dest='use_t1', default=True)
    parser.add_argument('--no-t1', action='store_false', dest='use_t1', default=True)
    parser.add_argument('--t1', action='store', dest='t1', default='T1_preprocessed.nii.gz')
    parser.add_argument('--mask', action='store', dest='mask', default='Consensus.nii.gz')
    parser.add_argument('--unet', action='store_const', const='unet', dest='select', default='unet')
    parser.add_argument('--encoder', action='store_const', const='encoder', dest='select', default='unet')
    parser.add_argument('--patches-seg', action='store_const', const='patches-seg', dest='select', default='unet')
    parser.add_argument('--patches-det', action='store_const', const='patches-det', dest='select', default='unet')

    args = parser.parse_args()

    selector = {
        'patches-seg': unet_patches3d_segmentation,
        'patches-det': unet_patches3d_detection,
        'encoder': autoencoder3d
    }

    selector[vars(args)['select']](vars(args))


def color_codes():
    codes = {'g': '\033[32m',
             'bg': '\033[32;1m',
             'b': '\033[1m',
             'nc': '\033[0m',
             'gc': '\033[32m, \033[0m'
             }
    return codes


def autoencoder3d(options):
    c = color_codes()
    image_sufix = get_sufix(
        options['use_flair'],
        options['use_pd'],
        options['use_t2'],
        options['use_gado'],
        options['use_t1']
    )

    print c['g'] + 'Loading the data for the ' + c['b'] + 'encoder' + c['nc']
    # Create the data
    encoder_data = load_encoder_data(
        test_size=options['test_size'],
        dir_name=options['folder'],
        use_flair=options['use_flair'],
        use_pd=options['use_pd'],
        use_t2=options['use_t2'],
        use_gado=options['use_gado'],
        use_t1=options['use_t1'],
        flair_name=options['flair'],
        pd_name=options['pd'],
        t2_name=options['t2'],
        gado_name=options['gado'],
        t1_name=options['t1'],
        min_shape=tuple(options['min_shape']) if options['min_shape'] is not None else None
    )
    (x_train, x_test, y_train, y_test, idx_train, idx_test) = encoder_data
    np.save(os.path.join(options['folder'], 'test_encoder.npy'), x_test)
    np.save(os.path.join(options['folder'], 'idx_test_encoder.npy'), idx_test)

    # Train the net and save it
    # net = create_encoder(x_train.shape, options.convo_size, options.pool_size, options.folder, options.number_filters)
    print c['g'] + 'Creating the ' + c['b'] + 'encoder' + c['nc']
    net = create_encoder3d_string(
        ''.join(options['layers']),
        x_train.shape,
        options['convo_size'],
        options['pool_size'],
        options['number_filters'],
        options['folder']
    )
    cPickle.dump(net, open(os.path.join(options['folder'], 'net.pkl'), 'wb'))

    print c['g'] + 'Training the ' + c['b'] + 'encoder' + c['nc']
    net.fit(x_train, y_train)

    # Load image names and test the net
    image_names = np.load(os.path.join(options['folder'], 'image_names_encoder.' + image_sufix + '.npy'))
    print c['g'] + 'Creating the encoded images' + c['nc']
    y_pred = net.predict(x_test)
    y = y_pred.reshape(x_test.shape)

    shape_str = c['nc'] + c['gc'].join([str(num) for num in y.shape]) + c['g']
    minmax_str = 'min = ' + c['nc'] + str(y.min()) + c['g'] + ', max = ' + c['nc'] + str(y.max()) + c['g']

    print c['bg'] + 'Shape' + c['nc'] + c['g'] + ' y: (' + shape_str + ')' + c['nc']
    print c['bg'] + 'Values' + c['nc'] + c['g'] + ' y (' + minmax_str + ')' + c['nc']

    np.save(options['folder'] + 'encoder_results.npy', y_pred.reshape(x_test.shape))

    images_names = [(y_im, image_names[:, idx]) for y_im, idx in zip(y, idx_test)]
    [reshape_save_nifti(im, name) for (ims, names) in images_names for (im, name) in zip(ims, names)]


def unet_patches3d_segmentation(options):
    c = color_codes()

    print c['g'] + 'Loading the data for the ' + c['b'] + 'unet CNN' + c['nc']
    # Create the data
    (x, y, names) = load_patches(
        dir_name=options['folder'],
        use_flair=options['use_flair'],
        use_pd=options['use_pd'],
        use_t2=options['use_t2'],
        use_gado=options['use_gado'],
        use_t1=options['use_t1'],
        flair_name=options['flair'],
        pd_name=options['pd'],
        t2_name=options['t2'],
        gado_name=options['gado'],
        t1_name=options['t1'],
        mask_name=options['mask'],
        size=tuple(options['patch_size'])
    )

    x_train = np.concatenate(x[:-1]).astype(dtype=np.float32)
    y_train = np.concatenate(y[:-1]).astype(dtype=np.int32)
    print 'Training vector shape = (' + ','.join([str(length) for length in x_train.shape]) + ')'
    print 'Training labels shape = (' + ','.join([str(length) for length in y_train.shape]) + ')'
    x_test = np.concatenate(x[-1:]).astype(dtype=np.float32)
    y_test = np.concatenate(y[-1:]).astype(dtype=np.int32)
    print 'Testing vector shape = (' + ','.join([str(length) for length in x_test.shape]) + ')'
    print 'Testing labels shape = (' + ','.join([str(length) for length in y_test.shape]) + ')'

    print c['g'] + 'Creating the ' + c['b'] + 'patch-based unet CNN' + c['nc']
    # Train the net and save it
    net = create_unet3d_string(
        ''.join(options['layers']),
        x_train.shape,
        options['convo_size'],
        options['pool_size'],
        options['number_filters'],
        options['folder']
    )
    # cPickle.dump(net, open(os.path.join(options['folder'], 'patches.pkl'), 'wb'))

    print c['g'] + 'Training the ' + c['b'] + 'patch-based unet CNN' + c['nc']
    net.fit(x_train, y_train.reshape([y_train.shape[0], -1]))

    print c['g'] + 'Computing the score' + c['nc']
    print net.score(x_test, y_test.reshape([y_test.shape[0], -1]))

    print c['g'] + 'Creating the test probability maps' + c['nc']
    y_pred = net.predict_proba(x_test)

    np.save(os.path.join(options['folder'], 'patches_results.npy'), y_pred)


def unet_patches3d_detection(options):
    c = color_codes()

    print c['g'] + 'Loading the data for the ' + c['b'] + 'unet CNN' + c['nc']
    # Create the data
    (x, y, names) = load_patches(
        dir_name=options['folder'],
        use_flair=options['use_flair'],
        use_pd=options['use_pd'],
        use_t2=options['use_t2'],
        use_gado=options['use_gado'],
        use_t1=options['use_t1'],
        flair_name=options['flair'],
        pd_name=options['pd'],
        t2_name=options['t2'],
        gado_name=options['gado'],
        t1_name=options['t1'],
        mask_name=options['mask'],
        size=tuple(options['patch_size'])
    )

    x_train = np.concatenate(x[:-1]).astype(dtype=np.float32)
    y_train = np.concatenate(y[:-1]).astype(dtype=np.int8)
    y_train = y_train[:, y_train.shape[1] / 2 + 1, y_train.shape[2] / 2 + 1, y_train.shape[3] / 2 + 1]
    print 'Training vector shape = (' + ','.join([str(length) for length in x_train.shape]) + ')'
    y_train_shape = ','.join([str(length) for length in y_train.shape])
    y_train_values = str(y_train.min()) + ',' + str(y_train.max())
    print 'Training labels shape = (' + y_train_shape + '); values = (' + y_train_values + ')'
    x_test = np.concatenate(x[-1:]).astype(dtype=np.float32)
    y_test = np.concatenate(y[-1:]).astype(dtype=np.int8)
    y_test = y_test[:, y_test.shape[1] / 2 + 1, y_test.shape[2] / 2 + 1, y_test.shape[3] / 2 + 1]
    print 'Testing vector shape = (' + ','.join([str(length) for length in x_test.shape]) + ')'
    print 'Testing labels shape = (' + ','.join([str(length) for length in y_test.shape]) + ')'

    print c['g'] + 'Creating the ' + c['b'] + 'patch-based unet CNN' + c['nc']
    # Train the net and save it
    net = create_patches3d_string(
        ''.join(options['layers']),
        x_train.shape,
        options['convo_size'],
        options['pool_size'],
        options['number_filters'],
        options['folder']
    )
    # cPickle.dump(net, open(os.path.join(options['folder'], 'patches.pkl'), 'wb'))

    print c['g'] + 'Training the ' + c['b'] + 'patch-based unet CNN' + c['nc']
    net.fit(x_train, y_train)

    print c['g'] + 'Computing the score' + c['nc']
    print net.score(x_test, y_test)

    print c['g'] + 'Creating the test probability maps' + c['nc']
    y_pred = net.predict_proba(x_test)

    np.save(os.path.join(options['folder'], 'patches_results.npy'), y_pred)


if __name__ == '__main__':
    main()
