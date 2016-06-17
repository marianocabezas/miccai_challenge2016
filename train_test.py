import os
from optparse import OptionParser
import numpy as np
import cPickle
from data_creation import load_unet_data, load_encoder_data, load_patches
from data_creation import reshape_save_nifti_to_dir, reshape_save_nifti
from data_creation import get_sufix
from nets import create_unet3D_string, create_encoder3D_string, create_patches3D_string


def main():
    # Parse command line options
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option('-f', '--folder', dest='folder',
                      default='/home/mariano/DATA/Challenge/',
                      help="read data from FOLDER")
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose', default=False)
    parser.add_option('-c', '--convolution-size',
                      action='store', dest='convo_size', type='int', nargs=1, default=3)
    parser.add_option('-p', '--pool-size',
                      action='store', dest='pool_size', type='int', nargs=1, default=2)
    parser.add_option('-t', '--test-size',
                      action='store', dest='test_size', type='float', nargs=1, default=0.25)
    parser.add_option('-n', '--number-filters',
                      action='store', dest='number_filters', type='int', nargs=1, default=4)
    parser.add_option('-l', '--forward-layers',
                      action='store', dest='layers', type='string', nargs=1, default='cac')
    parser.add_option('-w', '--window-size',
                      action='store', dest='patch_size', type='int', nargs=3, default=(15, 15, 15))
    parser.add_option('-i', '--image-size',
                      action='store', dest='min_shape', type='int', nargs=3, default=None)
    parser.add_option('--use-gado',
                      action='store_true', dest='use_gado')
    parser.add_option('--no-gado',
                      action='store_false', dest='use_gado', default=False)
    parser.add_option('--gado',
                      action='store', dest='gado', type='string', default='GADO_preprocessed.nii.gz')
    parser.add_option('--use-flair',
                      action='store_true', dest='use_flair')
    parser.add_option('--no-flair',
                      action='store_false', dest='use_flair', default=True)
    parser.add_option('--flair',
                      action='store', dest='flair', type='string', default='FLAIR_preprocessed.nii.gz')
    parser.add_option('--use-pd',
                      action='store_true', dest='use_pd')
    parser.add_option('--no-pd',
                      action='store_false', dest='use_pd', default=True)
    parser.add_option('--pd',
                      action='store', dest='pd', type='string', default='DP_preprocessed.nii.gz')
    parser.add_option('--use-t2',
                      action='store_true', dest='use_t2')
    parser.add_option('--no-t2',
                      action='store_false', dest='use_t2', default=True)
    parser.add_option('--t2',
                      action='store', dest='t2', type='string', default='T2_preprocessed.nii.gz')
    parser.add_option('--use-t1',
                      action='store_true', dest='use_t1')
    parser.add_option('--no-t1',
                      action='store_false', dest='use_t1', default=True)
    parser.add_option('--t1',
                      action='store', dest='t1', type='string', default='T1_preprocessed.nii.gz')
    parser.add_option('--mask',
                      action='store', dest='mask', type='string', default='Consensus.nii.gz')
    parser.add_option('--unet',
                      action='store_const', const='unet', dest='select')
    parser.add_option('--encoder',
                      action='store_const', const='encoder', dest='select')
    parser.add_option('--patches',
                      action='store_const', const='patches', dest='select', default='unet')

    (options, args) = parser.parse_args()

    selector = {
        'unet': unet3d,
        'patches': unet_patches3d,
        'encoder': autoencoder3d
    }

    selector[options.select](options)


def autoencoder3d(options):
    c = color_codes()

    print c['g'] + 'Loading the data for the ' + c['b'] + 'encoder' + c['nc']
    image_sufix = get_sufix(options.use_flair, options.use_pd, options.use_t2, options.use_gado, options.use_t1)
    # Create the data
    encoder_data = load_encoder_data(
        test_size=options.test_size,
        dir_name=options.folder,
        use_flair=options.use_flair,
        use_pd=options.use_pd,
        use_t2=options.use_t2,
        use_gado=options.use_gado,
        use_t1=options.use_t1,
        flair_name=options.flair,
        pd_name=options.pd,
        t2_name=options.t2,
        gado_name=options.gado,
        t1_name=options.t1,
        min_shape=options.min_shape
    )
    (x_train, x_test, y_train, y_test, idx_train, idx_test) = encoder_data
    np.save(os.path.join(options.folder, 'test_encoder.npy'), x_test)
    np.save(os.path.join(options.folder, 'idx_test_encoder.npy'), idx_test)

    # Train the net and save it
    # net = create_encoder(x_train.shape, options.convo_size, options.pool_size, options.folder, options.number_filters)
    print c['g'] + 'Creating the ' + c['b'] + 'encoder' + c['nc']
    net = create_encoder3D_string(
        options.layers,
        x_train.shape,
        options.convo_size,
        options.pool_size,
        options.number_filters,
        options.folder
    )
    cPickle.dump(net, open(os.path.join(options.folder, 'net.pkl'), 'wb'))

    print c['g'] + 'Training the ' + c['b'] + 'encoder' + c['nc']
    net.fit(x_train, y_train)

    # Load image names and test the net
    image_names = np.load(os.path.join(options.folder, 'image_names_encoder.' + image_sufix + '.npy'))
    print c['g'] + 'Creating the encoded images' + c['nc']
    y_pred = net.predict(x_test)
    y = y_pred.reshape(x_test.shape)

    shape_str = c['nc'] + c['gc'].join([str(num) for num in y.shape]) + c['g']
    minmax_str = 'min = ' + c['nc'] + str(y.min()) + c['g'] + ', max = ' + c['nc'] + str(y.max()) + c['g']

    print c['bg'] + 'Shape' + c['nc'] + c['g'] + ' y: (' + shape_str + ')' + c['nc']
    print c['bg'] + 'Values' + c['nc'] + c['g'] + ' y (' + minmax_str + ')' + c['nc']

    np.save(options.folder + 'encoder_results.npy', y_pred.reshape(x_test.shape))

    images_names = [(y_im, image_names[:, idx]) for y_im, idx in zip(y, idx_test)]
    [reshape_save_nifti(im, name) for (ims, names) in images_names for (im, name) in zip(ims, names)]


def color_codes():
    codes = {'g': '\033[32m',
            'bg': '\033[32;1m',
            'b': '\033[1m',
            'nc': '\033[0m',
            'gc': '\033[32m, \033[0m'
            }
    return codes

def unet3d(options):
    c = color_codes()

    print c['g'] + 'Loading the data for the ' + c['b'] + 'patch-based unet CNN' + c['nc']
    image_sufix = get_sufix(options.use_flair, options.use_pd, options.use_t2, options.use_gado, options.use_t1)
    # Create the data
    unet_data = load_unet_data(
        test_size=options.test_size,
        dir_name=options.folder,
        use_flair=options.use_flair,
        use_pd=options.use_pd,
        use_t2=options.use_t2,
        use_gado=options.use_gado,
        use_t1=options.use_t1,
        flair_name=options.flair,
        pd_name=options.pd,
        t2_name=options.t2,
        gado_name=options.gado,
        t1_name=options.t1,
        mask_name=options.mask
    )
    (x_train, x_test, y_train, y_test, idx_train, idx_test) = unet_data
    np.save(os.path.join(options.folder, 'test_unet.npy'), x_test)
    np.save(os.path.join(options.folder, 'idx_test_unet.npy'), idx_test)

    print c['b'] + 'Creating the ' + c['b'] + 'unet CNN' + c['nc']
    # Train the net and save it
    net = create_unet3D_string(
        options.layers,
        x_train.shape,
        options.convo_size,
        options.pool_size,
        options.number_filters,
        options.folder
    )
    cPickle.dump(net, open(os.path.join(options.folder, 'unet.pkl'), 'wb'))

    print c['g'] + 'Training the ' + c['b'] + 'unet CNN' + c['nc']
    net.fit(x_train, y_train)

    # Load image names and test the net
    image_names = np.load(os.path.join(options.folder, 'image_names_unet.' + image_sufix + '.npy'))

    print c['g'] + 'Creating the test probability maps' + c['nc']
    y_pred = net.predict_proba(x_test)
    y = y_pred.reshape(x_test[1, 1, :].shape)

    shape_str = c['nc'] + c['gc'].join([str(num) for num in y.shape]) + c['g']
    minmax_str = 'min = ' + c['nc'] + str(y.min()) + c['g'] + ', max = ' + c['nc'] + str(y.max()) + c['g']

    print c['bg'] + 'Shape' + c['nc'] + c['g'] + ' y: (' + shape_str + ')' + c['nc']
    print c['bg'] + 'Values' + c['nc'] + c['g'] + ' y (' + minmax_str + ')' + c['nc']

    np.save(os.path.join(options.folder, 'unet_results.npy'), y)

    images_names = [(y_im, image_names[1, idx]) for y_im, idx in zip(y, idx_test)]
    [reshape_save_nifti_to_dir(im, name) for (im, name) in images_names]


def unet_patches3d(options):
    c = color_codes()

    print c['g'] + 'Loading the data for the ' + c['b'] + 'unet CNN' + c['nc']
    # Create the data
    (x, y, names) = load_patches(
        dir_name=options.folder,
        use_flair=options.use_flair,
        use_pd=options.use_pd,
        use_t2=options.use_t2,
        use_gado=options.use_gado,
        use_t1=options.use_t1,
        flair_name=options.flair,
        pd_name=options.pd,
        t2_name=options.t2,
        gado_name=options.gado,
        t1_name=options.t1,
        mask_name=options.mask,
        size=options.patch_size
    )

    x_train = np.concatenate(x[:-1])
    y_train = np.concatenate(y[:-1])
    x_test = np.concatenate(x[-1:])
    y_test = np.concatenate(y[-1:])

    print c['g'] + 'Creating the ' + c['b'] + 'patch-based unet CNN' + c['nc']
    # Train the net and save it
    net = create_patches3D_string(
        options.layers,
        x_train.shape,
        options.convo_size,
        options.pool_size,
        options.number_filters,
        options.folder
    )
    cPickle.dump(net, open(os.path.join(options.folder, 'patches.pkl'), 'wb'))

    print c['g'] + 'Training the ' + c['b'] + 'patch-based unet CNN' + c['b']
    net.fit(x_train, y_train)

    print c['g'] + 'Creating the test probability maps' + c['b']
    y_pred = net.predict_proba(x_test)

    np.save(os.path.join(options.folder, 'patches_results.npy'), y_pred)

    net.score(x_test, y_test)


if __name__ == '__main__':
    main()


