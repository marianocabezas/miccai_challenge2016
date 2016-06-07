import os
from optparse import OptionParser
import matplotlib
import numpy as np
import cPickle
from data_creation import load_unet_data, reshape_save_nifti_to_dir
from nets import create_unet_string
matplotlib.use('Agg')


if __name__ == '__main__':
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
                      action='store', dest='number_filters', type='int', nargs=1, default=64)
    parser.add_option('-l', '--forward-layers',
                      action='store', dest='layers', type='string', nargs=1, default='cac')
    parser.add_option('-i', '--image-size',
                      action='store', dest='min_shape', type='int', nargs=3, default=None)
    parser.add_option('--use-gado',
                      action='store_true', dest='use_gado', default=False)
    parser.add_option('--use-flair',
                      action='store_true', dest='use_flair', default=True)
    parser.add_option('--use-pd',
                      action='store_true', dest='use_pd', default=False)
    parser.add_option('--use-t2',
                      action='store_true', dest='use_t2', default=False)
    parser.add_option('--use-t1',
                      action='store_true', dest='use_t1', default=False)

    (options, args) = parser.parse_args()

    images_used = [options.use_flair, options.use_pd, options.use_t2, options.use_gado, options.use_t1]
    letters = ['fl', 'pd', 't2', 'gd', 't1']
    image_sufix = '.'.join(
        [letter for (letter, is_used) in zip(letters, images_used) if is_used]
    )

    # Create the data
    encoder_data = load_unet_data(
        test_size=options.test_size,
        dir_name=options.folder,
        use_flair=options.use_flair,
        use_pd=options.use_pd,
        use_t2=options.use_t2,
        use_gado=options.use_gado,
        use_t1=options.use_t1
    )
    (x_train, x_test, y_train, y_test, idx_train, idx_test) = encoder_data
    np.save(os.path.join(options.folder, 'test_unet.npy'), x_test)
    np.save(os.path.join(options.folder, 'idx_test_unet.npy'), idx_test)

    # Train the net and save it
    net = create_unet_string(
        options.layers,
        x_train.shape,
        options.convo_size,
        options.pool_size,
        options.folder,
        options.number_filters
    )
    cPickle.dump(net, open(os.path.join(options.folder, 'unet.pkl'), 'wb'))
    net.fit(x_train, y_train)

    # Load image names and test the net
    image_names = np.load(os.path.join(options.folder, 'image_names_unet.' + image_sufix + '.npy'))
    y_pred = net.predict_proba(x_test)
    y = y_pred.reshape(x_test[1, 1, :].shape)

    print 'Shape y: (' + ','.join([str(num) for num in y.shape])
    print 'Values y (min = %d, max = %d)' % (y_pred.min(), y_pred.max())

    np.save(os.path.join(options.folder, 'unet_results.npy', y))

    images_names = [(y_im, image_names[1, idx]) for y_im, idx in zip(y, idx_test)]
    [reshape_save_nifti_to_dir(im, name) for (im, name) in images_names]
