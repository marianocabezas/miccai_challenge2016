from optparse import OptionParser
import matplotlib
import numpy as np
import cPickle
from data_creation import load_encoder_data, reshape_save_nifti
from nets import create_encoder
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
                      action='store', dest='convo_size', type='int', nargs=1, default=15)
    parser.add_option('-p', '--pool-size',
                      action='store', dest='pool_size', type='int', nargs=1, default=2)
    parser.add_option('-t', '--test-size',
                      action='store', dest='test_size', type='float', nargs=1, default=0.25)
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

    # Create the data
    encoder_data = load_encoder_data(
        test_size=options.test_size,
        dir_name=options.folder,
        use_flair=options.use_flair,
        use_pd=options.use_pd,
        use_t2=options.use_t2,
        use_gado=options.use_gado,
        use_t1=options.use_t1
    )
    (x_train, x_test, y_train, y_test, idx_train, idx_test) = encoder_data
    np.save(options.folder + 'test_encoder.npy', x_test)
    np.save(options.folder + 'idx_test_encoder.npy', idx_test)

    # Train the net and save it
    net = create_encoder(x_train.shape, options.convo_size, options.pool_size, options.folder)
    cPickle.dump(net, open(options.folder + 'net.pkl', 'wb'))
    net.fit(x_train, y_train)

    # Load image names and test the net
    image_names = np.load(options.folder + 'image_names_encoder.npy')
    y_pred = net.predict(x_test)

    print 'Values y_pred (min = %d, max = %d)' % (y_pred.min(), y_pred.max())

    y = y_pred.reshape(x_test.shape)
    np.save(options.folder + 'encoder_results.npy', y_pred.reshape(x_test.shape))

    images_names = [(y_im, image_names[:, idx]) for y_im, idx in zip(y, idx_test)]
    niftis = [reshape_save_nifti(im, name) for (ims, names) in images_names for (im, name) in zip(ims, names)]