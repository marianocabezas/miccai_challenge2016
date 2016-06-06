from optparse import OptionParser
import matplotlib
import numpy as np
import cPickle
from data_creation import load_encoder_data
from nets import create_encoder
matplotlib.use('Agg')


if __name__ == '__main__':
    # parse command line options
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
    (options, args) = parser.parse_args()

    encoder_data = load_encoder_data(test_size=options.test_size, dir_name=options.folder)
    (X_train, X_test, y_train, y_test, idx_train, idx_test) = encoder_data
    np.save(options.folder + 'test_encoder.npy', X_test)
    np.save(options.folder + 'idx_test_encoder.npy', idx_test)
    net = create_encoder(X_train.shape, options.convo_size, options.pool_size)
    cPickle.dump(net, open(options.folder + 'net.pkl', 'wb'))

    net.fit(X_train, y_train.astype(np.float32))

    # Load the best weights from pickled model
    net.load_params_from('./model_weights.pkl')
    