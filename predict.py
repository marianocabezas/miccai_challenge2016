import numpy as np
from optparse import OptionParser
import cPickle as pickle


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
                      action='store', dest='test_size', type='float', nargs=1, default=2)
    (options, args) = parser.parse_args()

    X = np.load(options.folder + 'test_encoder.npy')

    net = pickle.load(open(options.folder + 'net.pkl', 'rb'))
    net.load_params_from('./model_weights.pkl')
    y_pred = net.predict(X)

    y_pred.save(options.folder + 'test_encoder.npy')

