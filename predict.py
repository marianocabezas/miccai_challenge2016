import numpy as np
from optparse import OptionParser
from nets import create_encoder


if __name__ == '__main__':
    # parse command line options
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option('-f', '--folder', dest='folder',
                      default='/home/mariano/DATA/Challenge/',
                      help="read data from FOLDER")
    parser.add_option('-v', '--verbose',
                      action='store_true', dest='verbose', default=False)
    (options, args) = parser.parse_args()

    X = np.load(options.folder + 'test_encoder.npy')
    X_train = np.load(options.folder + 'train_encoder.npy')

    net = create_encoder(X_train.shape, 15, 2)
    net.load_params_from('./model_weights.pkl')
    y_pred = net.predict(X)

    y_pred.save(options.folder + 'test_encoder.npy')

