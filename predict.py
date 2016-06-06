import numpy as np
from optparse import OptionParser
from data_creation import reshape_save_nifti
import cPickle


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
    idx_test = np.load(options.folder + 'idx_test_encoder.npy')
    image_names = np.load(options.folder + 'image_names_encoder.npy')

    net = cPickle.load(open(options.folder + 'net.pkl', 'rb'))
    net.load_params_from(options.folder + 'model_weights.pkl')
    y_pred = net.predict(X.astype(np.float32))
    print 'Values y_pred (min = %d, max = %d)' % (y_pred.min(), y_pred.max())

    y = y_pred.reshape(X.shape)
    np.save(options.folder + 'encoder_results.npy', y_pred.reshape(X.shape))

    images_names = [(y_im, image_names[:, idx]) for y_im, idx in zip(y, idx_test)]
    niftis = [reshape_save_nifti(im, name) for (ims, names) in images_names for (im, name) in zip(ims, names)]
