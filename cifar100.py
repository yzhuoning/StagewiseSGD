import tarfile
from six.moves import urllib
import sys
import numpy as np
import pickle
import os

data_dir = 'cifar100_data'
full_data_dir = 'cifar100_data/cifar-100-python/train'
vali_dir = 'cifar100_data/cifar-100-python/test'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 100

NUM_TRAIN_BATCH = 1 # How many batches of files you want to read in, from 0 to 5)
EPOCH_SIZE = 50000 

def maybe_download_and_extract():
    '''
    Will download and extract the cifar10 data automatically
    :return: nothing
    '''
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def _read_one_batch(path):
    fo = open(path, 'rb')
    dicts = pickle.load(fo,  encoding='latin1')
    fo.close()

    data = dicts['data']
    data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    label = np.array(dicts['fine_labels'])
    return data, label

def read_in_all_images(address_list, shuffle=True):
    address = address_list[0]
    #print ('Reading images from ' + address)
    batch_data, batch_label = _read_one_batch(address)
        
    data = batch_data
    label = batch_label
    num_data = len(label)

    if shuffle is True:
        #print ('Shuffling')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label

def read_train_data():
    data, label = read_in_all_images([full_data_dir])
    return data, label

def read_test_data():
    data, label = read_in_all_images([vali_dir])
    return data, label

def load_data():
    all_data, all_labels  = read_train_data()
    test_data, test_labels = read_test_data()
    return (all_data, all_labels), (test_data, test_labels)


maybe_download_and_extract()