"""
Implementation of data provider for
"""

import gzip, pickle, glob
import numpy as np
import cv2
from skimage.transform import radon, iradon
from scipy.ndimage import zoom
from tensorflow.keras.utils import Sequence
from skimage import io

# one independent mnist package
import mnist, h5py

MNIST_IMG_SIZE = 28
# NUM_MNIST = 60000
NUM_MNIST = 6000


#####################################################################################################
## Processing of mnist dataset. The mnist dataset is processed and saved as pickle
def preprocess_ds(mnist_ds):
    # possible preprocessing of mnist dataset
    # mnist_ds.shape = [60000, 28, 28]

    # resize mnist to 64
    img_size = 64
    ds = np.ndarray([NUM_MNIST, img_size, img_size])
    for i in range(NUM_MNIST):
        ds[i, :, :] = cv2.resize(mnist_ds[i, :, :], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mnist_ds = ds
    return mnist_ds

def mnist_ds_generate(ds_path):
    # Parse the .gz mnist dataset
    f = gzip.open(ds_path, 'rb')

    # pop out the headlines
    f.read(16)

    # first, read the image to the buffer. Then parse the ds with prior knowledge of
    buf = f.read(MNIST_IMG_SIZE * MNIST_IMG_SIZE * NUM_MNIST)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(NUM_MNIST, MNIST_IMG_SIZE, MNIST_IMG_SIZE)
    f.close()

    # possible preprocessing of data
    data = preprocess_ds(data)

    # split data into training (50000) and testing (10000) set,
    train_ds = data[0:int(NUM_MNIST *0.8), :, :]
    test_ds = data[int(NUM_MNIST * 0.8):-1, :, :]

    # save dataset to pickle
    pickle.dump([train_ds, test_ds], open('.\\data\\mnist_ds.p', 'wb'))

    return data

def tiff_dataset_generate(ds_path):
    fnames = glob.glob(ds_path+'\*.tif')
    f = h5py.File('.\\data\\ct_ds.hdf5', 'w')
    tiff_ds = None
    tar_size = 64
    for fname in fnames:
        im = np.array(io.imread(fname))
        # resize the image to 64

        im_out = np.zeros([im.shape[0], tar_size, tar_size])
        for i in range(im.shape[0]):
            im_out[i, :, :] = cv2.resize(im[i, :, :], (tar_size, tar_size), interpolation=cv2.INTER_LINEAR)
        if fname == fnames[-1]:
            # last image for test
            tiff_ds_test = im_out
            break

        if tiff_ds is None:
            tiff_ds = im_out
            print(tiff_ds.shape)
        else:
            tiff_ds = np.concatenate((tiff_ds, im_out), axis=0)
            print(tiff_ds.shape)
    tiff_ds_train = tiff_ds[:-200]
    tiff_ds_veri = tiff_ds[-200:]
    f.create_dataset("train", data=tiff_ds_train)
    f.create_dataset("test",  data=tiff_ds_test)
    f.create_dataset('veri', data=tiff_ds_veri)
    f.close()
    print(tiff_ds_train.shape, tiff_ds_veri.shape, tiff_ds_test.shape)
    print(np.amax(tiff_ds_train), np.amax(tiff_ds_test))

#####################################################################################################
## Helper functions

# generates sinograms from image batches, with specific intervals
def generate_radon(img, interval):
    sino = radon(img, theta=np.arange(0, 180, interval))
    return sino

def mnist_ds_split(ds, label, ex_digit):
    # split mnist dataset by digits
    ds_ = ds[label!=ex_digit]
    ds_ex_ = ds[label==ex_digit]
    return ds_, ds_ex_

def get_verification_ds(ds_veri, interval):
    num_veri = ds_veri.shape[0]
    x_veri, y_veri = sinogram_generator(ds_veri, num_veri, interval).__next__()
    return (x_veri, y_veri)

#####################################################################################################
# data generators for keras model

class keras_sinogram_generator(Sequence):
    def __init__(self, ds, bs, interval):
        self.index = np.arange(ds.shape[0])
        self.batch_size = bs
        self.interval = interval   # interval of sinogram
        self.dataset = ds

        self.len_sino = int(180.0 / self.interval)
        self.img_size = ds.shape[1:]

    def __len__(self):
        return int(np.floor(len(self.index) / self.batch_size))

    def __getitem__(self, item):
        idx = self.index[item*self.batch_size : (item+1)*self.batch_size]
        sinos = np.zeros([self.batch_size, self.img_size[0], self.len_sino, 1])
        labels = np.zeros([self.batch_size, self.img_size[0], self.img_size[1], 1])
        for i in range(self.batch_size):
            # normalize all sinos and labels to (0, 1]
            sinos[i, :, :, 0] = generate_radon(self.dataset[idx[i], :, :], self.interval)
            # sinos[i, :, :, 0] = sinos[i, :, :, 0] / np.amax(sinos[i, :, :, 0])

            labels[i, :, :, 0] = self.dataset[idx[i], :, :]
            # labels[i, :, :, 0] = labels[i, :, :, 0] / np.amax(labels[i, :, :, 0])

        sinos = sinos / np.amax(sinos)
        labels = labels / np.amax(labels)

        return sinos, labels

    def on_epoch_end(self):
        # shuffle the index for the new batch
        np.random.shuffle(self.index)

# generate image and sinograms.
def sinogram_generator(ds, bs, interval, aug=None):
    # ds: dataset;                  bs: batch size
    # interval: sinogram interval;  aug: keras augmentator, not implemented
    ind = 0                                 # pointor through the whole ds
    len_ds = ds.shape[0]                    # length of ds, used to limit the pointor
    img_size = ds.shape[1:]
    len_sino = int(180.0 / interval)        # length of sinogram, incomplete sino is used in the reconstruction
    while True:
        sinos = np.zeros([bs, img_size[0], len_sino, 1])
        labels = np.zeros([bs, img_size[0], img_size[1], 1])
        for i in range(bs):
            sinos[i, :, :, 0] = generate_radon(ds[ind, :, :], interval)
            labels[i, :, :, 0] = ds[ind, :, :]
            ind = (ind + 1) % len_ds

        # normalization of sinos and labels
        sinos = sinos / np.amax(sinos)
        labels = labels / np.amax(labels)

        yield (sinos, labels)

# Dummy generator for CUDA testing
def dummy_generator(ds, bs):
    ind = 0                                 # pointor through the whole ds
    len_ds = ds.shape[0]                    # length of ds, used to limit the pointor
    img_size = ds.shape[1:]
    while True:
        imgs = np.zeros([bs, img_size[0], img_size[1], 1])
        for i in range(bs):
            imgs[i, :, :, 0] = ds[ind, :, :]
            ind = (ind + 1) % len_ds

        # normalization of sinos and labels
        imgs = imgs / np.amax(imgs)
        labels = imgs.copy()

        yield (imgs, labels)



#####################################################################################################
# Main function for testing
if __name__ == '__main__':
    # test to display one image from the mnist ds.
    # mnist_ds_generate('.\\data\\train-images-idx3-ubyte.gz')
    # ds, _ = pickle.load(open('.\\data\\mnist_ds.p', 'rb'))
    (x_train, y_train), (x_test, y_test) = mnist.parse_data('.\\data')
    

    # cv2.imshow('sample img', ds[0, :, :])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # test of generator
    # sinoGen = sinogram_generator(ds=ds, bs=2400,
    #                              interval=1)
    # while True:
    #     sinos, labels = next(sinoGen)
    #     k = ord('b')
    #     for i in range(24):
    #         cv2.imshow('generator image', labels[i, :, :, 0] / np.amax(labels))
    #         cv2.imshow('generator sino', sinos[i, :, :, 0] / np.amax(labels))
    #         k = cv2.waitKey(0)
    #         if k == ord('q'):
    #             break
    #     if k == ord('q'):
    #         break
    #
    # cv2.destroyAllWindows()

    # Test of dataset split
    # x_train = preprocess_mnist_ds(x_train)
    # # ds_train, ds_test = mnist_ds_split(x_train, y_train, 2)
    # ds_train = x_train
    # ds_test = x_test
    # ds_veri = ds_train[-200:]
    # ds_train = ds_train[:-200]
    # #
    # # # dump using h5py
    # f = h5py.File('.\\data\\mnist_ds_512.hdf5', 'w')
    # f.create_dataset("train", data=ds_train)
    # f.create_dataset("test",  data=ds_test)
    # f.create_dataset('veri', data=ds_veri)
    # f.close()
    #
    # print(ds_train.shape)
    # print(ds_test.shape)
    # max = np.amax(ds_train)
    # for i in range(10):
    #     cv2.imshow("ds_train", ds_train[i, :, :]/max)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # max = np.amax(ds_test)
    # for i in range(10):
    #     cv2.imshow("ds_test", ds_test[i, :, :]/max)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Test of verification dataset
    # ds_h5f = h5py.File('.\\data\\ct_ds.hdf5', 'r')
    # train_ds = ds_h5f['train']
    # veri_ds = train_ds[-200:]
    # train_ds = train_ds[:-200]
    # x_veri, y_veri = get_verification_ds(veri_ds, interval=1)
    # for i in range(24):
    #     cv2.imshow('generator image', y_veri[i, :, :, 0] / np.amax(y_veri))
    #     cv2.imshow('generator sino', x_veri[i, :, :, 0] / np.amax(x_veri))
    #     k = cv2.waitKey(0)
    #     if k == ord('q'):
    #         break

    # Test of generate tiff dataset
    # tiff_path = 'E:\\wTVprocessedData'
    # tiff_dataset_generate(tiff_path)

    # Test of ct data
    ds_h5f = h5py.File('.\\data\\ct_ds.hdf5', 'r')
    train_ds = ds_h5f['train']
    # x_veri, y_veri = get_verification_ds(train_ds[:24], interval=1)
    y_veri = train_ds
    diff = 0
    for i in range(24):
        print("image range: [", np.amin(y_veri[i, :, :]), ', ', np.amax(y_veri[i, :, :]))
        cv2.imshow('generator image', y_veri[i, :, :] / np.amax(y_veri))
        # cv2.imshow('generator sino', x_veri[i, :, :, 0] / np.amax(x_veri))
        # if i > 1:
        #     diff = np.sum(np.abs(y_veri[i, :, :] - y_veri[i-1, :, :]))

        k = cv2.waitKey(0)
        if k == ord('q'):
            break