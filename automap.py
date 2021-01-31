import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from datetime import datetime
import os, sys, pickle
from data_provider import sinogram_generator, generate_radon, keras_sinogram_generator
import cv2, h5py



""""
Implementation of AUTOMAP, a deep-learning reconstruction idea for CT scanning
ref: <Image reconstruction by domain-transform manifold learning>

author: Chang Liu
"""

# easiest implementation for CUDA testing 
def create_easyNet(input_shape, nfilters=64):
    input = keras.layers.Input(shape=input_shape)
    conv1 = keras.layers.Conv2D(filters=nfilters,
                                kernel_size=5,
                                strides=(1, 1),
                                padding='same',
                                activation='relu')(input)
    conv2 = keras.layers.Conv2D(filters=nfilters,
                                kernel_size=5,
                                strides=(1, 1),
                                padding='same',
                                activation='relu')(conv1)
    model = keras.Model(inputs=input,
                        outputs=conv2)
    return model


# AUTOMAP implementation based on Keras.
def create_automap(input_shape, output_shape, nfilters=64, trans_n=2000,
                   FC_activation = 'tanh', conv_activation = 'relu', BN = True):
    # input_shape: the 3D tuple of input shape, e.g. (56, 180, 1)
    # output_shape: the 3D tuple of the output shape, e.g. (56, 56, 1)
    # nfilters: number of conv kernels in conv layers
    # trans_n: controlling the small-sized FC layer
    # FC_activation: the activation of FC layer, 'tanh' in the paper
    # conv_activation: the activation of conv layer, 'relu' in the paper
    # BN: True if batch normalization is to be used in the network

    input = keras.layers.Input(shape=input_shape)

    # Architecture details: input -> flatten -> FC0 -> FC1 -> FC2 -> reshape ->  conv1 -> conv2 -> Deconv -> output
    # n_prod = np.prod(input_shape[0]**2)
    n_prod = np.prod(input_shape)
    flatten = keras.layers.Flatten()(input)

    # Here replace the large (N^2 * 2N^2) FC with two FCs: (N^2 * trans_n), (trans_n, 2N^2),
    # so the model can fit in my PC ^^.
    # 1st FC layer with BN and activation
    if trans_n == 0:
        FC0_m = keras.layers.Dense(units=2 * n_prod)(flatten)
    else:
        FC0_0 = keras.layers.Dense(units=trans_n)(flatten)
        FC0_m = keras.layers.Dense(units= 2 * n_prod)(FC0_0)

    if BN is True:
        bn_0 = keras.layers.BatchNormalization()(FC0_m)
        FC0_1 = keras.layers.Activation(FC_activation)(bn_0)
    else:
        FC0_1 = keras.layers.Activation(FC_activation)(FC0_m)

    # 2nd FC layer with BN and activation
    if trans_n == 0:
        FC1_m = keras.layers.Dense(units=n_prod)(FC0_1)
    else:
        FC1_0 = keras.layers.Dense(units=trans_n)(FC0_1)
        FC1_m = keras.layers.Dense(units= n_prod)(FC1_0)
    if BN is True:
        bn_1 = keras.layers.BatchNormalization()(FC1_m)
        FC1_1 = keras.layers.Activation(FC_activation)(bn_1)
    else:
        FC1_1 = keras.layers.Activation(FC_activation)(FC1_m)

    # 3rd FC layer with BN and activation
    if trans_n == 0:
        FC2_m = keras.layers.Dense(units=input_shape[0] * input_shape[0])(FC1_1)
    else:
        FC2_0 = keras.layers.Dense(units=trans_n)(FC1_1)
        FC2_m = keras.layers.Dense(units= input_shape[0] * input_shape[0])(FC2_0)
    if BN is True:
        bn_2 = keras.layers.BatchNormalization()(FC2_m)
        FC2_1 = keras.layers.Activation(FC_activation)(bn_2)
    else:
        FC2_1 = keras.layers.Activation(FC_activation)(FC2_m)

    reshape = keras.layers.Reshape(output_shape)(FC2_1)

    conv1 = keras.layers.Conv2D(filters=nfilters,
                                kernel_size=5,
                                strides=(1, 1),
                                padding='same',
                                activation=conv_activation)(reshape)
    conv2 = keras.layers.Conv2D(filters=nfilters,
                                kernel_size=5,
                                strides=(1, 1),
                                padding='same',
                                activation=conv_activation)(conv1)
    # in the literature, the deconv layer uses 64 layers to generate 1 2d output, which is not sensible
    # here, the 'deconv' is replaced with 2 layers, conv3+output
    conv3 = keras.layers.Conv2D(filters=nfilters,
                                kernel_size=7,
                                strides=(1, 1),
                                padding='same',
                                activation=conv_activation,
                                kernel_regularizer=keras.regularizers.l1(0.00001))(conv2)
    output = keras.layers.Conv2D(filters=1,
                                 kernel_size=1,
                                 strides=(1, 1))(conv3)
    # final activation
    output = keras.layers.Activation(activation=conv_activation)(output)

    model = keras.Model(inputs=input,
                        outputs=output )
    return model


####################################################################################################
# Training strategy and callbacks

# custom callback
class custom_callback(keras.callbacks.Callback):
    def __init__(self, eval_x, eval_y, model_name):
        # init parent class
        super().__init__()

        # eval_imgs is used to test the reconstruction after each epoch. that is sinogram
        # eval_imgs.shape = [x, width, height, 1], x <= 10
        self.eval_imgs = eval_x
        self.eval_ori = eval_y
        self.log_dir = ".\\logs\\test_imgs\\" + model_name + "\\"
        self.model_cache_dir = ".\\models\\"+model_name+"\\"
        os.mkdir(self.log_dir)

    def save_recons(self, recons, idx):
        # combine recon and src images to one single image
        # format all images to [0, 255]int
        recons = np.uint8(recons / np.amax(recons) * 255)
        evals = np.uint8(self.eval_ori / np.amax(self.eval_ori) * 255)

        # combine and generate the saved image
        n_eval = self.eval_imgs.shape[0]
        shape = self.eval_ori.shape[1: 3]
        tar = np.zeros([2 * shape[0], n_eval * shape[1]], dtype=np.uint8)
        for i in range(n_eval):
            tar[0: shape[0], i * shape[1] : (i+1) * shape[1]] = evals[i, :, :, 0]
            tar[shape[0] : 2 * shape[0], i * shape[1] : (i+1) * shape[1]] = recons[i, :, :, 0]

        # save the image according to idx

        cv2.imwrite(self.log_dir + str(idx) + ".jpg", tar)


    def on_train_begin(self, logs=None):
        # reconstruct using the initialized automap
        recons = self.model.predict(self.eval_imgs)
        self.save_recons(recons, -1)

    def on_epoch_end(self, epoch, logs=None):
        # evaluate
        recons = self.model.predict(self.eval_imgs)
        self.save_recons(recons, epoch)

        # cache model after 15 epochs
        if epoch > 45:
            self.model.save(self.model_cache_dir + "epoch_" + str(epoch) + ".h5")



def train_automap(model, ds_gen, optimizer='adam', loss="mean_squared_error",
                  steps_per_epoch=500, epochs=10, validation_data=None):
    # bs stands for batch size
    # Compile keras model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    # model named after the time
    model_fname = "model_" + datetime.now().strftime("%d%m%y-%H-%M")

    # log for tensorboard visualization
    logdir = "logs\\scalars\\" + model_fname
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, update_freq=10, profile_batch=0)

    # Generate evaluation during the training
    sino, label = ds_gen.__getitem__(0)
    if sino.shape[0] > 5:
        eval_x = sino[:5, :, :, :]
        eval_y = label[:5, :, :, :]
    else:
        eval_x = sino
        eval_y = label

    eval_callback = custom_callback(eval_x, eval_y, model_fname)

    # create directory for models and caches
    os.mkdir(".\\models\\" + model_fname)
    model.fit(
        ds_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[tensorboard_callback, eval_callback],
        validation_data=validation_data
    )

    # save the model after the training

    model.save(".\\models\\"+model_fname+"\\final.h5")
    return model_fname

def automap_predict(img, model, size=56, interval=1):
    # encode the input img using radon transformation, and decode the image using automap
    # img.shape(56, 56), TODO: implementation for random image size
    # model: pretrained automap model

    # convert img to [0,1], and resize to (56, 56, 1)
    # img = img / np.amax(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    # radon transform and add dimensions
    sino = generate_radon(img, interval)
    sino = np.reshape(sino, (1, size, int(180/interval), 1)).astype(np.float32) / np.amax(sino)

    # reconstruct using automap and decrease dimension
    recon = model.predict(sino)
    recon = recon[0, :, :, 0]
    return recon


if __name__ == '__main__':
    # create data generator for training
    img_size = 56
    bs = 24
    ds_h5f = h5py.File('.\\data\\ct_ds.hdf5', 'r')
    train_ds = ds_h5f['train']
    test_ds = ds_h5f["test"]
    # sinoGen = sinogram_generator(ds=train_ds, bs=bs,
    #                              interval=90)
    sinoGen = keras_sinogram_generator(ds=train_ds, bs=bs,
                                 interval=45)

    model = create_automap(input_shape=(img_size, 4, 1),
                           output_shape=(img_size, img_size, 1),
                           nfilters=64,
                           FC_activation='relu')
    train_automap(model=model,
                  ds_gen=sinoGen,
                  steps_per_epoch=2300, epochs=20,
                  validation_data=None)



