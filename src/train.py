import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split

from data import Jaffe, CK
from model import CNN1, CNN2, CNN3, ResNet, VGG
from visualize import plot_loss, plot_acc

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="jaffe", help="dataset to train, jaffe or ck+")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--plot_history", type=bool, default=True)
opt = parser.parse_args()
his = None
print(opt)

if opt.dataset == "jaffe":
    expressions, x, y = Jaffe().gen_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 为了统一几个数据集，必须增加一列为0的
    y = np.hstack((y, np.zeros((y.shape[0], 1))))

    # 划分训练集验证集
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)
    print("load jaffe dataset successfully, it has {} train images and {} valid iamges".format(y_train.shape[0],
                                                                                               y_valid.shape[0]))
    train_generator = ImageDataGenerator(rotation_range=5,
                                         width_shift_range=0.01,
                                         height_shift_range=0.01,
                                         horizontal_flip=True,
                                         shear_range=0.1,
                                         zoom_range=0.1).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)

    model = VGG()

    sgd = Adam(lr=0.0001)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    callback = [ModelCheckpoint('../models/VGG_best_weights_jaffe.h5', monitor='val_accuracy', verbose=True,
                                save_best_only=True,
                                save_weights_only=True)]
    history_jaffe = model.fit(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                              validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                              callbacks=callback)
    his = history_jaffe


else:
    expr, x, y = CK().gen_train()
    y = to_categorical(y).reshape(y.shape[0], -1)
    # 划分训练集验证集
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)
    print("load CK+ dataset successfully, it has {} train images and {} valid iamges".format(y_train.shape[0],
                                                                                             y_valid.shape[0]))
    train_generator = ImageDataGenerator(rotation_range=10,
                                         width_shift_range=0.05,
                                         height_shift_range=0.05,
                                         horizontal_flip=True,
                                         shear_range=0.2,
                                         zoom_range=0.2).flow(x_train, y_train, batch_size=opt.batch_size)
    valid_generator = ImageDataGenerator().flow(x_valid, y_valid, batch_size=opt.batch_size)
    model = VGG()
    sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    callback = [
        ModelCheckpoint('../models/VGG_best_weights_CK+.h5', monitor='val_accuracy', verbose=True, save_best_only=True,
                        save_weights_only=True)]
    history_ck = model.fit_generator(train_generator, steps_per_epoch=len(y_train) // opt.batch_size, epochs=opt.epochs,
                                     validation_data=valid_generator, validation_steps=len(y_valid) // opt.batch_size,
                                     callbacks=callback)
    his = history_ck

if opt.plot_history:
    plot_loss(his.history, opt.dataset)
    plot_acc(his.history, opt.dataset)
