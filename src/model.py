from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, \
    AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import PReLU
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras import layers, models


def CNN1(input_shape=(48, 48, 1), n_classes=8):
    """
    参考VGG思路设计的第一个模型，主要注意点是感受野不能太大，以免获得很多噪声信息
    :param input_shape: 输入图片的尺寸
    :param n_classes: 目标类别数目
    :return:
    """
    # input
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # block2
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # block3
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # fc
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def CNN2(input_shape=(48, 48, 1), n_classes=8):
    """
    参考论文Going deeper with convolutions在输入层后加一层的1*1卷积增加非线性表示

    :param input_shape:
    :param n_classes:
    :return:
    """
    # input
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(32, (5, 5), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block3
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def CNN3(input_shape=(48, 48, 1), n_classes=8):
    """
    参考论文实现
    A Compact Deep Learning Model for Robust Facial Expression Recognition
    :param input_shape:
    :param n_classes:
    :return:
    """
    # input
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    # block1
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def ResNet(input_shape=(48, 48, 1), n_classes=8):
    """
    基于 ResNet50 构建的模型，适用于灰度图面部表情识别。
    :param input_shape: 输入图片的尺寸 (48, 48, 1) 表示灰度图
    :param n_classes: 目标类别数目
    :return:
    """
    # 输入层
    input_layer = layers.Input(shape=input_shape)

    # 第一层卷积和批量归一化
    x = layers.Conv2D(64, (3, 3), padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 生成ResNet的基本块
    def basic_block(x, filters, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (3, 3), strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Skip connection
        if stride != 1:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    # 构建ResNet
    x = basic_block(x, 64)
    x = basic_block(x, 128, stride=2)
    x = basic_block(x, 256, stride=2)
    x = basic_block(x, 512, stride=2)

    # 全局平均池化层
    x = layers.GlobalAveragePooling2D()(x)

    # 全连接层
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model


def VGG(input_shape=(48, 48, 1), n_classes=8):
    """
    VGG-like model for image classification.
    :param input_shape: Input image shape (height, width, channels), where channels can be 1 for grayscale
    :param n_classes: Number of output classes
    :return: VGG model
    """
    model = models.Sequential()

    # Adding layers to the model
    model.add(layers.InputLayer(input_shape=input_shape))  # Input layer

    # Define VGG-like architecture
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']  # VGG11 architecture
    in_channels = input_shape[2]  # Input channels
    for x in cfg:
        if x == 'M':  # 'M' represents max pooling
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
        else:
            model.add(layers.Conv2D(x, kernel_size=(3, 3), padding='same', activation='relu'))
            model.add(layers.BatchNormalization())
            in_channels = x

    # Global average pooling instead of flattening
    model.add(layers.GlobalAveragePooling2D())

    # Flattening and dropout
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))

    # Fully connected layer
    model.add(layers.Dense(1024, activation='relu'))

    # Output layer
    model.add(layers.Dense(n_classes, activation='softmax'))  # Softmax for multi-class classification

    return model
