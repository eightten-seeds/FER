from keras.src.utils import load_img, img_to_array
from tqdm import tqdm
import os
import numpy as np


class Jaffe(object):
    """
    Jaffe没有测试数据，需要自己划分
    """

    def __init__(self):
        self.folder = '../dataset/jaffe'

    def gen_train(self):
        """
        产生训练数据
        注意产生的是(213, 48, 48, 1)和(213, )的x和y，如果输入灰度图需要将x的最后一维squeeze掉
        :return:
        """
        folder = os.path.join(self.folder, 'Training')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_train_no(self):
        """
        产生训练数据
        :return:
        """
        import cv2
        folder = os.path.join(self.folder, 'Training')
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'contempt':
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train)
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_data(self):
        """
        生成划分后的数据集，实际使用需要交叉验证
        :return:
        """
        _, x, y = self.gen_train()
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2019)
        return x_train, x_test, y_train, y_test


class CK(object):
    """
    CK+没有测试数据，需要自己划分
    """

    def __init__(self):
        self.folder = '../dataset/ck+'

    def gen_train(self):
        """
        产生训练数据
        :return:
        """
        folder = self.folder
        # 为了模型训练统一，这里加入neural
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'neutral':
                # 没有中性表情，直接跳过
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = load_img(os.path.join(expression_folder, images[j]), target_size=(48, 48), color_mode="grayscale")
                img = img_to_array(img)  # 灰度化
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train).astype('float32') / 255.
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_train_no(self):
        """
        产生训练数据
        :return:
        """
        import cv2
        folder = self.folder
        # 为了模型训练统一，这里加入neural
        expressions = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        x_train = []
        y_train = []
        for i in tqdm(range(len(expressions))):
            if expressions[i] == 'neutral':
                # 没有中性表情，直接跳过
                continue
            expression_folder = os.path.join(folder, expressions[i])
            images = os.listdir(expression_folder)
            for j in range(len(images)):
                img = cv2.imread(os.path.join(expression_folder, images[j]), cv2.IMREAD_GRAYSCALE)
                x_train.append(img)
                y_train.append(i)
        x_train = np.array(x_train)
        y_train = np.array(y_train).astype('int')
        return expressions, x_train, y_train

    def gen_data(self):
        """
        生成划分后的数据集，实际使用需要交叉验证
        :return:
        """
        _, x, y = self.gen_train()
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2019)
        return x_train, x_test, y_train, y_test
