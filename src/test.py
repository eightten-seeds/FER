import numpy as np
import matplotlib.pyplot as plt
from model import CNN3
from data import Jaffe, CK


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """
    import cv2
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img[:, :])
    resized_images.append(face_img[2:45, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))
    resized_images.append(face_img[0:45, 0:45])
    resized_images.append(face_img[2:47, 0:45])
    resized_images.append(face_img[2:47, 2:47])

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_data_gen():
    """
    有增广预测并计算准确率和损失
    :return:
    """
    model = CNN3()
    model.load_weights('../models/CNN3_best_weights_jaffe.h5')

    # Compile the model before evaluating
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    expression, x_test, y_test = Jaffe().gen_train()
    x_test = np.squeeze(x_test, axis=-1)

    # 计算损失和准确率
    pred = []
    faces = []
    accuracies = []
    losses = []
    for index in range(x_test.shape[0]):
        augmented_faces = generate_faces(x_test[index])
        faces.append(augmented_faces)
        results = model.predict(augmented_faces)
        result_sum = np.sum(results, axis=0)
        label_index = np.argmax(result_sum, axis=0)
        pred.append(label_index)

    # 转换为 NumPy 数组
    pred = np.array(pred)

    # 计算准确率
    accuracy = np.sum(pred == y_test) / len(y_test)
    accuracies.append(accuracy)
    print(f"数据增强测试精度: {accuracy:.4f}")

    # 计算损失
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    losses.append(loss)
    print(f"数据增强测试损失: {loss:.4f}")


def predict_no_gen():
    """
    无增广预测并计算准确率和损失
    :return:
    """
    model = CNN3()
    model.load_weights('../models/CNN3_best_weights_jaffe.h5')

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    expression, x_test, y_test = Jaffe().gen_train()

    # 计算损失和准确率
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)

    accuracy = np.sum(pred == y_test) / len(y_test)
    print(f"原始数据测试精度: {accuracy:.4f}")

    # 计算损失
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"原始数据测试损失: {loss:.4f}")


if __name__ == '__main__':
    predict_data_gen()
    predict_no_gen()
