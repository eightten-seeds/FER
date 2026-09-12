import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_file(filename):
    data_file = open(filename, 'rb')
    data = pickle.load(data_file)
    data_file.close()
    return data.history


def plot_loss(his, ds):
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his['loss'])), his['loss'], label='train loss')
    plt.plot(np.arange(len(his['val_loss'])), his['val_loss'], label='valid loss')
    plt.title(ds + ' CNN3 training loss')
    plt.legend(loc='best')
    plt.savefig('/code/Recognition/assets/CNN3_loss_fer2013.png')


def plot_acc(his, ds):
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his['accuracy'])), his['accuracy'], label='train accuracy')
    plt.plot(np.arange(len(his['val_accuracy'])), his['val_accuracy'], label='valid accuracy')
    plt.title(ds + ' CNN3 training accuracy')
    plt.legend(loc='best')
    plt.savefig('/code/Recognition/assets/CNN3_acc_fer2013.png')


if __name__ == '__main__':
    history = load_file('../train_results/his_cnn2.pkl')
    print(np.max(history['val_acc']))
    plot_loss(history, "fer")
    plot_acc(history, "fer")
