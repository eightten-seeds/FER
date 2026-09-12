import sys
import os
from PyQt5 import QtWidgets
from model import CNN3
from ui import UI

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append(os.path.dirname(__file__) + '/ui')


def load_cnn_model():
    model = CNN3()
    model.load_weights('../models/cnn3_best_weights.h5')
    return model


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    model = load_cnn_model()
    ui = UI(form, model)
    form.show()
    sys.exit(app.exec_())
