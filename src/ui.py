import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
from src.recognition import predict_expression, face_detect

sys.path.append('/')

matplotlib.use("Qt5Agg")


class UI(object):
    def __init__(self, form, model):
        self.setup_ui(form)
        self.model = model

    def setup_ui(self, form):
        form.setObjectName("Form")
        form.resize(1200, 580)

        self.label_raw_pic = QtWidgets.QLabel(form)
        self.label_raw_pic.setGeometry(QtCore.QRect(10, 30, 600, 450))  # 图片选择框
        self.label_raw_pic.setStyleSheet("background-color:#bbbbbb;")
        self.label_raw_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.label_raw_pic.setObjectName("label_raw_pic")

        self.line1 = QtWidgets.QFrame(form)
        self.line1.setGeometry(QtCore.QRect(620, 20, 20, 480))  # 竖线
        self.line1.setFrameShape(QtWidgets.QFrame.VLine)
        self.line1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line1.setObjectName("line1")

        self.layout_widget = QtWidgets.QWidget(form)
        self.layout_widget.setGeometry(QtCore.QRect(140, 400, 320, 240))
        self.layout_widget.setObjectName("layoutWidget")
        self.vertical_layout = QtWidgets.QVBoxLayout(self.layout_widget)
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setObjectName("verticalLayout")

        self.horizontal_layout = QtWidgets.QHBoxLayout()
        self.horizontal_layout.setObjectName("horizontalLayout")
        self.pushButton_select_img = QtWidgets.QPushButton(self.layout_widget)
        self.pushButton_select_img.setObjectName("pushButton_2")
        self.horizontal_layout.addWidget(self.pushButton_select_img)
        self.vertical_layout.addLayout(self.horizontal_layout)

        self.label_result = QtWidgets.QLabel(form)
        self.label_result.setGeometry(QtCore.QRect(361, 21, 71, 16))  # 未知
        self.label_result.setObjectName("label_result")

        self.label_emotion = QtWidgets.QLabel(form)
        self.label_emotion.setGeometry(QtCore.QRect(730, 30, 300, 50))  # 表情包上面的单词
        self.label_emotion.setObjectName("label_emotion")
        self.label_emotion.setAlignment(QtCore.Qt.AlignCenter)
        # 设置字体大小
        font = QFont()
        font.setPointSize(26)  # 设置字体大小为14
        self.label_emotion.setFont(font)

        self.line = QtWidgets.QFrame(form)
        self.line.setGeometry(QtCore.QRect(640, 480, 540, 16))  # 右横线
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")

        self.label_rst = QtWidgets.QLabel(form)
        self.label_rst.setGeometry(QtCore.QRect(730, 100, 360, 360))  # 表情包
        self.label_rst.setAlignment(QtCore.Qt.AlignCenter)
        self.label_rst.setObjectName("label_rst")

        self.pushButton_select_img.clicked.connect(self.open_file_browser)
        self.retranslate_ui(form)
        QtCore.QMetaObject.connectSlotsByName(form)

    def retranslate_ui(self, form):
        _translate = QtCore.QCoreApplication.translate
        form.setWindowTitle(_translate("Form", "山东理工大学 凌宇航-表情识别系统"))
        # 设置字体大小
        font = QFont()
        font.setPointSize(26)  # 设置字体大小为14
        self.label_raw_pic.setFont(font)  # 设置QLabel的字体
        self.label_raw_pic.setText(_translate("Form", "识别图片"))
        # 设置字体大小
        font = QFont()
        font.setPointSize(20)  # 设置字体大小为14（您可以根据需要调整大小）
        self.pushButton_select_img.setFont(font)  # 设置QPushButton的字体
        self.pushButton_select_img.setText(_translate("Form", "选择图像"))
        form.repaint()

    def open_file_browser(self):
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileName(caption="选取图片", directory="../input/test/",
                                                                     filter="All Files (*);;Text Files (*.txt)")
        if file_name is not None and file_name != "":
            self.show_raw_img(file_name)
            emotion, possibility = predict_expression(file_name, self.model)
            self.show_results(emotion, possibility)

    def show_raw_img(self, filename):
        img = cv2.imread(filename)
        frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (600, 450))
        self.label_raw_pic.setPixmap(QtGui.QPixmap.fromImage(
            QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                         QtGui.QImage.Format_RGB888)))

    def show_results(self, emotion, possibility):
        self.label_emotion.setText(QtCore.QCoreApplication.translate("Form", emotion))
        if emotion != 'no':
            img = cv2.imread('../assets/icons/' + str(emotion) + '.png')
            frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (360, 360))
            self.label_rst.setPixmap(QtGui.QPixmap.fromImage(
                QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], 3 * frame.shape[1],
                             QtGui.QImage.Format_RGB888)))
        else:
            self.label_rst.setText(QtCore.QCoreApplication.translate("Form", "no result"))

    def get_faces_from_image(self, img_path):
        img, img_gray, faces = face_detect(img_path, 'blazeface')
        if len(faces) == 0:
            return None
        faces_gray = []
        for (x, y, w, h) in faces:
            face_img_gray = img_gray[y:y + h + 10, x:x + w + 10]
            face_img_gray = cv2.resize(face_img_gray, (48, 48))
            faces_gray.append(face_img_gray)
        return faces_gray


class MyFigureCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.axes = fig.add_subplot(111)

    def draw_(self, possibility):
        x = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral', 'contempt']
        self.axes.bar(x, possibility, align='center')
