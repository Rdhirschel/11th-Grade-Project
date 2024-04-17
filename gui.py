from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QWidget, QPushButton, QMainWindow, QApplication
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
from DL3 import *
import numpy as np
from PIL import Image

# Parameters
IMAGE_SIZE = (32, 32)
classNames = ["dandelion", "iris", "rose", "sunflower", "tulip"]

class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()

        self.model = model

        self.setWindowTitle("Flower Classifier GUI")
        self.setGeometry(100, 100, 800, 600)  

        self.button = QPushButton("Open Image", self)
        self.button.setFont(QFont('Arial', 20)) 
        self.button.clicked.connect(self.open_image)

        self.label = QLabel(self)
        self.label.setFont(QFont('Arial', 20))  
        self.label.setAlignment(Qt.AlignCenter) 

        self.image_label = QLabel(self) 
        self.image_label.setAlignment(Qt.AlignCenter)  

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.label)
        self.layout.setSpacing(10)  
        self.layout.setContentsMargins(10, 10, 10, 10) 

        self.central_widget = QWidget(self)
        self.central_widget.setLayout(self.layout)

        self.setCentralWidget(self.central_widget)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image")

        if file_name:
            global classNames
            image = Image.open(file_name)
            pixmap = QPixmap(file_name)  # Create a QPixmap from the image file
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))  # Set the pixmap of the image label

            image = image.resize(IMAGE_SIZE)
            if (image.mode != "RGB"):
                image = image.convert("RGB")
            image = np.array(image).reshape(1, IMAGE_SIZE[0]*IMAGE_SIZE[1]*3) / 255.0 - 0.5
            image = image.T
            prediction = self.model.predict(image)
            class_name = classNames[np.argmax(prediction)]
            self.label.setText(f"The AI thinks this image is a {class_name}.")

model = DLModel()
model.load_weights("saved_weights 80.71%", ["relu", "trim_sigmoid", "trim_tanh", "trim_softmax"], "categorical_cross_entropy")
app = QApplication([])

window = MainWindow(model)
window.show()

app.exec_()