import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QHBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2

class HistogramWindow(QWidget):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Histogram")
        self.resize(800, 600)
        self.image = image
        
        self.canvas = FigureCanvas(Figure())
        
        self.histogram_label = QLabel("Histogram Eşitleme Değeri:")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(70)  # Default value
        self.slider.valueChanged.connect(self.update_equalization)

        self.equalize_button = QPushButton("Histogram Eşitle")
        self.equalize_button.clicked.connect(self.apply_equalization)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.canvas)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.histogram_label)
        slider_layout.addWidget(self.slider)
        
        self.layout.addLayout(slider_layout)
        self.layout.addWidget(self.equalize_button)
        self.layout.addWidget(self.image_label)

        # Yeni butonları ekleyelim
        self.gray_button = QPushButton("Gri Seviye Göster")
        self.gray_button.clicked.connect(self.show_gray_level)
        self.binary_button = QPushButton("Binary Seviye Göster")
        self.binary_button.clicked.connect(self.show_binary_level)
        self.rgb_button = QPushButton("RGB Seviye Göster")
        self.rgb_button.clicked.connect(self.show_rgb_level)

        level_button_layout = QHBoxLayout()
        level_button_layout.addWidget(self.gray_button)
        level_button_layout.addWidget(self.binary_button)
        level_button_layout.addWidget(self.rgb_button)

        self.layout.addLayout(level_button_layout)
        
        self.setLayout(self.layout)
        
        self.plot_histogram()
        
    def plot_histogram(self):
        self.canvas.figure.clf()  # Clear previous plots
        image = self.image
        
        if len(image.shape) == 3:
            # RGB image
            color = ('b', 'g', 'r')
            self.axes = self.canvas.figure.subplots(2, 1)  # Two plots: RGB and grayscale
            for i, col in enumerate(color):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                self.axes[0].plot(hist, color=col)
                self.axes[0].set_xlim([0, 256])
            
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            self.axes[1].plot(gray_hist, color='black')
            self.axes[1].set_xlim([0, 256])
            self.axes[1].set_title('Grayscale Histogram')
            self.canvas.draw()
        else:
            # Grayscale image
            self.axes = self.canvas.figure.add_subplot(111)
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            self.axes.plot(hist, color='black')
            self.axes.set_xlim([0, 256])
            self.canvas.draw()

    def update_equalization(self):
        self.slider_value = self.slider.value() / 100

    def apply_equalization(self):
        if self.image is None:
            QMessageBox.critical(self, "Hata", "Önce bir görüntü yükleyin.")
            return

        if len(self.image.shape) == 3:
            ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            equalized_image = cv2.equalizeHist(self.image)

        self.display_image(equalized_image)
        self.image = equalized_image
        self.plot_histogram()

    def display_image(self, image):
        if len(image.shape) == 2:  # Grayscale image
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def show_gray_level(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.display_image(gray_image)

    def show_binary_level(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        self.display_image(binary_image)

    def show_rgb_level(self):
        self.display_image(self.image)
