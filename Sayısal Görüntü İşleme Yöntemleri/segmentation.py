import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout, QLineEdit, QSpinBox, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class SegmentationWindow(QWidget):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Segmentasyon İşlemleri")
        self.resize(800, 600)
        self.image = image
        self.processed_image = None
        
        self.initUI()
        self.display_image(self.image)
        
    def initUI(self):
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #0097B2; padding: 10px; background-color: white;")

        self.process_options = QComboBox()
        self.process_options.addItems([
            "Region Growing", "Active Contour"
        ])
        self.process_options.setStyleSheet("background-color: white; border: 2px solid #0097B2; padding: 5px;")
        self.process_options.currentIndexChanged.connect(self.update_parameters)

        self.apply_button = QPushButton("Uygula")
        self.apply_button.setStyleSheet("background-color: #0097B2; color: white; border-radius: 10px; padding: 10px;")
        self.apply_button.clicked.connect(self.process_image)

        self.parameters_layout = QVBoxLayout()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Segmentasyon Yöntemi Seçin:"))
        layout.addWidget(self.process_options)
        layout.addLayout(self.parameters_layout)
        layout.addWidget(self.apply_button)
        layout.addWidget(self.image_label)
        
        self.setLayout(layout)
        self.update_parameters()
        
    def update_parameters(self):
        for i in reversed(range(self.parameters_layout.count())): 
            self.parameters_layout.itemAt(i).widget().setParent(None)

        selected_option = self.process_options.currentText()

        if selected_option == "Region Growing":
            self.seed_point_x = QSpinBox()
            self.seed_point_x.setRange(0, self.image.shape[1] - 1)
            self.seed_point_y = QSpinBox()
            self.seed_point_y.setRange(0, self.image.shape[0] - 1)
            self.threshold_input = QLineEdit()
            self.threshold_input.setText("10")
            self.parameters_layout.addWidget(QLabel("Tohum Noktası X:"))
            self.parameters_layout.addWidget(self.seed_point_x)
            self.parameters_layout.addWidget(QLabel("Tohum Noktası Y:"))
            self.parameters_layout.addWidget(self.seed_point_y)
            self.parameters_layout.addWidget(QLabel("Eşik Değeri:"))
            self.parameters_layout.addWidget(self.threshold_input)

        elif selected_option == "Active Contour":
            self.alpha_input = QLineEdit()
            self.alpha_input.setText("0.1")
            self.beta_input = QLineEdit()
            self.beta_input.setText("0.1")
            self.gamma_input = QLineEdit()
            self.gamma_input.setText("0.1")
            self.parameters_layout.addWidget(QLabel("Alpha:"))
            self.parameters_layout.addWidget(self.alpha_input)
            self.parameters_layout.addWidget(QLabel("Beta:"))
            self.parameters_layout.addWidget(self.beta_input)
            self.parameters_layout.addWidget(QLabel("Gamma:"))
            self.parameters_layout.addWidget(self.gamma_input)

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
    
    def process_image(self):
        selected_option = self.process_options.currentText()

        # Region Growing
        if selected_option == "Region Growing":
            seed_point = (self.seed_point_y.value(), self.seed_point_x.value())
            threshold = int(self.threshold_input.text())
            self.processed_image = self.region_growing(self.image, seed_point, threshold)
        
        # Active Contour
        elif selected_option == "Active Contour":
            alpha = float(self.alpha_input.text())
            beta = float(self.beta_input.text())
            gamma = float(self.gamma_input.text())
            self.processed_image = self.active_contour(self.image, alpha, beta, gamma)

        self.display_image(self.processed_image)
    
    def region_growing(self, image, seed, threshold):
        if len(image.shape) == 3:  # Convert color image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        seed_value = image[seed]
        segmented = np.zeros_like(image)
        segmented[seed] = 255
        stack = [seed]

        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                    if segmented[nx, ny] == 0 and abs(int(image[nx, ny]) - int(seed_value)) < threshold:
                        segmented[nx, ny] = 255
                        stack.append((nx, ny))
        
        return segmented

    def active_contour(self, image, alpha, beta, gamma):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        contours = np.zeros_like(gray)
        contours[10:-10, 10:-10] = 1
        snake = cv2.GaussianBlur(contours.astype(np.float32), (15, 15), 0)
        cv2.normalize(snake, snake, 0, 1, cv2.NORM_MINMAX)

        for i in range(100):
            fx = cv2.Sobel(snake, cv2.CV_64F, 1, 0, ksize=5)
            fy = cv2.Sobel(snake, cv2.CV_64F, 0, 1, ksize=5)
            fxx = cv2.Sobel(fx, cv2.CV_64F, 1, 0, ksize=5)
            fyy = cv2.Sobel(fy, cv2.CV_64F, 0, 1, ksize=5)

            energy = alpha * (fxx + fyy) - beta * (fx**2 + fy**2) + gamma * gray
            cv2.normalize(energy, energy, 0, 1, cv2.NORM_MINMAX)
            snake = energy
        
        snake = (snake > 0.5).astype(np.uint8) * 255
        return snake


