import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout, QLineEdit, QSpinBox, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np
from grayscale_conversion import grayscale_conversion
from resize_image import resize_image
from detect_edges import detect_edges
from noise_reduction import noise_reduction
from increase_contrast import increase_contrast
from decrease_contrast import decrease_contrast
from sharpen_image import sharpen_image
from gauss_filter import apply_gaussian_filter
from sobel_filter import apply_sobel_filter
from blur_image import apply_blur_filter
from opening import opening, closing, erode, dilate  # Yeni eklenen import

class FilterWindow(QWidget):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Filtreleme İşlemleri")
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
            "Gri Tonlama", "Yeniden Boyutlandırma", "Kenar Tespiti",
            "Gürültü Azaltma", "Konstrat Arttırma", "Konstrat Azaltma",
            "Keskinleştirme", "Gauss Filtresi", "Sobel Filtresi", "Blurlaştırma",
            "Opening", "Closing", "Erosion", "Dilation"
        ])
        self.process_options.setStyleSheet("background-color: white; border: 2px solid #0097B2; padding: 5px;")
        self.process_options.currentIndexChanged.connect(self.update_parameters)

        self.apply_button = QPushButton("Uygula")
        self.apply_button.setStyleSheet("background-color: #0097B2; color: white; border-radius: 10px; padding: 10px;")
        self.apply_button.clicked.connect(self.process_image)

        self.parameters_layout = QVBoxLayout()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Filtre Seçin:"))
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

        if selected_option == "Yeniden Boyutlandırma":
            self.width_input = QSpinBox()
            self.width_input.setRange(1, self.image.shape[1])
            self.width_input.setValue(self.image.shape[1] // 2)
            self.height_input = QSpinBox()
            self.height_input.setRange(1, self.image.shape[0])
            self.height_input.setValue(self.image.shape[0] // 2)
            self.parameters_layout.addWidget(QLabel("Yeni Genişlik:"))
            self.parameters_layout.addWidget(self.width_input)
            self.parameters_layout.addWidget(QLabel("Yeni Yükseklik:"))
            self.parameters_layout.addWidget(self.height_input)

        elif selected_option == "Kenar Tespiti":
            self.threshold1_input = QSpinBox()
            self.threshold1_input.setRange(0, 255)
            self.threshold1_input.setValue(100)
            self.threshold2_input = QSpinBox()
            self.threshold2_input.setRange(0, 255)
            self.threshold2_input.setValue(200)
            self.parameters_layout.addWidget(QLabel("Eşik 1:"))
            self.parameters_layout.addWidget(self.threshold1_input)
            self.parameters_layout.addWidget(QLabel("Eşik 2:"))
            self.parameters_layout.addWidget(self.threshold2_input)

        elif selected_option == "Konstrat Arttırma" or selected_option == "Konstrat Azaltma":
            self.contrast_factor_input = QLineEdit()
            self.contrast_factor_input.setText("1.5" if selected_option == "Konstrat Arttırma" else "0.5")
            self.parameters_layout.addWidget(QLabel("Kontrast Faktörü:"))
            self.parameters_layout.addWidget(self.contrast_factor_input)

        elif selected_option == "Gauss Filtresi":
            self.kernel_size_input = QSpinBox()
            self.kernel_size_input.setRange(1, 31)
            self.kernel_size_input.setValue(5)
            self.sigma_input = QLineEdit()
            self.sigma_input.setText("1.5")
            self.parameters_layout.addWidget(QLabel("Çekirdek Boyutu:"))
            self.parameters_layout.addWidget(self.kernel_size_input)
            self.parameters_layout.addWidget(QLabel("Sigma Değeri:"))
            self.parameters_layout.addWidget(self.sigma_input)
        
        elif selected_option in ["Opening", "Closing", "Erosion", "Dilation"]:
            self.kernel_size_input = QSpinBox()
            self.kernel_size_input.setRange(1, 31)
            self.kernel_size_input.setValue(3)
            self.parameters_layout.addWidget(QLabel("Çekirdek Boyutu:"))
            self.parameters_layout.addWidget(self.kernel_size_input)
        
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

        # Gri tonlama
        if selected_option == "Gri Tonlama":
            self.processed_image = grayscale_conversion(self.image) 
        
        # Yeniden boyutlandırma
        elif selected_option == "Yeniden Boyutlandırma":
            new_width = self.width_input.value()
            new_height = self.height_input.value()
            self.processed_image = resize_image(self.image, new_width, new_height)

        # Kenar tespiti
        elif selected_option == "Kenar Tespiti":
            threshold1 = self.threshold1_input.value()
            threshold2 = self.threshold2_input.value()
            self.processed_image = detect_edges(self.image, threshold1, threshold2)

        # Gürültü azaltma
        elif selected_option == "Gürültü Azaltma":
            self.processed_image = noise_reduction(self.image)

        # Kontrast arttırma
        elif selected_option == "Konstrat Arttırma":
            contrast_factor = float(self.contrast_factor_input.text())
            self.processed_image = increase_contrast(self.image, contrast_factor)

        # Kontrast azaltma
        elif selected_option == "Konstrat Azaltma":
            contrast_factor = float(self.contrast_factor_input.text())
            self.processed_image = decrease_contrast(self.image, contrast_factor)

        # Keskinleştirme
        elif selected_option == "Keskinleştirme":
            self.processed_image = sharpen_image(self.image)

        # Gauss Filtresi      
        elif selected_option == "Gauss Filtresi":
            kernel_size = self.kernel_size_input.value()
            sigma = float(self.sigma_input.text())
            self.processed_image = apply_gaussian_filter(self.image, kernel_size, sigma)  

        # Sobel filtresi
        elif selected_option == "Sobel Filtresi":
            if len(self.image.shape) == 3:
                gray_image = grayscale_conversion(self.image)
            else:
                gray_image = self.image.copy()

            self.processed_image = apply_sobel_filter(gray_image) 

        # Blurlaştırma   
        elif selected_option == "Blurlaştırma":
            self.processed_image = apply_blur_filter(self.image)

        # Opening
        elif selected_option == "Opening":
            kernel_size = self.kernel_size_input.value()
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.processed_image = opening(self.image, kernel)

        # Closing
        elif selected_option == "Closing":
            kernel_size = self.kernel_size_input.value()
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.processed_image = closing(self.image, kernel)

        # Erosion
        elif selected_option == "Erosion":
            kernel_size = self.kernel_size_input.value()
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.processed_image = erode(self.image, kernel)

        # Dilation
        elif selected_option == "Dilation":
            kernel_size = self.kernel_size_input.value()
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            self.processed_image = dilate(self.image, kernel)

        else:
            self.processed_image = self.image  

        self.display_image(self.processed_image)
