import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QToolButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QTextEdit, QScrollArea, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize 
import cv2
import numpy as np

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sayısal Görüntü İşleme | Proje")
        self.resize(1200, 900)
        
        self.setStyleSheet("background-color: white;")

        self.setWindowIcon(QIcon(r"C:/Users/he_kr/Desktop/SayisalGoruntuIsleme_ProjeOdevi_190290012/assets/logo.png"))

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.select_button = QToolButton()
        self.select_button.setIcon(QIcon(r"C:/Users/he_kr/Desktop/SayisalGoruntuIsleme_ProjeOdevi_190290012/assets/upload_icon.png"))
        self.select_button.setText("Görsel Seç")
        self.select_button.setIconSize(QSize(48, 48))
        self.select_button.setStyleSheet("background-color: white; border-radius: 10px; border: 2px solid #0097B2;")
        self.select_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.select_button.clicked.connect(self.load_image)
        select_font = self.select_button.font()
        select_font.setPointSize(12)
        self.select_button.setFont(select_font)

        self.image_info_label = QLabel()
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setStyleSheet("background-color: white; border: 1px solid #0097B2; padding: 10px;")
        info_font = self.image_info_label.font()
        info_font.setPointSize(12)
        self.image_info_label.setFont(info_font)

        self.matrix_display = QTextEdit()
        self.matrix_display.setReadOnly(True)
        self.matrix_display.setStyleSheet("background-color: white; border: 1px solid #0097B2; padding: 10px;")
        matrix_font = self.matrix_display.font()
        matrix_font.setPointSize(10)
        self.matrix_display.setFont(matrix_font)

        self.matrix_properties_label = QLabel()
        self.matrix_properties_label.setAlignment(Qt.AlignCenter)
        self.matrix_properties_label.setStyleSheet("background-color: white; border: 1px solid #0097B2; padding: 10px;")
        properties_font = self.matrix_properties_label.font()
        properties_font.setPointSize(12)
        self.matrix_properties_label.setFont(properties_font)

        # Yeni butonları ekleyelim
        self.histogram_button = QPushButton("Histogram")
        self.histogram_button.setStyleSheet("background-color: white; border-radius: 10px; border: 2px solid #0097B2;")
        histogram_font = self.histogram_button.font()
        histogram_font.setPointSize(12)
        self.histogram_button.setFont(histogram_font)
        self.histogram_button.clicked.connect(self.open_histogram)

        self.filter_button = QPushButton("Filtreleme")
        self.filter_button.setStyleSheet("background-color: white; border-radius: 10px; border: 2px solid #0097B2;")
        filter_font = self.filter_button.font()
        filter_font.setPointSize(12)
        self.filter_button.setFont(filter_font)
        self.filter_button.clicked.connect(self.open_filter)

        self.segmentation_button = QPushButton("Segmentasyon")
        self.segmentation_button.setStyleSheet("background-color: white; border-radius: 10px; border: 2px solid #0097B2;")
        segmentation_font = self.segmentation_button.font()
        segmentation_font.setPointSize(12)
        self.segmentation_button.setFont(segmentation_font)
        self.segmentation_button.clicked.connect(self.open_segmentation)

        # Butonları layout'a ekleyelim
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.histogram_button)
        button_layout.addWidget(self.filter_button)
        button_layout.addWidget(self.segmentation_button)

        layout = QVBoxLayout()
        layout.addWidget(self.select_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.image_info_label)
        layout.addWidget(self.matrix_properties_label)
        layout.addWidget(self.matrix_display)
        layout.addLayout(button_layout)

        widget = QWidget()
        widget.setLayout(layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(widget)

        self.setCentralWidget(scroll_area)

        self.image = None

    def load_image(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Görsel Seç", "", "Image Files (*.png *.jpg *.bmp)")
        if filepath:
            self.image = cv2.imread(filepath)
            if self.image is not None:
                self.display_image(self.image)
                self.display_image_info(self.image)
                self.display_matrix(self.image)
                self.display_matrix_properties(self.image)
            else:
                QMessageBox.critical(self, "Hata", "Görüntü yüklenemedi.")

    def display_image(self, image):
        if len(image.shape) == 2:  # Grayscale image
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:  # Color image
            height, width, channel = image.shape
            bytes_per_line = channel * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)

        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def display_image_info(self, image):
        height, width, channels = image.shape if len(image.shape) == 3 else (image.shape[0], image.shape[1], 1)
        info_text = f"Görüntü Boyutları: {width} x {height}\nKanal Sayısı: {channels}"
        self.image_info_label.setText(info_text)

    def display_matrix(self, image):
        matrix_text = np.array2string(image, separator=', ')
        self.matrix_display.setText(matrix_text)

    def display_matrix_properties(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        
        mean = np.mean(gray_image)
        std_dev = np.std(gray_image)
        min_val = np.min(gray_image)
        max_val = np.max(gray_image)

        properties_text = f"Matris Özellikleri:\nOrtalama: {mean:.2f}\nStandart Sapma: {std_dev:.2f}\nMinimum Değer: {min_val}\nMaksimum Değer: {max_val}"
        self.matrix_properties_label.setText(properties_text)

    def open_histogram(self):
        import histogram
        self.histogram_window = histogram.HistogramWindow(self.image)
        self.histogram_window.show()

    def open_filter(self):
        import filter
        self.filter_window = filter.FilterWindow(self.image)
        self.filter_window.show()

    def open_segmentation(self):
        import segmentation
        self.segmentation_window = segmentation.SegmentationWindow(self.image)
        self.segmentation_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
