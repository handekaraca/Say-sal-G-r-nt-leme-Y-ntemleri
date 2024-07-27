import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QToolButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QComboBox, QMessageBox, QPushButton, QSpacerItem, QSizePolicy, QScrollArea
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize 
import cv2
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
from histogram_equalization import histogram_equalization

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

        self.process_button = QToolButton()
        self.process_button.setIcon(QIcon(r"C:/Users/he_kr/Desktop/SayisalGoruntuIsleme_ProjeOdevi_190290012/assets/process_icon.png"))
        self.process_button.setText("İşlemi Uygula")
        self.process_button.setIconSize(QSize(48, 48))
        self.process_button.setStyleSheet("background-color: white; border-radius: 10px; border: 2px solid #0097B2;")
        self.process_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        process_font = self.process_button.font()
        process_font.setPointSize(12)
        self.process_button.setFont(process_font)

        self.download_button = QPushButton()
        self.download_button.clicked.connect(self.download_image)
        self.download_button.setText("Resmi İndir")
        self.download_button.setIconSize(QSize(48, 48))
        self.download_button.setIcon(QIcon(r"C:/Users/he_kr/Desktop/SayisalGoruntuIsleme_ProjeOdevi_190290012/assets/download_icon.png"))
        self.download_button.setStyleSheet("background-color: white; border-radius: 10px; border: 2px solid #0097B2;")
        download_font = self.download_button.font()
        download_font.setPointSize(12)
        self.download_button.setFont(download_font)
        self.download_button.setEnabled(False)  # Başlangıçta devre dışı bırakılır

        self.process_options = QComboBox()
        self.process_options.addItems(["Gri Tonlama", "Yeniden Boyutlandırma", "Kenar Tespiti", "Gürültü Azaltma", "Konstrat Arttırma", "Konstrat Azaltma", "Keskinleştirme", "Gauss Filtresi", "Sobel Filtresi", "Blurlaştırma", "Histogram Eşitleme"])

        layout = QHBoxLayout()
        layout.addWidget(self.select_button)
        layout.addWidget(self.process_button)

        option_layout = QVBoxLayout()
        option_layout.addWidget(self.process_options)
        options_font = self.process_options.font()
        options_font.setPointSize(12)
        self.process_options.setFont(options_font)
        self.process_options.setStyleSheet("background-color: white; border-radius: 10px; border: 2px solid #0097B2;")

        button_layout = QHBoxLayout()
        spacer_left = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        button_layout.addItem(spacer_left)
        button_layout.addWidget(self.download_button)
        spacer_right = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        button_layout.addItem(spacer_right)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(option_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_widget.setLayout(main_layout)
        scroll_area.setWidget(scroll_widget)
        self.setCentralWidget(scroll_area)

        self.image = None

    def load_image(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Görsel Seç", "", "Image Files (*.png *.jpg *.bmp)")
        if filepath:
            self.image = cv2.imread(filepath)
            if self.image is not None:
                self.display_image(self.image)
                self.process_button.setEnabled(True)
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

    def process_image(self):
        selected_option = self.process_options.currentText()
        global processed_image

        #Gri tonlama
        if selected_option == "Gri Tonlama":
            self.processed_image = grayscale_conversion(self.image) 
        
        #Yeniden boyutlandırma
        elif selected_option == "Yeniden Boyutlandırma":
            new_width = self.image.shape[1] // 2
            new_height = self.image.shape[0] // 2
            self.processed_image = resize_image(self.image, new_width, new_height)

        #Kenar tespiti
        elif selected_option == "Kenar Tespiti":
            self.processed_image = detect_edges(self.image, 100, 200)

        #Gürültü azaltma
        elif selected_option == "Gürültü Azaltma":
            self.processed_image = noise_reduction(self.image)

        #Konstrat arttırma
        elif selected_option == "Konstrat Arttırma":
            contrast_factor = 1.5  # kendi belirlediğimiz değer
            self.processed_image = increase_contrast(self.image, contrast_factor)

        #Konstrat azaltma
        elif selected_option == "Konstrat Azaltma":
            contrast_factor = 0.5  # kendi belirlediğimiz değer
            self.processed_image = decrease_contrast(self.image, contrast_factor)

        #Keskinleştirme
        elif selected_option == "Keskinleştirme":
            self.processed_image = sharpen_image(self.image)

        #Gauss Filtresi      
        elif selected_option == "Gauss Filtresi":
            kernel_size = 5  # çekirdek boyutu için kullanmak istediğiniz herhangi bir tek sayı
            sigma = 1.5  # kendi belirlediğimiz değer
            self.processed_image = apply_gaussian_filter(self.image, kernel_size, sigma)  

        #Sobel filtresi
        elif selected_option == "Sobel Filtresi":
            if len(self.image.shape) == 3:
                gray_image = grayscale_conversion(self.image)
            else:
                gray_image = self.image.copy()

            self.processed_image = apply_sobel_filter(gray_image) 

        #Blurlaştırma   
        elif selected_option == "Blurlaştırma":
            self.processed_image = apply_blur_filter(self.image)

        #Histogram Eğiştleme     
        elif selected_option == "Histogram Eşitleme":        
            if len(self.image.shape) == 3:
                gray_image = grayscale_conversion(self.image)
            else:
                gray_image = self.image.copy()

            processed_image = histogram_equalization(gray_image)   

        else:
            self.processed_image = self.image  

        self.display_image(self.processed_image)
        self.download_button.setEnabled(True)  # İşlem tamamlandığında indirme düğmesini etkinleştir

    def download_image(self):
        if self.process_image is not None:
            file_dialog = QFileDialog()
            filepath, _ = file_dialog.getSaveFileName(self, "Resmi Kaydet", "", "Image Files (*.png *.jpg *.bmp)")
            if filepath:
                cv2.imwrite(filepath, self.processed_image)  



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())
