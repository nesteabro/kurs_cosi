import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from scipy.ndimage import gaussian_filter, laplace
from skimage.util import random_noise
from skimage.segmentation import felzenszwalb
from skimage.filters import threshold_otsu
from skimage.feature import canny
from scipy.ndimage import gaussian_laplace

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None

    def initUI(self):
        layout = QVBoxLayout()
        
        self.label = QLabel("Загрузите изображение")
        layout.addWidget(self.label)
        
        self.loadButton = QPushButton("Загрузить изображение")
        self.loadButton.clicked.connect(self.load_image)
        layout.addWidget(self.loadButton)
        
        self.noiseButton = QPushButton("Добавить гауссов шум")
        self.noiseButton.clicked.connect(self.add_gaussian_noise)
        layout.addWidget(self.noiseButton)
        
        self.filterButton = QPushButton("Применить низкочастотный фильтр")
        self.filterButton.clicked.connect(self.apply_lowpass_filter)
        layout.addWidget(self.filterButton)
        
        self.edgeButton = QPushButton("Выделение границ (Марр-Хилдретт)")
        self.edgeButton.clicked.connect(self.apply_marr_hildreth)
        layout.addWidget(self.edgeButton)
        
        self.clusterButton = QPushButton("Сегментация (Эйквил)")
        self.clusterButton.clicked.connect(self.cluster_segmentation)
        layout.addWidget(self.clusterButton)
        
        self.sezanButton = QPushButton("Сегментация (Сезана)")
        self.sezanButton.clicked.connect(self.sezan_segmentation)
        layout.addWidget(self.sezanButton)
        
        self.setLayout(layout)
        self.setWindowTitle("Обработка цифровых изображений")
    
    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Выбрать изображение', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        if fname:
            self.image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)
    
    def display_image(self, img):
        height, width = img.shape
        bytes_per_line = width
        qimg = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.label.setPixmap(QPixmap.fromImage(qimg))
    
    def add_gaussian_noise(self):
        if self.image is not None:
            noisy_img = random_noise(self.image, mode='gaussian', var=0.01)
            self.image = (noisy_img * 255).astype(np.uint8)
            self.display_image(self.image)
    
    def apply_lowpass_filter(self):
        if self.image is not None:
            filtered_img = gaussian_filter(self.image, sigma=1)
            self.image = filtered_img.astype(np.uint8)
            self.display_image(self.image)
    
    def apply_marr_hildreth(self):
        if self.image is not None:
            log_img = gaussian_laplace(self.image, sigma=1)
            self.image = np.clip(log_img, 0, 255).astype(np.uint8)
            self.display_image(self.image)
    
    def cluster_segmentation(self):
        if self.image is not None:
            segmented_img = felzenszwalb(self.image, scale=100)
            self.image = ((segmented_img / segmented_img.max()) * 255).astype(np.uint8)
            self.display_image(self.image)
    
    def sezan_segmentation(self):
        if self.image is not None:
            hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
            thresh = np.argmax(hist)
            segmented_img = (self.image > thresh) * 255
            self.image = segmented_img.astype(np.uint8)
            self.display_image(self.image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())

