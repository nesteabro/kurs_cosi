#Загрузка библиотек
import sys
import numpy as np
import cv2
import random
from PyQt5.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout, QWidget,
                             QFileDialog, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
#Класс обработки изображений
class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Цифровая обработка изображений")
        self.image = None
        self.original_image = None

        self.image_label = QLabel("Изображение не загружено")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.load_btn = QPushButton("Загрузить изображение")
        self.noise_btn = QPushButton("Добавить гауссов шум")
        self.filter_btn = QPushButton("Низкочастотная фильтрация")
        self.marr_btn = QPushButton("Оператор Марр-Хилдретта")
        self.eykvil_btn = QPushButton("Сегментация Эйквила")
        self.sezan_btn = QPushButton("Сегментация Сезана")
        self.reset_btn = QPushButton("Сброс изображения")

        self.load_btn.clicked.connect(self.load_image)
        self.noise_btn.clicked.connect(self.add_gaussian_noise)
        self.filter_btn.clicked.connect(self.low_pass_filter)
        self.marr_btn.clicked.connect(self.marr_hildreth_edge)
        self.eykvil_btn.clicked.connect(self.eykvil_segmentation)
        self.sezan_btn.clicked.connect(self.sezan_segmentation)
        self.reset_btn.clicked.connect(self.reset_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.noise_btn)
        btn_layout.addWidget(self.filter_btn)
        btn_layout.addWidget(self.marr_btn)
        btn_layout.addWidget(self.eykvil_btn)
        btn_layout.addWidget(self.sezan_btn)
        btn_layout.addWidget(self.reset_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)
#Загрузка изображения и преобразование в полутоновое
    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Открыть файл', '.', 'Image files (*.png *.jpg *.bmp)')
        if fname:
            self.image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            self.original_image = self.image.copy()
            self.display_image(self.image)
#Отображение изображения
    def display_image(self, img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        h, w = img.shape
        q_img = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_label.size(), Qt.KeepAspectRatio))
#Гауссов шум
    def add_gaussian_noise(self, mean=0, sigma=20):
       
        noisy = self.image.copy().astype(np.float32)

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                noisy[i, j] += random.gauss(mean, sigma)

        self.image = np.clip(noisy, 0, 255).astype(np.uint8)
        self.display_image(self.image)
#Низкочастотный фильтр
    def low_pass_filter(self):
        if self.image is None:
            return
        kernel = np.ones((5, 5), dtype=np.float32) / 25
        self.image = self.convolve(self.image, kernel)
        self.display_image(self.image)
#Оператор Марр-Хилдрета
    def marr_hildreth_edge(self):
        if self.image is None:
            return
        blurred = self.convolve(self.image, self.gaussian_kernel(5, 1.0))
        log = self.laplacian_of_gaussian(blurred)
        edges = self.zero_crossing(log)
        self.image = edges * 255
        self.display_image(self.image)

#Сегментация Эйквила
    def eykvil_segmentation(self):
        if self.image is None:
            return

        img = self.image.copy()
        small_size = 8    # размер малого окна S
        large_size = 24   # размер большого окна L
        l_threshold = 15  # минимальная разница средних

        height, width = img.shape
        result = np.zeros_like(img)

        pad = large_size // 2
        padded = np.pad(img, pad, mode='edge')

        for y in range(0, height, small_size):
            for x in range(0, width, small_size):
                cy, cx = y + small_size // 2, x + small_size // 2
                cy_p, cx_p = cy + pad, cx + pad

                # Большое окно L
                L = padded[cy_p - large_size // 2: cy_p + large_size // 2,
                           cx_p - large_size // 2: cx_p + large_size // 2].flatten()

                if L.size == 0:
                    continue

                # Порог — среднее по окну L
                B = np.mean(L)

                # Кластеры
                G1 = [v for v in L if v <= B]
                G2 = [v for v in L if v > B]

                if len(G1) == 0 or len(G2) == 0:
                    continue

                mu1 = sum(G1) / len(G1)
                mu2 = sum(G2) / len(G2)

                # Малое окно S (без padding)
                S = img[y:y + small_size, x:x + small_size]

                if abs(mu1 - mu2) >= l_threshold:
                    # Бинаризация в S по порогу B
                    binarized = np.where(S > B, 255, 0)
                    result[y:y + small_size, x:x + small_size] = binarized
                else:
                    # Усреднение яркости в S
                    mean_val = int(np.mean(S))
                    result[y:y + small_size, x:x + small_size] = mean_val

        self.image = result
        self.display_image(self.image)
#Сегментация Сезана
    def sezan_segmentation(self):
        if self.image is None:
            return
        hist, _ = np.histogram(self.image.flatten(), bins=256, range=(0, 256))
        cumulative = np.cumsum(hist)
        total = cumulative[-1]
        mean = np.cumsum(hist * np.arange(256)) / total
        variance = mean[-1] * cumulative - np.cumsum(hist * np.arange(256))
        thresholds = np.where(variance > 0)[0]
        if len(thresholds) == 0:
            t = np.mean(self.image)
        else:
            t = thresholds[np.argmax(variance[thresholds])]
        segmented = (self.image > t) * 255
        self.image = segmented
        self.display_image(self.image)
    #Возвращение исходного изображения
    def reset_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.display_image(self.image)
    #Свертка
    def convolve(self, img, kernel):
        k = kernel.shape[0] // 2
        padded = np.pad(img, k, mode='edge')
        result = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+2*k+1, j:j+2*k+1]
                result[i, j] = np.sum(region * kernel)
        return result
    #Ядро Гаусс
    def gaussian_kernel(self, size, sigma):
        ax = np.linspace(-(size // 2), size // 2, size)
        gauss = np.exp(-0.5 * np.square(ax) / sigma**2)
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)
    #Ядро Лапласиан Гаусс
    def laplacian_of_gaussian(self, img):
        kernel = np.array([[0,  0, -1,  0,  0],
                           [0, -1, -2, -1,  0],
                           [-1, -2, 16, -2, -1],
                           [0, -1, -2, -1,  0],
                           [0,  0, -1,  0,  0]], dtype=np.float32)
        return self.convolve(img, kernel)
    #Вспомогательная функция
    def zero_crossing(self, log_img):
        zc = np.zeros_like(log_img, dtype=np.uint8)
        for i in range(1, log_img.shape[0] - 1):
            for j in range(1, log_img.shape[1] - 1):
                patch = log_img[i-1:i+2, j-1:j+2]
                if np.max(patch) > 0 and np.min(patch) < 0:
                    zc[i, j] = 1
        return zc

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageProcessor()
    win.resize(1000, 700)
    win.show()
    sys.exit(app.exec_())