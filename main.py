import sys
import cv2
import numpy as np
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

"""
Расчетные, вспомогательные функции
"""

def add_gaussian_noise(image, mean=0, sigma=20):
    """
    I_noisy=(x,y)=I(x,y)+N(μ,σ^2)

    Где:
    I(x,y) — исходная яркость пикселя в точке (x,y),
    N(μ,σ^2) — случайная величина из нормального распределения,
    μ (mean) — среднее значение шума (обычно 0),
    σ (sigma) — стандартное отклонение (сила шума).
    """
    noisy = image.copy().astype(np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            noisy[i, j] += random.gauss(mean, sigma)

    return np.clip(noisy, 0, 255).astype(np.uint8)

def mean_filter(image, kernel_size=3):
    """
    I_filtered(i,j)=1/k^2 * x=−p∑p y=−p∑p I(i+x,j+y)

    k — размер ядра (kernel_size),
    p=k//2 (например, для k=3: p=1).
    """
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='edge')
    filtered = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.mean(region)
    return filtered.astype(np.uint8)

def laplacian_of_gaussian(image, kernel_size=5, sigma=1.0):
    """
    LoG(x,y)=−(1/πσ^4)(1−(x^2+y^2/2σ^2))exp[−x^2+y^2/2σ^2]
    """
    def log_kernel(size):
        kernel = np.zeros((size, size), dtype=np.float32)
        "Создание ядра LoG, дискретного аналога оператора ∇^2G"
        offset = size // 2

        for x in range(-offset, offset + 1):
            for y in range(-offset, offset + 1):
                r2 = x**2 + y**2
                kernel[x + offset, y + offset] = (
                    ((r2 - 2 * sigma**2) / (sigma**4)) *
                    np.exp(-r2 / (2 * sigma**2))
                )
        kernel -= np.mean(kernel)

        return kernel

    kernel = log_kernel(kernel_size)
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='edge')
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.sum(region * kernel)

    return np.clip(result, 0, 255).astype(np.uint8)

def eikvil_threshold(image, epsilon=1):
    """
    Алгоритм ищет оптимальный порог T, разделяющий изображение на два класса:

    Класс 1 (G₁): Пиксели со значениями > T (обычно объекты)
    Класс 2 (G₂): Пиксели со значениями ≤ T (обычно фон)
    """
    T = np.mean(image) # Начальный порог ==> средняя интенсивность
    prev_T = 0
    while abs(T - prev_T) >= epsilon:
        G1 = image[image > T]
        G2 = image[image <= T]
        m1 = G1.mean() if len(G1) > 0 else 0
        m2 = G2.mean() if len(G2) > 0 else 0
        prev_T = T
        T = (m1 + m2) / 2
    binary = (image > T).astype(np.uint8) * 255
    # Пиксели выше порога становятся 255 (белые), остальные — 0 (чёрные).
    return binary


def abutaleb_threshold(image):
    # Построение двумерной гистограммы
    hist2d = np.zeros((256, 256), dtype=np.float64)
    padded = np.pad(image, 1, mode='edge')

    for i in range(1, padded.shape[0] - 1):
        for j in range(1, padded.shape[1] - 1):
            neighborhood = padded[i - 1:i + 2, j - 1:j + 2]
            avg_local = int(np.mean(neighborhood))
            pixel_val = int(padded[i, j])
            hist2d[pixel_val, avg_local] += 1

    # Нормализация гистограммы
    hist2d /= np.sum(hist2d)

    max_entropy = -np.inf
    best_t = 0

    for t in range(1, 255):
        # Подматрицы A и B
        A = hist2d[:t+1, :t+1]
        B = hist2d[t+1:, t+1:]

        pA = A.sum()
        pB = B.sum()

        if pA == 0 or pB == 0:
            continue

        hA = -np.sum((A / pA) * np.log((A + 1e-12) / pA))
        hB = -np.sum((B / pB) * np.log((B + 1e-12) / pB))

        total_entropy = hA + hB

        if total_entropy > max_entropy:
            max_entropy = total_entropy
            best_t = t

    # Бинаризация по найденному порогу
    binary = np.where(image > best_t, 255, 0).astype(np.uint8)
    return binary


"""
Втроенная оптимизация QT
"""

class ProcessingThread(QThread):
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self, image, operation):
        super().__init__()
        self.image = image
        self.operation = operation

    def run(self):
        if self.operation == "noise":
            result = add_gaussian_noise(self.image)
        elif self.operation == "filter":
            result = mean_filter(self.image)
        elif self.operation == "log":
            result = laplacian_of_gaussian(self.image)
        elif self.operation == "eikvil":
            result = eikvil_threshold(self.image)
        elif self.operation == "abutaleb":
            result = abutaleb_threshold(self.image)
        else:
            result = self.image
        self.result_ready.emit(result)

"""
Приложение
"""

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обработка цифровых изображений — курсовая")

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.original_image = None
        self.processed_image = None

        # Кнопки
        load_btn = QPushButton("Загрузить")
        noise_btn = QPushButton("Добавить шум")
        filter_btn = QPushButton("Фильтрация")
        log_btn = QPushButton("Оператор ЛоГ")
        eikvil_btn = QPushButton("Сегм. Эйквила")
        abutaleb_btn = QPushButton("Сегм. Абуталеба")

        load_btn.clicked.connect(self.load_image)
        noise_btn.clicked.connect(lambda: self.process("noise"))
        filter_btn.clicked.connect(lambda: self.process("filter"))
        log_btn.clicked.connect(lambda: self.process("log"))
        eikvil_btn.clicked.connect(lambda: self.process("eikvil"))
        abutaleb_btn.clicked.connect(lambda: self.process("abutaleb"))

        # Layout
        btn_layout = QHBoxLayout()
        for btn in [load_btn, noise_btn, filter_btn, log_btn, eikvil_btn, abutaleb_btn]:
            btn_layout.addWidget(btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выбрать изображение", "", "Images (*.png *.jpg *.bmp)")
        if path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.original_image = img
            self.processed_image = img.copy()
            self.display_image(img)

    def display_image(self, image):
        height, width = image.shape
        bytes_per_line = width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img).scaled(512, 512, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def process(self, operation):
        if self.processed_image is not None:
            self.thread = ProcessingThread(self.processed_image, operation)
            self.thread.result_ready.connect(self.on_result)
            self.thread.start()

    def on_result(self, result):
        self.processed_image = result
        self.display_image(result)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec_())
