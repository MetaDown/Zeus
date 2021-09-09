from PyQt5.QtWidgets import QMenu, QGraphicsPixmapItem, QGraphicsScene, QFileDialog, QMessageBox
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from Primer_Programa import *
from PIL import ImageOps
import tensorflow as tf
import numpy as np
import cv2 as cv
import warnings
import PIL
import os


warnings.filterwarnings("ignore", category=DeprecationWarning)


#Clase principal para el GUI
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *arg, **kwargs):
        super().__init__()

        self.setupUi(self)

        