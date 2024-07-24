# Empowers the end user to make choices between performing document conversion and optical character recognition (OCR)

import os
import cv2
import glob
import shutil
import pytesseract
import numpy as np
from datetime import datetime
from deskew import determine_skew
from pdf2image import convert_from_path

from skimage.color import rgb2gray
from skimage.transform import rotate

from Detect import Detect
# from OpticalCharacterRecognition import OCR

poppler_path = r'C:\Program Files\poppler-23.05.0\Library\bin'
os.environ["PATH"] += os.pathsep + poppler_path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
