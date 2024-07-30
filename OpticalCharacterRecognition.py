import os
import re
import cv2
import pandas as pd
from fuzzywuzzy import fuzz
from paddleocr import PaddleOCR

from Bill import Bill
from TabularRule import TabularRule


class OCR:
    bill = None
    claim_no = None
    output_path = None  # Current save path
    images_path = None  # Input images path

    df = pd.DataFrame()

    cols = []
    table_data_list = []

    def __init__(self, output_path, images_path, claim_no):
        self.bill = Bill()
        self.table_data_list.clear()
        self.claim_no = claim_no
        self.output_path = output_path
        self.images_path = images_path
