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

from RowDetection import RowDetection
from TableDetection import TableDetection

poppler_path = r'C:\Program Files\poppler-23.05.0\Library\bin'
os.environ["PATH"] += os.pathsep + poppler_path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class CID:
    claim_no = None
    images_path = None
    dataset_path = None
    output_folder_path = None

    images_list = []

    def __init__(self):
        self.images_list = []
        self.images_path = None
        self.dataset_path = None
        self.output_folder_path = None

    def runner(self):
        pass

    @staticmethod
    def setFolderPath(file_name):
        path = f"output/{file_name}_{str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))}"
        CID.createFolder(path)
        return path

    '''
    Create a new folder if it does not already exist.
    @param directory: a string representing the path of the directory to be created.
    '''
    @staticmethod
    def createFolder(directory):
        try:
            os.makedirs(directory)
            print(f'{directory} has been made')
        except FileExistsError:
            pass

    '''
    Process the selected files by copying them to the specified destination folder.
    @param files: a list of strings representing the paths of the files to be processed.
    @param destination: a string representing the path of the destination folder.
    '''
    @staticmethod
    def processSelectedFiles(files, destination):
        # Remove existing files in the destination folder
        for existing_file in os.listdir(destination):
            file_path = os.path.join(destination, existing_file)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the existing file

        # Copy the selected files to the destination folder
        for path in files:
            shutil.copy(path, destination)  # Copy the file to the destination folder

    def converter(self, file_path):
        # Split folder and file name
        self.output_folder_path, file_name_with_ext = os.path.split(file_path)
        file_name, ext = os.path.splitext(file_name_with_ext)

        # Convert image
        images = convert_from_path(file_path, dpi=300)

        # Save images to pre-defined location
        self.saveImages(images, file_name)

    '''
    Save converted images to the output folder.
    @param images: A list of images to be saved.
    @param of: A string representing the path to the folder where the images will be saved.
    @param pdf_name: A string representing the name of the original PDF file (without extension).
    '''
    def saveImages(self, images, pdf_name):
        for idx, img in enumerate(images):
            deskewed_img = CID.deskew(np.array(img))
            page_index = str(idx + 1).zfill(2)
            img_path = os.path.join(self.output_folder_path, f'{pdf_name}_page_{page_index}.png')
            cv2.imwrite(img_path, deskewed_img)

    '''
    Deskews an image and saves it back to the input path.
    @:param input_path (str)
    '''
    @staticmethod
    def deskew(image):
        # Convert the image to grayscale
        grayscale = rgb2gray(image)

        # Determine the skew angle of the image
        angle = determine_skew(grayscale)

        # Rotate the image to correct the skew and scale it back to 8-bit
        rotated = rotate(image, angle, resize=True) * 255
        rotated = rotated.astype(np.uint8)

        return rotated

    def tableDetection(self):
        table_detector = TableDetection(self.output_folder_path)
        table_detector.runner()
        self.images_list = table_detector.images_list

    def rowDetection(self, claim_no):
        row_detector = RowDetection(self.output_folder_path, self.images_list)
        row_detector.runner(claim_no)


if __name__ == '__main__':
    cid = CID()
    cid.runner()
