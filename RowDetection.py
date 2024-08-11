import os
import cv2
import pytesseract

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from Detect import Detect
from OpticalCharacterRecognition import OCR


class RowDetection:
    """
    Initialize RowDetection with the output folder path and images list.
    @param output_folder_path: A string representing the path to the output folder.
    @param images_list: A list of image names to process.
    """
    def __init__(self, output_folder_path, images_list):
        self.output_folder_path = output_folder_path
        self.images_list = images_list

    '''
    Perform row detection and run OCR.
    @param claim_no: The claim number being processed.
    '''
    def runner(self, claim_no):
        table_boxes_path = f'{self.output_folder_path}/labels/table_boxes.txt'
        selected_pages = RowDetection.readSelectedPages(table_boxes_path)
        new_img_list = [img_name + '_crop' for img_name in self.images_list if int(img_name.split('_')[-1]) in selected_pages]
        self.createRowFolder()
        self.parseAndDetect(selected_pages, new_img_list)

        ocr = OCR(self.output_folder_path, self.createRowFolder(), claim_no)
        ocr.runner()

    '''
    Create a list of modified image names based on selected pages.
    @param selected_pages: A list of selected page numbers.
    @return: A list of new image names.
    '''
    @staticmethod
    def readSelectedPages(table_boxes_path):
        with open(table_boxes_path, 'r') as file:
            return [int(line.split()[1]) for line in file]

    '''
    Create a 'Row' folder in the output folder path.
    @return: A string representing the path to the 'Row' folder.
    '''
    def createRowFolder(self):
        os.makedirs(os.path.join(self.output_folder_path, 'Row'), exist_ok=True)
        return f'{self.output_folder_path}/Row'

    '''
    Parse options and detect rows using the 'row.pt' file.
    @param selected_pages: A list of selected page numbers.
    @param new_img_list: A list of new image names.
    '''
    def parseAndDetect(self, selected_pages, new_img_list):
        for page, img in zip(selected_pages, new_img_list):
            Detect.parseOpt(self.output_folder_path, img, 'row.pt', 0.3)
            self.processDetectedRows(page, img)

    '''
    Process detected rows, merge them, and save the results.
    @param page: The page number being processed.
    @param img: The image name being processed.
    '''
    def processDetectedRows(self, page, img):
        table_img_path = f'{self.output_folder_path}/{img}.png'
        row_boxes_path = f'{self.output_folder_path}/labels/row_boxes.txt'

        tb_img = cv2.imread(table_img_path)
        crop_img = tb_img.copy()

        with open(row_boxes_path, 'r') as file:
            lines = file.readlines()
            values = [list(map(float, line.strip().split()[1:])) for line in lines]

        values.sort(key=lambda j: j[1], reverse=False)
        merged_values = RowDetection.mergeRows(values)

        RowDetection.saveMergedRows(row_boxes_path, page, merged_values)
        self.drawAndSaveRows(tb_img, crop_img, page, merged_values, img)

    '''
    Merge rows if y-coordinate difference is less than or equal to the threshold.
    @param values: A list of row box values.
    @return: A list of merged row values.
    '''
    @staticmethod
    def mergeRows(values):
        threshold = 0.003
        merged_values = []
        merged_row = values[0]

        for idx in range(1, len(values)):
            current_row = values[idx]
            prev_row = merged_row

            if abs(current_row[1] - prev_row[1]) <= threshold:
                merged_row = [
                    min(prev_row[0], current_row[0]),
                    current_row[1],
                    current_row[2],
                    max(prev_row[3], current_row[3])
                ]
            else:
                merged_values.append(merged_row)
                merged_row = current_row

        merged_values.append(merged_row)
        merged_values.sort(key=lambda j: j[1], reverse=False)
        return merged_values

    '''
    Save merged rows to a text file.
    @param row_boxes_path: The path to the original row boxes file.
    @param page: The page number being processed.
    @param merged_values: A list of merged row values.
    '''
    @staticmethod
    def saveMergedRows(row_boxes_path, page, merged_values):
        with open(f'{row_boxes_path[:-4]}_{page}.txt', 'w') as output_file:
            for value in merged_values:
                output_file.write(f'{page} {value[0]} {value[1]} {value[2]} {value[3]}\n')

    '''
    Draw lines on the image and save cropped rows.
    @param tb_img: The table image.
    @param crop_img: The cropped image.
    @param page: The page number being processed.
    @param merged_values: A list of merged row values.
    '''
    def drawAndSaveRows(self, tb_img, crop_img, page, merged_values, img):

        row_folder = self.createRowFolder()

        for idx, value in enumerate(merged_values):
            x, y, w, h = value[0], value[1], value[2], value[3]
            y = int((y + h / 2) * tb_img.shape[0])
            w = int(w * tb_img.shape[1])
            h = int(h * tb_img.shape[0])

            cv2.line(tb_img, (0, y), (tb_img.shape[0] + w, y), (255, 0, 0), 2)
            cv2.line(tb_img, (0, 0 if y - h < 0 else y - h), (tb_img.shape[0] + w, 0 if y - h < 0 else y - h),
                     (255, 0, 0), 2)

            cropped_row = crop_img[0 if y - h < 0 else y - h:y, 0:crop_img.shape[1]]
            cropped_path = f'{row_folder}/row_{page}_{str(idx).zfill(3)}.png'
            cv2.imwrite(cropped_path, cropped_row)

            if idx > 0:
                RowDetection.removeEmptyImages(cropped_path)

        cv2.imwrite(f'{self.output_folder_path}/{img[:-5]}_row_revised.png', tb_img)

    '''
    Remove empty images after OCR check.
    @param cropped_path: The path to the cropped image.
    '''
    @staticmethod
    def removeEmptyImages(cropped_path):
        check_img = cv2.imread(cropped_path)
        gray = cv2.cvtColor(check_img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray).strip()
        desc_keywords = ['billing', 'group', 'sub', 'mma', 'code', 'gross', 'tax']
        is_exception_line = any(word in text.lower() for word in desc_keywords)

        if not text or is_exception_line:
            os.remove(cropped_path)

    '''
    Detects if the given text is in English.
    @param text: The text to detect the language for.
    @return: True if the text is in English, False otherwise.
    '''
    @staticmethod
    def isEnglish(text):
        try:
            language = detect(text)
            return language == 'en'
        except LangDetectException:
            return False






