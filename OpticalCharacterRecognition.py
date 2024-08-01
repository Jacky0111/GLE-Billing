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

    '''
    Execution method to process images, extract text using OCR, and save the results.
    - Processes each image in the directory.
    - Extracts text and bounding boxes using OCR.
    - Adjusts data alignment based on image index.
    - Applies tabular rules and saves the results.
    '''
    def runner(self):
        t1 = 0
        t2 = 0
        t3 = 0
        cols_name = None
        img_file_list = self.getSortedImageFiles()

        for idx, file in enumerate(img_file_list):
            img_path = os.path.join(self.images_path, file)
            img = cv2.imread(img_path)
            temp_df = OCR.imageToData(img)
            temp_df = temp_df.sort_values(by='left', ascending=True)

            print(temp_df)

            if idx == 0:
                t1 = temp_df[4:]

                print(t1)
                print('t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1t1')
                self.drawBoundingBox(img, temp_df)
                cv2.imwrite(f'{self.images_path}/bbox_{file}', img)

                cols_name, temp_df = self.checkHospital(t3.iloc[:, :-1])

                continue
            # elif idx == 1:
            #     t2 = self.adjustSecondImage(temp_df)
            #
            #     print(t2)
            #     print('t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2t2')
            #
            #     t3 = pd.concat([t2, t1]).reset_index(drop=True)
            #
            #     print(t3)
            #     print('t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3t3')
            #
            #     cols_name, temp_df = self.checkHospital(t3.iloc[:, :-1])
            # else:
            #     temp_df = self.filterTempDataFrame(temp_df, cols_name)
            #     self.df = pd.concat([self.df, temp_df], ignore_index=True)
            #     self.drawBoundingBox(img, temp_df)
            #     cv2.imwrite(f'{self.images_path}/bbox_{file}', img)
            #
            #     bill_list = self.bill.assignCoordinate(temp_df)
            #     tr = TabularRule(bill_list, True if idx == 1 else False)
            #     tr.runner()
            #     self.table_data_list.append(tr.row_list)

            tb_list = [[element.text for element in row] for row in self.table_data_list]
            self.postProcessData(tb_list)

    '''
    Get a sorted list of image files from the given directory.
    @return: Sorted list of image file names
    '''
    def getSortedImageFiles(self):
        img_file_list = [file for file in os.listdir(self.images_path) if not file.startswith('._')]
        return sorted(img_file_list, key=self.extractNumbers)

    '''
    Adjust the second row of the DataFrame for proper alignment.
    @param s: Filename string.
    @return: A tuple of integers representing the extracted numbers.
    '''
    @staticmethod
    def extractNumbers(s):
        match = re.search(r'row_(\d+)_(\d+).png', s)
        return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

    '''
    Extract text and bounding boxes from an image using OCR.
    @param img: The image to be processed.
    @return: DataFrame containing bounding box coordinates, text, and confidence scores.
    '''
    @staticmethod
    def imageToData(img):
        # paddle = PaddleOCR(use_angle_cls=True, lang='en')
        paddle = PaddleOCR(lang='en',
                           use_angle_cls=True,
                           det_algorithm='DB',
                           det_db_box_thresh=0.3,
                           det_db_unclip_ratio=1.5,
                           rec_algorithm='CRNN',
                           max_candidates=1000)
        # paddle = PaddleOCR(det_db_box_thresh=0.3, det_db_unclip_ratio=0.5, use_angle_cls=True, lang='en')

        result = paddle.ocr(img, cls=True)

        lines = []
        for line in result:
            if line:
                for word_info in line:
                    coordinates = word_info[0]
                    x_values, y_values = zip(*coordinates)
                    left, top, right, bottom = min(x_values), min(y_values), max(x_values), max(y_values)
                    width, height = right - left, bottom - top
                    text = word_info[1][0]
                    conf = f"{word_info[1][1]:.4f}"
                    lines.append([left, top, width, height, conf, text])

        columns = ['left', 'top', 'width', 'height', 'conf', 'text']

        return pd.DataFrame(lines, columns=columns)

    '''
    Adjust the alignment of the second image's data.
    @param temp_df: DataFrame containing the OCR results from the second image.
    @return t2: Adjusted DataFrame with modified alignment.
    '''
    @staticmethod
    def adjustSecondImage(temp_df):
        t2 = temp_df.iloc[:3]
        t2 = pd.concat([t2.iloc[:1], t2]).reset_index(drop=True)
        t2.loc[1, 'width'] /= 2
        t2.loc[0, 'width'] = t2.loc[1, 'width']
        t2.loc[0, 'left'] = 0
        t2.loc[1, 'left'] = t2.loc[1, 'width'] + 41
        return t2

    '''
    Filter the temporary DataFrame based on similarity to header names.
    @param temp_df: DataFrame containing the OCR results to be filtered.
    @param cols_name: List of expected column headers.
    @return: Filtered DataFrame.
    '''
    @staticmethod
    def filterTempDataFrame(temp_df, cols_name):
        temp_df['most_similar_header'], temp_df['similarity_score'] = zip(
            *temp_df['text'].apply(OCR.findMostSimilarHeaderAndSimilarity, header_name=cols_name))
        return temp_df[temp_df['similarity_score'] <= 50]

    '''
    Find the most similar header and calculate similarity score using fuzzy matching.
    @param text: The text to be matched.
    @param header_name: List of header names to match against.
    @return most_similar_header
    @return max_similarity: similarity score
    '''
    @staticmethod
    def findMostSimilarHeaderAndSimilarity(text, header_name):
        max_similarity, most_similar_header = max(
            ((fuzz.ratio(text.lower(), header.lower()), header) for header in header_name),
            key=lambda x: x[0]
        )
        return most_similar_header, max_similarity

    '''
    Draw bounding boxes on the image based on OCR results.
    @param img: The image to be processed.
    @param boxes: DataFrame with bounding box coordinates and text.
    '''
    @staticmethod
    def drawBoundingBox(img, boxes):
        red = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for _, box in boxes.iterrows():
            x, y, w, h = int(box['left']), int(box['top']), int(box['width']), int(box['height'])
            cv2.rectangle(img, (x, y), (w + x, h + y), red, 1)
            text = f"{box['text']} {_}"
            cv2.putText(img, text, (x, y), font, 0.5, red, 1)

    '''
    Post-process the extracted data and save it to files.
    - Adjusts columns and adds additional information.
    - Saves the processed data to both Excel and CSV files.
    @param tb_list: List of text data extracted from images.
    '''
    def postProcessData(self, tb_list):
        itemized_data = pd.DataFrame()
        date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'

        self.cols.append(tb_list[0])

        for sublist in tb_list:
            if len(sublist) > 1:
                sentence = sublist[1]
                dates = re.findall(date_pattern, sentence)
                if dates:
                    date = dates[0]
                    sentence = re.sub(date_pattern, '', sentence)
                    del sublist[1]
                    sublist.insert(1, date)
                    sublist.insert(1, sentence)

        try:
            itemized_data = pd.DataFrame(tb_list[1:], columns=self.cols[0])
        except ValueError as e:
            self.adjustCols(tb_list, e)
        itemized_data.insert(0, 'ClaimNo', self.claim_no * len(itemized_data))

        df_temp = pd.read_excel(r'claim_data.xlsx')
        # Get the PolicyNo from the matching row
        policy_number = df_temp.loc[df_temp['ClaimNo'] == self.claim_no[0], 'PolicyNo'].iloc[0]

        self.saveToExcel(self.df, 'image_to_data')

        itemized_data.insert(0, 'PolicyNo', policy_number)
        # self.saveToExcel(itemized_data, 'itemized_data')
        self.saveToCSV(itemized_data, 'itemized_data')

    '''
    Adjust the number of columns in the data based on the error encountered.
    @param tb_list: List of text data extracted from images.
    @param e: Exception raised during DataFrame creation.
    '''
    @staticmethod
    def adjustCols(tb_list, e):
        numbers = list(map(int, re.findall(r'\d+', str(e))))
        if numbers[0] > numbers[1]:
            num_columns_in_data = len(tb_list[0])
            max_columns = len(tb_list[1])
            tb_list[1].extend([None] * (num_columns_in_data - max_columns))
        elif numbers[0] < numbers[1]:
            num_columns = len(tb_list[0])
            max_columns_in_data = max(len(row) for row in tb_list[1:])
            tb_list[0].extend([None] * (max_columns_in_data - num_columns))

    '''
    Adjust the data based on the specific hospital's format.
    @param data: DataFrame containing the text data to be adjusted.
    @return: Adjusted column headers and DataFrame.
    '''
    def checkHospital(self, data):
        # if self.code == 'BAGAN':
        #     return OCR.BAGANAdjustment(data)
        # elif self.code == 'GNC':
        #     return OCR.GNCAdjustment(data)
        # elif self.code == 'KPJ':
        #     return OCR.KPJAdjustment(data)
        # elif self.code == 'RSH':
        #     return OCR.RSHAdjustment(data)

        return OCR.GLEAdjustment(data)

    @staticmethod
    def BAGANAdjustment(data):
        return

    @staticmethod
    def GLEAdjustment(data):
        header_name = ['Txn Date',
                       'Description',
                       'Qty',
                       'MMA Code',
                       'Amount (RM)',
                       'Discount (RM)',
                       'Gross Amount (RM)',
                       'Tax Amount (RM)',
                       'GST/Tax Code (RM)',
                       'Payable (RM)']

        # Insert the new row at the beginning of the DataFrame
        text_col = pd.DataFrame(header_name, columns=['text'])

        data = pd.concat([data, text_col], axis=1)
        print(data)

        return header_name, data

    @staticmethod
    def KPJAdjustment(data):
        header_name = ['Price Code',
                       'Description',
                       'Trans Date',
                       'Qty',
                       'Amount (RM)',
                       'GST/Tax Amount (RM)',
                       'Payable Amt (RM)']

        # Insert the new row at the beginning of the DataFrame
        text_col = pd.DataFrame(header_name, columns=['text'])

        data = pd.concat([data, text_col], axis=1)
        print(data)

        return header_name, data

    @staticmethod
    def RSHAdjustment(data):
        return

    '''
    Saved recognized text to csv file
    @param path
    '''
    def saveToCSV(self, data, name):
        print(f'{self.output_path}/{name}.csv')
        data.to_csv(f'{self.output_path}/{name}.csv', index=False)

    '''
    Saved recognized text to xlsx file
    @param path
    '''
    def saveToExcel(self, data, name):
        print(f'{self.output_path}/{name}.xlsx')
        data.to_excel(f'{self.output_path}/{name}.xlsx', index=False)
