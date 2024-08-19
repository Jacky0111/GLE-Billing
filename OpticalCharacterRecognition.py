import os
import re
import cv2
import pandas as pd
from fuzzywuzzy import fuzz
from paddleocr import PaddleOCR

from Bill import Bill
from TabularRule import TabularRule
from TextPostProcessing import TPP


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
        cols_name = None
        img_file_list = self.getSortedImageFiles()

        for idx, file in enumerate(img_file_list):
            img_path = os.path.join(self.images_path, file)
            img = cv2.imread(img_path)
            temp_df = OCR.imageToData(img)
            temp_df = temp_df.sort_values(by='left', ascending=True)
            temp_df.reset_index(drop=True, inplace=True)

            print(temp_df)

            self.drawBoundingBox(img, temp_df)
            cv2.imwrite(f'{self.images_path}/bbox_{file}', img)

            # 10 columns
            if idx == 0:
                double_counter = 0
                triple_counter = 0
                t1 = temp_df[:10].copy()

                for index, row in t1.iterrows():
                    # First col
                    if index == 0:
                        t1.loc[index, 'left'] = index
                        t1.loc[index, 'width'] = temp_df.loc[index, 'width'] * 0.4 + temp_df.loc[index, 'left']

                    # Second col
                    elif index == 1:
                        t1.loc[index, 'left'] = temp_df.loc[index-1, 'left'] + temp_df.loc[index-1, 'width'] - temp_df.loc[index-1, 'width'] * 0.55
                        t1.loc[index, 'top'] = t1.loc[index-1, 'top']
                        t1.loc[index, 'width'] = temp_df.loc[index-1, 'width'] * 0.55
                        t1.loc[index, 'height'] = t1.loc[index-1, 'height']

                    # Third col
                    if index == 2:
                        t1.loc[index] = temp_df.loc[index]

                    # Forth col and after
                    elif index in [2, 3, 4, 5, 8, 9]:
                        temp_row = index + double_counter

                        if index == 8 or index == 9:
                            temp_row += 4

                        x0 = temp_df.loc[temp_row-1, 'left']
                        x1 = temp_df.loc[temp_row, 'left']

                        y0 = temp_df.loc[temp_row-1, 'top']
                        y1 = temp_df.loc[temp_row, 'top']

                        w0 = temp_df.loc[temp_row-1, 'width']
                        w1 = temp_df.loc[temp_row, 'width']

                        t1.loc[index, 'left'] = x0 if x0 < x1 else x1
                        t1.loc[index, 'top'] = y0 if y0 < y1 else y1
                        t1.loc[index, 'width'] = w0 if w0 > w1 else w1
                        if t1.loc[index, 'top'] == y0:
                            t1.loc[index, 'height'] = temp_df.loc[temp_row, 'height'] + y1
                        elif t1.loc[index, 'top'] == y1:
                            t1.loc[index, 'height'] = temp_df.loc[temp_row-1, 'height'] + y0

                        double_counter += 1

                    elif index == [6, 7]:
                        temp_row = index + 2 + 2 * triple_counter

                        x0 = temp_df.loc[temp_row, 'left']
                        x1 = temp_df.loc[temp_row+1, 'left']
                        x2 = temp_df.loc[temp_row+2, 'left']

                        y0 = temp_df.loc[temp_row, 'top']
                        y1 = temp_df.loc[temp_row+1, 'top']
                        y2 = temp_df.loc[temp_row+2, 'top']

                        w0 = temp_df.loc[temp_row, 'width']
                        w1 = temp_df.loc[temp_row+1, 'width']
                        w2 = temp_df.loc[temp_row+2, 'width']

                        t1.loc[index, 'left'] = min(x0, x1, x2)
                        t1.loc[index, 'top'] = min(y0, y1, y2)
                        t1.loc[index, 'width'] = max(w0, w1, w2)
                        max_y = max(y0, y1, y2)
                        temp_df_range = temp_df.loc[temp_row:temp_row+3]
                        matching_height = temp_df_range[temp_df_range['top'] == max_y]['height'].values[0]
                        t1.loc[index, 'height'] = max_y + matching_height

                        triple_counter += 1

                cols_name, temp_df = self.checkHospital(t1.iloc[:, :-1])
            else:
                try:
                    temp_df = self.filterTempDataFrame(temp_df, cols_name)
                except ValueError:
                    pass

                # Concatenate the data to the final DataFrame
                self.df = pd.concat([self.df, temp_df], ignore_index=True)

            self.drawBoundingBox(img, temp_df)
            cv2.imwrite(f'{self.images_path}/bbox_{file}', img)

            bill_list = self.bill.assignCoordinate(temp_df)
            for bill in bill_list:
                print(f'{idx}. bill_list: {bill}')
            tr = TabularRule(bill_list, True if idx == 0 else False)
            tr.runner()
            self.table_data_list.append(tr.row_list)

            tb_list = [[element.text for element in row] for row in self.table_data_list]

            print()
            print(f'tb_list: \t{tb_list}')

            tpp = TPP(self.claim_no, self.cols)
            itemized_data = tpp.runner(tb_list)

        self.saveToExcel(self.df, 'image_to_data')
        self.saveToCSV(itemized_data, 'itemized_data')

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
    def adjustFirstImage(temp_df):
        df = temp_df.iloc[:3]
        df = pd.concat([df.iloc[:1], df]).reset_index(drop=True)
        df.loc[1, 'width'] /= 2
        df.loc[0, 'width'] = df.loc[1, 'width']
        df.loc[0, 'left'] = 0
        df.loc[1, 'left'] = df.loc[1, 'width'] + 41
        return df

    '''
    Adjust the alignment of the second image's data.
    @param temp_df: DataFrame containing the OCR results from the second image.
    @return t2: Adjusted DataFrame with modified alignment.
    '''
    @staticmethod
    def adjustSecondImage(temp_df):
        df = temp_df.iloc[:3]
        df = pd.concat([df.iloc[:1], df]).reset_index(drop=True)
        df.loc[1, 'width'] /= 2
        df.loc[0, 'width'] = df.loc[1, 'width']
        df.loc[0, 'left'] = 0
        df.loc[1, 'left'] = df.loc[1, 'width'] + 41
        return df

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
        # date_pattern = r'\b\d{2}/\d{2}/\d{4}\b'

        self.cols.append(tb_list[0])

        # for sublist in tb_list:
        #     if len(sublist) > 1:
        #         sentence = sublist[1]
        #         dates = re.findall(date_pattern, sentence)
        #         if dates:
        #             date = dates[0]
        #             sentence = re.sub(date_pattern, '', sentence)
        #             del sublist[1]
        #             sublist.insert(1, date)
        #             sublist.insert(1, sentence)

        try:
            itemized_data = pd.DataFrame(tb_list[1:], columns=self.cols[0])
        except ValueError as e:
            self.adjustCols(tb_list, e)

        # Print adjusted columns to check
        print("Adjusted Columns:", tb_list[0])
        itemized_data = pd.DataFrame(tb_list[1:], columns=tb_list[0])


        itemized_data.insert(0, 'ClaimNo', self.claim_no * len(itemized_data))

        # df_temp = pd.read_excel(r'claim_data.xlsx')
        # Get the PolicyNo from the matching row
        # policy_number = df_temp.loc[df_temp['ClaimNo'] == self.claim_no[0], 'PolicyNo'].iloc[0]
        policy_number = '1234567'

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
            print(f'numbers[0] > numbers[1]: {numbers[0] > numbers[1]}')
            num_columns_in_data = len(tb_list[0])
            print(f'num_columns: {num_columns_in_data}')
            max_columns = len(tb_list[1])
            print(f'max_columns_in_data: {max_columns}')
            # If the number of columns in headers is less than the number of columns in any data row, add None or ''
            print(f'Before: {tb_list[1]}')
            tb_list[1].extend([None] * (num_columns_in_data - max_columns))

        elif numbers[0] < numbers[1]:
            print(f'numbers[0] < numbers[1]: {numbers[0] < numbers[1]}')
            num_columns = len(tb_list[0])
            print(f'num_columns: {num_columns}')
            max_columns_in_data = max(len(row) for row in tb_list[1:])
            print(f'max_columns_in_data: {max_columns_in_data}')
            # If the number of columns in headers is less than the number of columns in any data row, add None or ''
            print(f'Before: {tb_list[0]}')
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

        data = data.drop([3, 8])
        header_name = [text for i, text in enumerate(header_name) if i not in [3, 8]]

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
