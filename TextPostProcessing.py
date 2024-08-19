import re
import pandas as pd


class TPP:
    def __init__(self, claim_no, columns):
        self.claim_no = claim_no
        self.columns = columns

    def runner(self, table_list):
        data = pd.DataFrame()

        # Append column names to the list
        self.columns.append(table_list[0])

        try:
            # Create DataFrame using the provided column names
            data = pd.DataFrame(table_list[1:], columns=self.columns[0])
        except ValueError as e:
            # Adjust columns in case of mismatch
            self.adjustCols(table_list, e)

        # Print adjusted columns for verification
        print("Adjusted Columns:", table_list[0])

        try:
            # Create DataFrame using the provided column names
            data = pd.DataFrame(table_list[1:], columns=table_list[0])
        except ValueError as e:
            # Adjust columns in case of mismatch
            self.adjustCols(table_list, e)

        # Convert values first
        data = TPP.convertValues(data)

        # Apply the functions to the DataFrame
        data = self.handleConsecutiveStrings(data)
        data = TPP.handleDatesAndShift(data)

        # Insert ClaimNo column at the beginning
        data.insert(0, 'ClaimNo', self.claim_no)

        # Placeholder policy number (to be fetched from another source)
        policy_number = '1234567'

        df_temp = pd.read_excel(r'claim_data.xlsx')
        # Get the PolicyNo from the matching row
        # Find the row where ClaimNo is equal to 'ALMCIP02180441'
        matching_row = df_temp[df_temp['ClaimNo'] == self.claim_no]
        # Get the PolicyNo from the matching row
        policy_number = matching_row['PolicyNo'].iloc[0] if not matching_row.empty else None
        print(f'Type: {type(policy_number)}')

        # Insert PolicyNo column at the beginning
        data.insert(0, 'PolicyNo', policy_number)

        return data
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

    @staticmethod
    def convertValues(df):
        for col in df.columns[1:]:  # Skip the first column
            df[col] = df[col].apply(lambda x: TPP.tryConvert(x) if isinstance(x, str) else x)
        return df

    # Function to try to convert value to float
    @staticmethod
    def tryConvert(value):
        try:
            return float(value.replace(',', ''))
        except ValueError:
            return value

    # Function to handle combining consecutive strings in the 2nd and 3rd columns
    def handleConsecutiveStrings(self, df):
        new_data = []

        for i, row in df.iterrows():
            new_row = row.tolist()
            print(f'new_row: {new_row}')
            # Check from the 2nd column onwards
            col_index = 1

            while col_index < len(new_row) - 1:
                print(f'col_index: {col_index}')
                if isinstance(new_row[col_index], str) and isinstance(new_row[col_index + 1], str):
                    print(f'new_row[{col_index}]: {new_row[col_index]}')
                    print(f'new_row[{col_index + 1}]: {new_row[col_index + 1]}')

                    combined = f"{new_row[col_index]} {new_row[col_index + 1]}"
                    print(f'combined: {combined}')

                    # Move all subsequent elements to the left by one position
                    new_row[col_index:] = new_row[col_index + 1:]
                    new_row[col_index] = combined

                    if col_index + 1 < len(new_row) and not isinstance(new_row[col_index + 1], float):
                        break

                col_index += 1
            new_data.append(new_row)

        # Ensure that each row in new_data has the same number of columns as df.columns
        for i in range(len(new_data)):
            if len(new_data[i]) < len(df.columns):
                new_data[i].extend([None] * (len(df.columns) - len(new_data[i])))

        try:
            # Create DataFrame using the provided column names
            df = pd.DataFrame(new_data, columns=df.columns)
        except ValueError as e:
            # Adjust columns in case of mismatch
            self.adjustCols(new_data, e)

        return df

    # Function to handle the conversion of the first column to date and shift rows
    @staticmethod
    def handleDatesAndShift(df):
        new_rows = []
        for i, row in df.iterrows():
            try:
                # Try converting the first column to a date
                pd.to_datetime(row[0], format='%d.%m.%Y')
                new_rows.append(row.tolist())
            except ValueError:
                # If conversion fails, insert an empty string at the start and shift the row
                new_row = [''] + row.tolist()[:-1]
                new_rows.append(new_row)
        return pd.DataFrame(new_rows, columns=df.columns)
