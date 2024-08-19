import os
import pandas as pd

# Define paths
destination_folder = r'Run Data'  # Destination folder in the current directory
excel_file_path = r'claim_data.xlsx'

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read the Excel file
df = pd.read_excel(excel_file_path)

# Get a list of filenames in the destination folder
files_in_destination = os.listdir(destination_folder)

# Check if each ClaimNO exists in the destination folder
extracted_status = []
for claim_no in df['ClaimNo']:
    if any(claim_no in filename for filename in files_in_destination):
        extracted_status.append(True)
    else:
        extracted_status.append(False)


# Add the 'Extracted' column to the dataframe
df['Extracted'] = extracted_status

# Save the updated dataframe back to the Excel file
df.to_excel(excel_file_path, index=False)
