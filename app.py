import os
import sys
import pandas as pd
from pathlib import Path

import streamlit as st
from streamlit import runtime
from streamlit.web import cli as stcli

from ConsignmentItemizedData import CID

st.set_page_config(layout="wide")
pd.set_option('display.max_columns', None)


class App:
    cid = None

    files_name = []
    pdf_files = []
    uploaded_files = []
    previous_files = []

    def __init__(self):
        self.header()
        self.uploadFile()
        self.processor()

    '''
    Set the title and page configuration for wider layout
    '''
    def header(self):
        st.write('# Consignment Itemized Data')

    '''
    Upload pdf file
    '''
    def uploadFile(self):
        self.uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        self.files_name = [Path(f.name).stem for f in self.uploaded_files]

        st.write(self.files_name)

    def processor(self):
        if hasattr(self, 'files_name') and self.files_name:
            for file, up_file in zip(self.files_name, self.uploaded_files):
                self.cid = CID()

                # Create a directory to store data based on name and datetime
                location = self.cid.setFolderPath(file)

                # Save the uploaded PDF file respective location
                pdf_path = f'{location}/{file}.pdf'
                with open(pdf_path, "wb") as f:
                    f.write(up_file.read())
                st.success(f"File '{file}'.pdf has been successfully uploaded.")

                # Convert PDF to images
                self.cid.converter(pdf_path)

                # Detect Table(s)
                self.cid.tableDetection()

                # Detect Row(s)
                self.cid.rowDetection(file)

    @staticmethod
    def deleteLocalFiles(file):
        local_path = f'data/temp/{file.name}'
        st.write(f'local_path: {local_path}')
        if os.path.exists(local_path):
            os.remove(local_path)
            st.warning(f"File '{file.name}' has been deleted from local storage.")


if __name__ == '__main__':
    if runtime.exists():
        # If the runtime environment exists, create a Deployment object and start the runner
        dep = App()

    else:
        # If the runtime environment doesn't exist, start the Streamlit application
        sys.argv = ['streamlit', 'run', 'app.py', '--server.runOnSave=true']
        sys.exit(stcli.main())

# This code checks for the presence of a specific runtime environment and launches a Streamlit application if the
# environment doesn't exist. The 'runtime' object is used to check for the environment, and the 'exists()' function
# returns True if the environment is present and False otherwise. If the environment exists, a Deployment object is
# created and its 'runner()' method is called. Otherwise, the Streamlit application is started using the 'sys.argv' and
# 'stcli.main()' functions.
