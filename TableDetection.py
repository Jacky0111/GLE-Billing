import os
import glob

from Detect import Detect


class TableDetection:
    def __init__(self, output_folder_path):
        self.output_folder_path = output_folder_path
        self.images_list = []

    '''
    Perform table detection by changing directory, finding images, moving up directories, and detecting tables.
    '''
    def runner(self):
        self.changedDirectory()
        self.findImages()
        self.moveUpDirectories()
        self.detectTables()

    '''
    Change the current working directory to the output folder path.
    '''
    def changedDirectory(self):
        os.chdir(self.output_folder_path)

    '''
    Find all PNG images in the current directory and store their names without extension.
    '''
    def findImages(self):
        self.images_list = [os.path.splitext(filename)[0] for filename in glob.glob('*.png')]

    '''
    Move up two directories from the current working directory.
    '''
    def moveUpDirectories(self):
        os.chdir(os.path.dirname(os.path.dirname(os.getcwd())))

    '''
    Detect tables in the images using the 'table.pt' file and a threshold of 0.7.
    '''
    def detectTables(self):
        for img in self.images_list:
            Detect.parseOpt(self.output_folder_path, img, 'table.pt', 0.7)
