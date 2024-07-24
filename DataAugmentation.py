import os
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew


class DataAugmentation:
    directory = None

    def __init__(self, directory):
        self.directory = directory

    def processor(self):
        # List all the image files in the folder
        images = [f for f in os.listdir(self.directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Iterate through each image file and deskew
        for index, img in enumerate(images):
            image_path = os.path.join(self.directory, img)
            print(f'{str(index + 1)}. {image_path}')
            self.Deskew(image_path)

    """
    Deskews an image and saves it back to the input path.
    @:param input_path (str)
    """
    @staticmethod
    def Deskew(input_path):
        # Load the image
        image = io.imread(input_path)

        # Convert the image to grayscale
        grayscale = rgb2gray(image)

        # Determine the skew angle of the image
        angle = determine_skew(grayscale)

        # Rotate the image to correct the skew and scale it back to 8-bit
        rotated = rotate(image, angle, resize=True) * 255
        rotated = rotated.astype(np.uint8)

        # Save the deskewed image back to the input path
        io.imsave(input_path, rotated)


if __name__ == "__main__":
    # Input directory can be provided as a command-line argument or hardcoded here
    input_directory = r'CVAT/KPJ'
    deskewer = DataAugmentation(input_directory)
    deskewer.processor()
