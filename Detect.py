# Conduct object detection utilizing YOLOv8, a pre-trained model from PyTorch.

import os
import cv2
import torch
import argparse
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import set_logging
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator


class Detect:
    """
    Execute trained weights to detect the table as well as annotate the Regions of Interest (ROI) of the image.
    @param opt
    @return img_saved_paths
    """
    @staticmethod
    def detect(opt):
        source = opt.source
        imgsz = opt.img_size
        weights = opt.weights
        save_txt = opt.save_txt

        # Directory information
        project = opt.project
        name = opt.name
        img_name = opt.img_name
        ext = opt.ext

        # Directories
        save_dir = Path(project) / name  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = YOLO(weights)
        results = model.predict(source=source, save_txt=save_txt, imgsz=imgsz, half=half,
                                save=True, conf=opt.conf_thres, iou=opt.iou_thres)

        # Get class name(s) from the YOLO model
        target_name = model.module.names if hasattr(model, 'module') else model.names

        # Load original image
        ori_img = cv2.imread(f'{save_dir}/{img_name}.{ext}')
        gn = torch.tensor(ori_img.shape)[[1, 0, 1, 0]]  # # Normalization gain whwh

        # Process the detection results
        result = results[0]
        cls = result.boxes.cls
        conf = result.boxes.conf

        xyxy = result.boxes.xyxy.to(device)  # Convert detected bounding boxes to the chosen device

        if xyxy.numel() > 0:  # Check if the tensor is not empty
            gn = gn.to(device)  # Move the normalization gain tensor to the chosen device
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(-1, 4)) / gn).view(-1).tolist()  # normalized xywh
            lines = (cls, *xywh, conf) if opt.save_conf else (cls, xywh)  # label format

            tensor_values = lines[0]
            list_values = lines[1]

            if save_txt:
                try:
                    with open(f'{save_dir}/labels/table_boxes.txt', 'a') as f:
                        # Write the first element of the tuple (tensor) as a float
                        f.write('%g ' % tensor_values.item())  # Extract the float value from the tensor and write it

                        # Write the second element as the number of page
                        f.write('%g ' % int(img_name.split('_')[-1]))  # Write each number followed by a space
                        print(f'int(img_name[-1]): {int(img_name.split("_")[-1])}')
                        # page_numbers = [string.split('_')[-1] for string in strings]

                        # Write the third element of the tuple (list) as a series of floats
                        for number in list_values:
                            f.write('%g ' % number)  # Write each number followed by a space

                        f.write('\n')  # Add a newline character at the end of the line
                except (ValueError, RuntimeError):
                    with open(f'{save_dir}/labels/row_boxes.txt', 'w') as f:
                        # Loop through the values
                        for i in range(len(list_values) // 4):
                            start = i * 4
                            end = start + 4
                            f.write(f"{int(tensor_values[i])} {' '.join(map(lambda x: f'{x:.6f}', list_values[start:end]))}\n")

            # Check if the tensor has only one row
            if xyxy.dim() == 2 and xyxy.size(0) == 1:
                # Save cropped image
                crop_img_name = f'{save_dir}/{img_name}_crop.png'  # File path for the cropped image
                xyxy_list = xyxy.cpu().numpy().flatten()
                # Assuming the tensor represents two points (x1,y1) and (x2,y2) of the bounding box
                x1, y1, x2, y2 = xyxy_list

                crop_img = ori_img[int(y1):int(y2), int(x1):int(x2)]  # Cropping the image based on the coordinates
                cv2.imwrite(crop_img_name, crop_img)

                print(f'Crop Image Name: {crop_img_name}')

            annotated_img = None
            for box, x2y2 in zip(result.boxes, xyxy):
                cls = box.cls
                conf = box.conf

                label = f'{target_name[int(cls)]} {float(conf):.2f}'  # Text to display beside of the box
                x2y2 = np.array(x2y2.tolist()).ravel()  # Flatten the numpy array
                annotator = Annotator(ori_img)
                annotator.box_label(x2y2, label, (0, 0, 255))  # Add one xyxy box to image with label
                annotated_img = annotator.result()  # Return annotated image as array

            # print(f'Image name: {img_name}')

            det = 'table' if Path(weights).stem.lower() == 'table' else 'row'
            annotated_img_name = f"{save_dir}/{img_name[:-5] if det=='row' else img_name}_{det}.png"
            print(f'annotated_img_name: {annotated_img_name}')
            cv2.imwrite(annotated_img_name, annotated_img)

            print(f'Image name: {img_name}')

    '''
    Define the required arguments to command-line interfaces.
    @param saved_path
    @param image_name
    '''
    @staticmethod
    def parseOpt(saved_path, image_name, best_weight, conf):
        parser = argparse.ArgumentParser()

        print(f'Saved path: {saved_path}')
        print(f'Image name: {image_name}')

        parser.add_argument('--weights', nargs='+', type=str, default=f'C:/Users/ChiaChungLim/PycharmProjects/Consignment-Itemized-Data/model/{best_weight}', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=f'{saved_path}/{image_name}.png', help='source')
        parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=conf, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', default=True, help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=os.path.split(saved_path)[0], help='save results to project/name')
        parser.add_argument('--name', default=os.path.split(saved_path)[1], help='save results to project/name')
        parser.add_argument('--img-name', default=image_name, help='image name in project/name')
        parser.add_argument('--ext', default='png', help='default image extension is *.png')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)

        Detect.detect(opt)
