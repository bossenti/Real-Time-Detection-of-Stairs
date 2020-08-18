import numpy as np
import random
import math
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


def extract_xml_information(xml_path, stairs_only):
    """
    This function extracts the necessary information (label and coordinates for all bounding boxes) from the
    corresponding '.xml' file.

    Args:
        xml_path: The path to the xml file
        stairs_only: If True, then change up/down label to simply 'stairs'

    Returns:
        boxes: A list of bounding boxes that contains a dictionary for each box with the following keys:
            label - The label of the bounding box (up or down)
            x0, y0, x1, y1, x2, y2, x3, y3 - The coordinates of the four corner points of the rectangular box
    """
    # Create an element tree to query the '.xml' file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # List for all bounding boxes of the file
    boxes = []
    # Iterate over all bounding boxes
    for i, box in enumerate(root.findall('.//object')):
        # Create a dictionary for the bounding box information
        box_info = dict()
        # Extract the label
        box_info['label'] = box.find('.//attributes').text
        # Make sure Label is correct, if not ignore picture
        if box_info['label'] not in ['None', 'up', 'down', None, '']:
            return 'error'
        # If stairs only, then change up/down to stairs
        if stairs_only and box_info['label'] in ('up', 'down'):
            box_info['label'] = 'stairs'
        # Iterate over the 4 edge points and extract the coordinates
        for j, point in enumerate(box.findall('.//pt')):
            # If the bounding box is not well-defined, then ignore picture
            if point.find('.//x').text != 'NaN' and point.find('.//y').text != 'NaN':
                box_info[f'x{j}'] = int(point.find('.//x').text)
                box_info[f'y{j}'] = int(point.find('.//y').text)
            else:
                return 'error'
        # Append the bounding box to the list of boxes
        boxes.append(box_info)

    return boxes


def resize_image(image, annotation, target_size):
    """
    This function resizes an image to the target size (target width = target height). The bounding box coordinates are
    scaled to fit the new image.

    Args:
        image: A image as a OpenCV object
        annotation: All the information on bounding boxes and labels (see function 'extract_xml_information')
        target_size: The target size of the image in pixel

    Returns:
        img: The resized image as a numpy matrix
        annotation: The annotations with scaled coordinates

    """
    # Extract the height and width of the images (in pixels)
    y_size = image.shape[0]
    x_size = image.shape[1]

    # Calculate the scaling factor to match the coordinates with the target size
    x_scale = target_size / x_size
    y_scale = target_size / y_size

    # Resize the image
    img = cv2.resize(image, (target_size, target_size))
    # Create a numpy matrix from the image
    img = np.array(img)

    # Scale the coordinates for each bounding box
    for box in annotation:
        box['x0'] = int(np.round(box['x0'] * x_scale))
        box['x1'] = int(np.round(box['x1'] * x_scale))
        box['x2'] = int(np.round(box['x2'] * x_scale))
        box['x3'] = int(np.round(box['x3'] * x_scale))
        box['y0'] = int(np.round(box['y0'] * y_scale))
        box['y1'] = int(np.round(box['y1'] * y_scale))
        box['y2'] = int(np.round(box['y2'] * y_scale))
        box['y3'] = int(np.round(box['y3'] * y_scale))

    return img, annotation


def standardize_box(annotation):
    """
    This method changes the bounding box coordinates if they are not standardized. As standard the point x0y0 has to be top left.

    Args:
        annotation: All the information on bounding boxes and labels (see function 'extract_xml_information')

    Returns:
        annotation: The annotations with changed coordinates of thr bounding box
    """

    # Change the coordinates of the bounding boxes
    for box in annotation:
        all_x = [box['x0'], box['x1'], box['x2'], box['x3']]
        all_y = [box['y0'], box['y1'], box['y2'], box['y3']]

        box['x0'] = min(all_x)
        box['x1'] = max(all_x)
        box['x2'] = max(all_x)
        box['x3'] = min(all_x)
        box['y0'] = min(all_y)
        box['y1'] = min(all_y)
        box['y2'] = max(all_y)
        box['y3'] = max(all_y)

    return annotation


def convert_to_yolo(img_data, img_width, img_height, stairs_only, target_dir, output_prefix=''):
    """
    This function prepares the data to train the yolo network in the darknet format.
    Yolo needs a text file (with the same name as the corresponding image file) that contains information about the
    bounding box class, the center coordinates and the height and width of the bounding box. This method creates a yolo
    folder in the target directory and stores all files and images in there.

    Args:
        img_data: the image data tuple (numpy matix, annotation) from the load_image_data method
        img_width: the absolute width of the image
        img_height: the absolute height of the image
        stairs_only: if True, then only 'stairs' is a label, instead of 'up' and 'down'
        target_dir: Path to the output folder
        output_prefix: Defines which set is currently processed (Train, Valid, or Test)

    Returns:
        None

    """
    # Create yolo folder (if it does not exist already)
    Path(target_dir + f'/{output_prefix}YoloData').mkdir(exist_ok=True)

    # Define labels and their class integer
    labels = ['None', 'up', 'down']
    # Consider stairs_only flag
    if stairs_only:
        labels = ['None', 'stairs']
    dict_label_to_class = {label: i for i, label in enumerate(labels)}
    # Write label file
    with open(target_dir + f'/{output_prefix}YoloData/_darknet.labels', 'w') as file:
        file.write('\n'.join(labels))

    # Iterate over image data
    for i, (img, annotation) in enumerate(img_data):
        yolo_annotation = []
        # Convert all annotations in the darknet format
        for box in annotation:
            box_class = dict_label_to_class[str(box['label'])]
            box_center_x = ((box['x0'] + box['x1']) / 2) / img_width
            box_center_y = ((box['y0'] + box['y2']) / 2) / img_height
            box_width = (box['x1'] - box['x0']) / img_width
            box_height = (box['y2'] - box['y0']) / img_height
            box_annotation = f'{box_class} {box_center_x} {box_center_y} {box_width} {box_height}\n'
            yolo_annotation.append(box_annotation)

        # Write annotation file
        with open(target_dir + f'/{output_prefix}YoloData/yolo{i}.txt', 'w') as file:
            file.writelines(yolo_annotation)

        # Write corresponding image file
        cv2.imwrite(target_dir + f'/{output_prefix}YoloData/yolo{i}.jpg', img)


def crop_image(image, annotation, size):
    """
    This method creates a new image that crops out the old image in the given size (height = width). The bounding box coordinates are
    changed depended on the new image size.

    Args:
        image: A image as a OpenCV object
        annotation: All the information on bounding boxes and labels (see function 'extract_xml_information')
        size: The required image size (height = width) in pixels

    Returns:
        img_crop: The new image that crops out the old image in the given size
        new_annotation: The annotations with new coordinates

    """
    # Choose random start coordinate on the xaxis and calculate other coordinates depend on this
    x1 = random.randint(0, image.shape[1] - size)
    x2 = x1 + size
    y1 = x1
    y2 = x2

    # Crop the image out the old image in the range x1:x2 (row) and y1:y2 (column)
    img_crop = image[y1:y2, x1:x2]
    new_annotation = annotation
    # Update the coordinates for each bounding box
    for box in new_annotation:
        # The case if the bounding box is out of the new image
        if x2 < box['x0'] or y2 < box['y0'] or x1 > box['x2'] or y1 > box['y2']:
            box['label'] = "None"
            box['x0'] = 0
            box['x1'] = 0
            box['x2'] = 0
            box['x3'] = 0
            box['y0'] = 0
            box['y1'] = 0
            box['y2'] = 0
            box['y3'] = 0

        else:
            box['x0'] = max(box['x0'] - x1, 0)
            box['x3'] = max(box['x0'] - x1, 0)

            if x2 > box['x2']:
                box['x1'] = max(0, box['x1'] - x1)
                box['x2'] = max(0, box['x2'] - x1)

            else:
                box['x1'] = size
                box['x2'] = size

            box['y0'] = max(box['y0'] - y1, 0)
            box['y1'] = max(box['y1'] - y1, 0)

            if y2 > box['y2']:
                box['y2'] = max(0, box['y2'] - y1)
                box['y3'] = max(0, box['y3'] - y1)
            else:
                box['y2'] = size
                box['y3'] = size

    return img_crop, new_annotation


def rotate_point(center_point, point, angle):
    """This method rotates a point around another centerPoint. Angle is in degrees.

    Args:
        center_point: The center point of the image
        point: The point to be new donate
        angle: The angle of the rotation. angle > 0 for counter-clockwise and angle < 0 for clockwise rotation

    Returns:
        new_point: The new coordinates of the point
    """
    # Transform the angle into radians
    angle = math.radians(-angle)

    # Calculate new coordinates of the point
    new_point = point[0] - center_point[0], point[1] - center_point[1]
    new_point = (new_point[0] * math.cos(angle) - new_point[1] * math.sin(angle),
                 new_point[0] * math.sin(angle) + new_point[1] * math.cos(angle))
    new_point = new_point[0] + center_point[0], new_point[1] + center_point[1]

    return new_point


def rotate_image(image, annotation, target_size, angle):
    """
    This method creates a new image by the rotating of the origin image. The bounding box coordinates are
    changed depened on the angle of the rotation.

    Args:
        image: A image as a OpenCV object
        annotation: All the information on bounding boxes and labels (see function 'extract_xml_information')
        target_size: The image size (height = width) in pixels for calculating of the image center
        angle: The angle of the rotation. angle > 0 for counter-clockwise and angle < 0 for clockwise rotation

    Returns:
        rot_image: The new rotated image
        rot_annotation: The annotations with new coordinates

    """

    # Denote the image center
    center_point = (target_size / 2, target_size / 2)

    rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, 1)
    rot_image = cv2.warpAffine(image, rotation_matrix, (target_size, target_size))

    rot_annotation = annotation

    # Update the coordinates for each bounding box
    for box in rot_annotation:

        # Calculate new coordinates of the points for the rotated image
        x0, y0 = rotate_point(center_point, (box['x0'], box['y0']), angle)
        x1, y1 = rotate_point(center_point, (box['x1'], box['y1']), angle)
        x2, y2 = rotate_point(center_point, (box['x2'], box['y2']), angle)
        x3, y3 = rotate_point(center_point, (box['x3'], box['y3']), angle)

        coordinates = [x0, x1, x2, x3, y0, y1, y2, y3]

        # The case if the bounding box is out of the new image
        for coordinate in coordinates:
            coordinate = min(299, coordinate)
            coordinate = max(0, coordinate)

        # Counter-clockwise rotation
        if angle > 0:
            box['x0'] = int(np.round(x0))
            box['x1'] = int(np.round(x2))
            box['x2'] = int(np.round(x2))
            box['x3'] = int(np.round(x0))
            box['y0'] = int(np.round(y1))
            box['y1'] = int(np.round(y1))
            box['y2'] = int(np.round(y3))
            box['y3'] = int(np.round(y3))

        # Clockwise rotation
        else:
            box['x0'] = int(np.round(x3))
            box['x1'] = int(np.round(x1))
            box['x2'] = int(np.round(x1))
            box['x3'] = int(np.round(x3))
            box['y0'] = int(np.round(y0))
            box['y1'] = int(np.round(y0))
            box['y2'] = int(np.round(y2))
            box['y3'] = int(np.round(y2))

    return rot_image, rot_annotation

def validate_box(image, annotation):
    """
    This method checks if a bounding box is valid. A valid bounding box is completely inside the picture.

    Args:
        image: A image as a OpenCV object
        annotation: All the information on bounding boxes and labels (see function 'extract_xml_information')

    Returns:
        valid: boolean value (true = bounding box is valid)

    """
    y_size = image.shape[0]
    x_size = image.shape[1]

    for box in annotation:
        if box['x0'] < 0 or box['x0'] > x_size:
            return False
        if box['x1'] < 0 or box['x1'] > x_size:
            return False
        if box['x2'] < 0 or box['x2'] > x_size:
            return False
        if box['x3'] < 0 or box['x3'] > x_size:
            return False

        if box['y0'] < 0 or box['y0'] > y_size:
            return False
        if box['y1'] < 0 or box['y1'] > y_size:
            return False
        if box['y2'] < 0 or box['y2'] > y_size:
            return False
        if box['y3'] < 0 or box['y3'] > y_size:
            return False
    
    return True


############## Tensorflow records section ######################
def convert_to_tf_records(img_data, img_width, img_height, stairs_only, target_dir, output_prefix=''):
    """
    This method converts the data into the tf_reecords format and stores it in a new folder called "TFRecordsData"

    Args:
        img_data: The preprocessed image data
        img_width: image width
        img_height: image height
        stairs_only: Flag, if we only want to label stairs instead of up/down
        target_dir: Path to the output folder
        output_prefix: Defines which set is currently processed (Train, Valid, or Test)

    Returns:
        None

    """
    # Create TF Records folder (if it does not exist already)
    Path(f'{target_dir}/{output_prefix}TFRecordsData').mkdir(exist_ok=True)
    Path(f'{target_dir}/{output_prefix}TFRecordsData/Annotations').mkdir(exist_ok=True)
    Path(f'{target_dir}/{output_prefix}TFRecordsData/Images').mkdir(exist_ok=True)

    # Define labels and their class integer
    labels = ['None', 'up', 'down']
    # Consider stairs_only flag
    if stairs_only:
        labels = ['None', 'stairs']
    dict_label_to_class = {label: i for i, label in enumerate(labels)}

    # Write label file
    with open(f'{target_dir}/{output_prefix}TFRecordsData/Annotations/label_map.pbtxt', 'w') as file:
        pbtxt_content = ""
        for i, class_name in enumerate(labels):
            if i == 0:
                continue
            pbtxt_content = pbtxt_content + f"item {{\n    id: {i}\n    name: '{class_name}'\n}}\n\n"
        pbtxt_content = pbtxt_content.strip()
        file.write(pbtxt_content)

    tf_annotation_df = pd.DataFrame(columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
    # Iterate over image data
    for i, (img, annotation) in enumerate(img_data):
        filename = f'tf_records{i}.jpg'
        # Convert all annotations in the tf records format
        for box in annotation:
            width = img_width
            height = img_height
            try: 
                _class = dict_label_to_class[box['label']]
            except:
                continue
            x = [box['x0'], box['x1'], box['x2'], box['x3']]
            y = [box['y0'], box['y1'], box['y2'], box['y3']]
            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)
            tf_annotation_df = tf_annotation_df.append({
                'filename': filename,
                'width': width,
                'height': height,
                'class': box['label'],
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }, ignore_index=True)

        # Write corresponding image file
        cv2.imwrite(f'{target_dir}/{output_prefix}TFRecordsData/Images/{filename}', img)

    tf_annotation_df.to_csv(f'{target_dir}/{output_prefix}TFRecordsData/Annotations/labels.csv', index=False)
