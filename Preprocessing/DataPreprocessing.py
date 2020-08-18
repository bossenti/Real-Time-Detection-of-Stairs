import cv2
import os
import logging
import pickle
from copy import deepcopy
from pathlib import Path

from Preprocessing.PreprocessingUtilities import crop_image, rotate_image, rotate_point, convert_to_yolo, resize_image,\
    extract_xml_information, standardize_box, convert_to_tf_records, validate_box
from Preprocessing.Other.Playground import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Preprocessing')


def process_image_data(img_path, ann_path, target_dir, target_size, stairs_only=False, grey=False, crop=False, crop_size=100,
                       crop_number=5, rotate=False, rot_angle=45, yolo=False, tf_records=False, output_prefix='Train'):
    """
    This is the main method of the data preprocessing step. It gathers all images and annotations available and turns
    them from '.jpg' and '.xml' files into readable input for the models in the form of numpy matrices and dictionaries.
    The resulting data set is either stored as a pickle file or converted into the darknet or TFRecords format.

    Args:
        img_path: Path to the folder containing all images
        ann_path: Path to the folder containing all annotations
        target_dir: Path to the output folder
        target_size: The required image size (height = width) in pixels
        stairs_only: If True, only use stairs label, not up/down addition (default False)
        grey: If True, then use Greyscale image
        crop: If True, a new image will be created that crops out the old image (default False)
        crop_size: the size of croped images  (default 200)
        crop_number: the number of new images to crop (default 5)
        rotate: If True, a new image will be created that rotate the original image by angle degree (default False) 
        rot_angle: Degree for the image rotation, degree > 0 for counter-clockwise and degree < 0 for clockwise rotation (default 45)
        yolo: If True, then convert annotations to yolo-readable darknet format
        tf_records: If True, then convert annotations to tensorflow-readable tf_records format (Faster R-CNN, SSD)
        output_prefix: Defines which set is currently processed (Train, Valid, or Test)

    Returns:
        None

    """
    # Create target dir
    Path(target_dir).mkdir(exist_ok=True)

    # Get the file names for all images
    image_file_names = [image for image in os.listdir(img_path)]
    logger.info(f'starting preprocessing for {len(image_file_names)} images')
    if crop:
        logger.info(f'cropping is activated with {crop_number} crops per image')
    if rotate:
        logger.info(f'rotation is activated')

    # Create a list to store the processed images in
    image_data = []

    # Iterate over all images
    for image_file_name in image_file_names:
        # Extract the image name, without the file extension
        image_id = image_file_name[:-3]
        # Get the corresponding annotation file for the image
        annotation_path = os.path.join(ann_path, (image_id + 'xml'))
        # Extract the annotation information from the '.xml' file
        annotation = extract_xml_information(annotation_path, stairs_only)
        # Ignore image in case of annotation error
        if annotation == 'error':
            logger.warning(f'the annotation for the image "{image_file_name}" contains an error')
            continue
        # Read the image file with OpenCV
        if grey:
            image = cv2.imread(os.path.join(img_path, image_file_name), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(os.path.join(img_path, image_file_name), cv2.IMREAD_COLOR)
        # Resize the image and the coordinates to target size, convert image to numpy matrix
        image, annotation = resize_image(image, annotation, target_size)

        # Standardize the bounding box coordinates so that x0y0 is always placed top left
        annotation = standardize_box(annotation)

        # Check bounding box
        if not validate_box(image, annotation):
            logger.warning(f'Bounding box for "{image_file_name}" exceeds limits')
            continue

        # Add the image and corresponding annotation as a tuple to the list of images
        image_data.append((image, annotation))

        # Create new image and the coordinates if crop = True
        if crop:
            for i in range(0, crop_number):

                # Copy the image and the annotation for crop method
                anno_copy = deepcopy(annotation)
                image_copy = deepcopy(image)

                # Crop new image and store new image and new annotation
                croping = crop_image(image_copy, anno_copy, crop_size)

                # Resize the croped image and the coordinates to target size, convert image to numpy matrix
                croped_image, croped_annotation = resize_image(croping[0], croping[1], target_size)

                # Check bounding box
                if validate_box(image, annotation) == False:
                    continue

                # Add the image and corresponding annotation as a tuple to the list of images
                image_data.append((croped_image, croped_annotation))

        # Create new image and the coordinates if rotate = True
        if rotate:

            # Copy the image and the annotation for crop method
            anno_copy = deepcopy(annotation)
            image_copy = deepcopy(image)      
            
            # Rotate the copy_image and store new image and new annotation
            rot_image, rot_annotation = rotate_image(image_copy, anno_copy, target_size, rot_angle)

            # Check bounding box
            if validate_box(image, annotation) == False:
                continue

            # Add the image and corresponding annotation as a tuple to the list of images
            image_data.append((rot_image, rot_annotation))

    # Store the data in the yolo format
    if yolo:
        convert_to_yolo(image_data, target_size, target_size, stairs_only, target_dir, output_prefix)
        logger.info('converted images to darknet format')

    # Store the data in the tf_records format
    if tf_records:
        convert_to_tf_records(image_data, target_size, target_size, stairs_only, target_dir, output_prefix)
        logger.info('converted images to tf_records format')

    # Store the complete data set as a pickle file
    else:
        pickle.dump(image_data, open('image_data.p', 'wb'))

    logger.info(f'preprocessing done for {len(image_data)} images')


if __name__ == '__main__':
    # If you used the Data Loading function first, then you only have to specify the input_dir and target_dir here
    input_dir = '/Users/Dany1/PycharmProjects/semester-challange/Pictures'
    target_dir = '/Users/Dany1/PycharmProjects/semester-challange/ProcessedData'
    for dir in ['train', 'valid', 'test']:
        a_path = input_dir + '/' + dir + '/' + 'anot'
        i_path = input_dir + '/' + dir + '/' + 'img'
        process_image_data(i_path, a_path, target_dir, 300, crop=True, rotate=True, stairs_only=False, tf_records=True,
                           yolo=True, output_prefix=dir, grey=False)
