import argparse
import os
import random
import tempfile
import shutil
import xml.etree.ElementTree as ET


def unpack_label_me(input_dir, output_dir):
    """
        This script helps to unpack downloaded labelme collections and moves all files in a single folder according to the following routine:
        1) Scan input folder and its subfolders for .tar.gz files
        2) Unpack .tar.gz files and move them to a temporary directory
        3) For each original collection (=each unzipped file):
        3a) Crawl all .xml files
        3b) Replace filename by a global counter (to avoid issues by similar filenames over different collections)
        3c) Save renamed .xml and image files in output dir
        4) Delete temporary folder

        Args:
            --inputDir  -i  Directory that will be crawled for .tar.gz files downloaded from labelme
            --outputDir -o  Directory where all unpacked .xml and .jpg files will be moved to

        Returns:
            Nothing

        Usage:
            python DataLoading.py -i [PATH_TO_DOWNLOAD_FOLDER] -o [TARGET_FOLDER]
        """

    os.makedirs(os.path.join(output_dir), exist_ok=True)

    # Initiate temporary directory
    tmpdir = tempfile.mkdtemp()
    print(f'Created tmpdir {tmpdir}')

    # Walk through inputDir, unzip all files and move to tempdir
    suffix = ".tar.gz"  # file extension to extract
    for folder, dirs, files in os.walk(input_dir, topdown=False, followlinks=False):
        for name in files:
            if name.endswith(suffix):
                shutil.unpack_archive(os.path.join(folder, name), os.path.join(tmpdir, os.path.splitext(name)[0]))
                print(f'Successfully unpacked {os.path.join(folder, name)}')

    file_counter = 0  # counter for filename
    # Walk toplevel (=unpacked collections) of tempdir
    for collection in os.listdir(tmpdir):
        if not os.path.isdir(os.path.join(tmpdir, collection)):
            continue

        # Crawl tempdir/collection, rename and move files to output folder
        suffix = ".xml"
        collection_dir = os.path.join(tmpdir, collection)
        rename_dict = {}  # Dict containing rename rules for image files

        # Grab .xml files, rename filename attribute
        for folder, dirs, files in os.walk(collection_dir, topdown=False, followlinks=False):
            for name in files:
                if name.endswith(suffix):
                    xmlfile = os.path.join(collection_dir, folder, name)
                    # open xml file
                    try:
                        tree = ET.parse(xmlfile)
                        root = tree.getroot()
                        original_file_name = root.find("filename").text
                        file_counter += 1
                    except:
                        print(f'Invalid annotation file {xmlfile}')
                        continue

                    # manipulate filename attribute
                    new_file_name = str(file_counter) + os.path.splitext(original_file_name)[1]
                    rename_dict[original_file_name] = new_file_name
                    tree.find("filename").text = new_file_name

                    # save manipulated xml file in output dir
                    output_path = os.path.join(output_dir, str(file_counter) + suffix)
                    tree.write(open(output_path, 'wb'), encoding='utf-8')

        # Grab image files and copy them to output dir (with new name!)
        for folder, dirs, files in os.walk(collection_dir, topdown=False, followlinks=False):
            for name in files:
                if name in rename_dict:
                    old_path = os.path.join(collection_dir, folder, name)
                    new_path = os.path.join(output_dir, rename_dict[name])
                    shutil.copy2(old_path, new_path)

        print(f'Finished renaming and copying for {collection_dir}')

    # Delete tempdir
    shutil.rmtree(tmpdir)
    print(f'Cleaned tmpdir {tmpdir}')
    print(f'Extraction completed. You find all image and annotation files in {output_dir}')


def train_test_valid_split(output_dir, weights):
    """
    Takes up work from unpack_label_me and assigns all files extracted to train, test and validation set randomly
    given the weights
    Files are moved to a folder for images (img) and one for annotations (anot)

    Args:
        output_dir (str): Directory where all unpacked files from LabelMe are saved
        weights (list): Weights for train, validation and test set (probability to be assigned in %)

    Returns:
        None

    """

    print(f"Split data into train, validation and test set, with the following shares [{weights[0]}, {weights[1]},"
          f" {weights[2]}]")

    # create directories for all sets
    list_split_folders = ["/train/", "/valid/", "/test/"]
    for fold in list_split_folders:
        os.makedirs(output_dir + fold, exist_ok=True)
        os.makedirs(output_dir + fold + 'img/', exist_ok=True)
        os.makedirs(output_dir + fold + 'anot/', exist_ok=True)

    # create a list of all image files
    list_img_files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]

    # go over all image files and draw a set to be assigned randomly
    for img_file in list_img_files:
        idx = random.choices(range(0, 3), weights=weights, k=1)[0]

        # move img and xml to the assigned directory
        # as for each image a corresponding xml exists, we can simply use the image file name for the xml as well
        shutil.move(output_dir + img_file, output_dir + list_split_folders[idx] + 'img/' + img_file)
        shutil.move(output_dir + img_file.replace('.jpg', '.xml'),
                    output_dir + list_split_folders[idx] + 'anot/' + img_file.replace('.jpg', '.xml'))

    print(f'Split is successfully performed. \n'
          f'You will find the sets in the sub-folders of {output_dir}')


def load_data(inputDir, outputDir, train=50, valid=20, test=30):
    """
    Serves as wrapper method for the following two tasks:
    1) unpack collections from LabelMe
    2) Split the unpacked files into train, test and validation sets

    Args:
        inputDir: Directory that will be crawled for .tar.gz files downloaded from labelme
        outputDir: Directory where all unpacked .xml and .jpg files will be moved to (make sure that it ends with a "/")
        train: Percentage of train images of total data set (default: 50)
        valid: Percentage of validation images of total data set (default: 20)
        test: Percentage of test images of total data set (default: 30)

    Returns:
        None
    """
    # Validate input args
    assert os.path.isdir(inputDir)

    # unpack all files from LabelMe
    unpack_label_me(inputDir, outputDir)

    # assign files to train, test and validation set
    weights = [train, valid, test]
    train_test_valid_split(outputDir, weights=weights)


def main():
    """
    Use this method with console arguments!
    Serves as wrapper method for the following two tasks:
    1) unpack collections from LabelMe
    2) Split the unpacked files into train, test and validation sets

    Args:
            --inputDir  -i  Directory that will be crawled for .tar.gz files downloaded from labelme
            --outputDir -o  Directory where all unpacked .xml and .jpg files will be moved to
            --train -train  Percentage of train images of total data set (default: 50)
            --valid -valid  Percentage of validation images of total data set (default: 20)
            --test -test    Percentage of test images of total data set (default: 30)

        Returns:
            Nothing

        Usage:
            python DataLoading.py -i [PATH_TO_DOWNLOAD_FOLDER] -o [TARGET_FOLDER] -train [TRAIN_SHARE]
                                    -valid [VALIDATION_SHARE] -test [TEST_SHARE]

    """
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Extract labelme downloads and merge into single folder"
    )
    parser.add_argument(
        "-i",
        "--inputDir",
        help="Path that shall be crawled",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--outputDir",
        help="Directory where all images and annotations shall be moved to",
        type=str,
    )

    parser.add_argument(
        "-train",
        "--train",
        help="share of test images in percent (e.g. 50)",
        type=int,
        default=50,
    )

    parser.add_argument(
        "-valid",
        "--valid",
        help="share of validation images in percent (e.g. 30)",
        type=int,
        default=20,
    )

    parser.add_argument(
        "-test",
        "--test",
        help="share of test images in percent (e.g. 20)",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    # Validate input args
    assert os.path.isdir(args.inputDir)

    # unpack all files from LabelMe
    unpack_label_me(args.inputDir, args.outputDir)

    # assign files to train, test and validation set
    weights = [args.train, args.valid, args.test]
    train_test_valid_split(args.outputDir, weights=weights)


if __name__ == "__main__":
    load_data(inputDir='/Users/Dany1/Desktop/LabelMe/',
              outputDir='/Users/Dany1/PycharmProjects/semester-challange/Pictures/')
