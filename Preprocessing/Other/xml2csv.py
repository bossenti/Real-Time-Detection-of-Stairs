"""
Credits to
https://github.com/Tony607/object_detection_demo/blob/master/xml_to_csv.py
"""

import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
from termcolor import colored


def xml_to_csv(path, upDown):
    """
    This function grabs all .xml annotation in a given directory and combines them in a Pandas datagram
    .xml files must have at least following elements in this structure (labelme standard export):
        root
        |-filename
        |-imagesize
        |--nrows
        |--ncols
        |-object (for each object one)
        |--name
        |--polygon
        |---pt (4x)
        |----x
        |----y

    Args:
        path:           The path containing the .xml files
        upDown:         If true add up / down attribute to class name

    Returns:
        xml_df:         Pandas DataFrame containing extracted object information
        classes_names:  List containing all class names found in .xml files
    """

    #list storing all occuring classes
    classes_names = []

    #list storing tuples in the form of 'column_name' (see definiton below), each bounding box is represented by one list element
    xml_list = []
    
    #iterate through all .xml files
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        #basic picture information
        filename = root.find("filename").text

        try:
            height = int(root.find("imagesize")[0].text)
            width = int(root.find("imagesize")[1].text)
        except:
            print('Warning: No valid image size found in {0}.'.format(xml_file))
            height = 0
            width = 0

        #iterate through all all objects ("bounding boxes") in .xml file
        for member in root.findall("object"):
        
            objectClass = member.find('name').text

            #Optionally add up/down attribute to object class
            if upDown:
                try:
                    attribute = member.find('attributes').text
                    assert attribute == "up" or attribute == "down"
                    objectClass = objectClass + "_" + attribute
                except:
                    print('Warning: No valid attribute found in {0}.'.format(xml_file))

            classes_names.append(objectClass)

            #transform bounding box from ((x1,y1), (x2,y2), (x3,y3), (x4,y4)) to ((xmin, ymin), (xmax, ymax))
            x = []
            y = []
            for pt in member.find('polygon').findall('pt'):
                try:
                    x.append(int(pt[0].text))
                    y.append(int(pt[1].text))
                except:
                    print('Warning: No valid bounding box anchor found in {0}.'.format(xml_file))
                    x.append(0)
                    y.append(0)

            value = (
                filename,
                width,
                height,
                objectClass,                    
                min(x),         #xmin
                min(y),         #ymin
                max(x),         #xmax
                max(y),         #ymax
            )
            xml_list.append(value)

    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]

    #create, sort and return pandas dataframe containing all objects
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return xml_df, classes_names


def main():
    """
    This function grabs all .xml annotation in a given directory and combines them to a single, tensorflow compatible, .csv file

    Args:
        --inputDir      -i: The path containing the .xml files
        --outputFile    -o: Path where labels.csv file shall be created
        --labelMapDir   -l: (Optional) Path where label_map.pbtxt shall be created
        --upDown        -u: If True, add up / down attribute to class (default=False)

    Returns:
        labels.csv:         All object annotations found in .xml files in a single tensorflow compatible .csv file
        label_map.pbtxt:    File mapping class names to ids in a tensorflow compatible format
    
    Usage:
        python xml2csv.py -i [PATH_TO_XML_FOLDER]/train -o [TARGET_FOLDER]/train_labels.csv -l [TARGET_FOLDER]/label_map.pbtxt
    """

    #Initiate argument parser
    parser = argparse.ArgumentParser(
        description="LabelMe TensorFlow XML-to-CSV converter"
    )
    parser.add_argument(
        "-i",
        "--inputDir",
        help="Path to the folder where the input .xml files are stored",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--outputFile",
        help="Name of output .csv file (including path)",
        type=str
    )

    parser.add_argument(
        "-l",
        "--labelMapDir",
        help="Directory path to save label_map.pbtxt file is specified.",
        type=str,
        default="",
    )

    parser.add_argument(
        "-u",
        "--upDown",
        help="If True, add up/down attribute to class",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    #If no input args are given use current working directory
    if args.inputDir is None:
        args.inputDir = os.getcwd()
    if args.outputFile is None:
        args.outputFile = args.inputDir + "/labels.csv"

    #check if input and output path exists / can be created
    assert os.path.isdir(args.inputDir)
    os.makedirs(os.path.dirname(args.outputFile), exist_ok=True)

    #convert xml to pandas and further to csv
    xml_df, classes_names = xml_to_csv(args.inputDir, args.upDown)
    xml_df.to_csv(args.outputFile, index=None)
    print("Successfully converted xml to csv.")

    #Optionally (if path in args given) create tensorflow label_map.pbtxt file
    if args.labelMapDir:
        os.makedirs(args.labelMapDir, exist_ok=True)
        label_map_path = os.path.join(args.labelMapDir, "label_map.pbtxt")
        print("Generate `{}`".format(label_map_path))

        #Create the `label_map.pbtxt` file
        pbtxt_content = ""
        for i, class_name in enumerate(classes_names):
            pbtxt_content = (
                pbtxt_content
                + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(
                    i + 1, class_name
                )
            )
        pbtxt_content = pbtxt_content.strip()
        with open(label_map_path, "w") as f:
            f.write(pbtxt_content)


if __name__ == "__main__":
    main()