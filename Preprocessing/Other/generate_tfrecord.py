"""
Credits to
https://github.com/Tony607/object_detection_demo/blob/master/xml_to_csv.py
"""

import os
import io
import pandas as pd
import tensorflow as tf
import sys
import argparse
from PIL import Image

sys.path.append("../../models/research")
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from object_detection.utils import label_map_util


def split(df, group):
    """
    Groups label dataframe by filename

    Args:
        df:     pandas dataframe
        group:  column name for df.groupby(group)

    Returns:
        [data]: [(filename, objects)]
    """

    data = namedtuple("data", ["filename", "object"]) #initiate "data" tyoe
    gb = df.groupby(group) #group df by group attribute
    return [
        data(filename, gb.get_group(x))
        for filename, x in zip(gb.groups.keys(), gb.groups)
    ]

def create_tf_example(group, path, label_map):
    """
    Merges a image file and its corresponding labels into a tf.train.Example() object.

    Args:
        group:      image filename
        path:       path to image folder
        label_map:  

    Returns:
        tf_example: tf.train.Example() object that can be serialized and added to TFRecord
    """

    #load image and extract attributes (width, height, filename)
    with tf.gfile.GFile(os.path.join(path, "{}".format(group.filename)), "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"jpg"
    
    #tf.train.Example() expects several objects in lists
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        #Extract bounding box
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)

        #Extract class name and retrieve class id
        #classes_text.append(row["class"].encode("utf8"))
        class_index = label_map.get(str(row["class"]))
        
        #Check if class id could be retrieved
        assert (
            class_index is not None
        ), "class label: `{}` not found in label_map: {}".format(
            row["class"], label_map
        )

        #For troubleshooting only
        print(f"{filename} has class_index {class_index} and class {row['class']}")

        classes.append(class_index)

    #Build tf_example object
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": dataset_util.int64_feature(height),
                "image/width": dataset_util.int64_feature(width),
                "image/filename": dataset_util.bytes_feature(filename),
                "image/source_id": dataset_util.bytes_feature(filename),
                "image/encoded": dataset_util.bytes_feature(encoded_jpg),
                "image/format": dataset_util.bytes_feature(image_format),
                "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
                "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
                "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
                "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
                "image/object/class/text": dataset_util.bytes_list_feature(
                    classes_text
                ),
                "image/object/class/label": dataset_util.int64_list_feature(classes),
            }
        )
    )
    return tf_example


def main():
    """
    Converts image folder, labels.csv and label_map.pbtxt into a TFRecord file.
    Attention: Depends on Tensorflow Object Detection API
        1) https://github.com/tensorflow/models/research must be in ../models/research
        2) CD to ../models/research
        3) Run protoc object_detection/protos/*.proto --python_out=.
        (For more info visit https://github.com/tensorflow/models/issues/1591)

    Args:
        --csvInput      -c: Path to the labels.csv file (make sure it's UTF-8 encoded!)
        --labelMap      -l: Path to the label_map.pbtxt file (make sure it's UTF-8 encoded!) 
        --images        -i: Path to the label_map.pbtxt file
        --outputFile    -o: Path to output TFRecord file

    Returns:
        tf_example: tf.train.Example() object that can be serialized and added to TFRecord
    """

    #Initiate argument parser
    parser = argparse.ArgumentParser(
        description="LabelMe TensorFlow XML-to-CSV converter"
    )
    parser.add_argument(
        "-c",
        "--csvInput",
        help="Path to the labels.csv file",
        type=str,
    )

    parser.add_argument(
        "-l",
        "--labelMap",
        help="Path to the label_map.pbtxt file",
        type=str,
    )

    parser.add_argument(
        "-i",
        "--images",
        help="Path to image folder",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--outputFile",
        help="Path to output TFRecord file",
        type=str
    )

    args = parser.parse_args()

    #If no input args are given use current working directory
    if args.csvInput is None:
        args.csvInput = os.getcwd() + "/labels.csv"
    if args.labelMap is None:
        args.labelMap = os.getcwd() + "/label_map.pbtxt"
    if args.images is None:
        args.images = os.getcwd()
    if args.outputFile is None:
        args.outputFile = os.getcwd() + "/train.record"

    #check if input paths exists
    assert os.path.isdir(args.images)
    assert os.path.isfile(args.csvInput)
    assert os.path.isfile(args.labelMap)

    #Initiate TFRecordWriter
    writer = tf.io.TFRecordWriter(args.outputFile)
    
    #Read labels from .csv into pd dataframe
    labels = pd.read_csv(args.csvInput)

    #Load the `label_map` from pbtxt file.
    label_map = label_map_util.load_labelmap(args.labelMap)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
    label_map = {} #Dict resolving class name to class id
    for k, v in category_index.items():
        label_map[v.get("name")] = v.get("id")

    #Group labels dataframe by filename
    grouped = split(labels, "filename")

    #for each filename
    for group in grouped:
        #create a tf_example for each image including all labels
        tf_example = create_tf_example(group, args.images, label_map)
        writer.write(tf_example.SerializeToString())

    #Close TFRecordWriter and save to file
    writer.close()
    output_path = os.path.join(os.getcwd(), args.outputFile)
    print("Successfully created the TFRecords: {}".format(args.outputFile))


if __name__ == "__main__":
    main()