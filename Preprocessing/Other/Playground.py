import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from Preprocessing.DataPreprocessing import process_image_data

path = '/Pictures/Daniel/Images/users/serviceAnalytics/daniel'


def test_draw(image, annotation, write_image_file=None):
    """
    A debugging method to plot images and bounding boxes. Nonessential to the project.

    Args:
        image: The image to be drawn
        annotation: The annotations

    Returns:
        None

    """

    def draw_box(boxes, image, write_image_file=None):
        for i in range(0, len(boxes)):
            # changed color and width to make it visible
            cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)
        cv2.imshow("img", image)
        if write_image_file is not None:
            write_image_file = '/Users/Dany1/PycharmProjects/semester-challange/Preprocessing/' + write_image_file
            cv2.imwrite(write_image_file, image)
        #cv2.waitKey(0)
        cv2.waitKey()
        cv2.destroyAllWindows()

    drawing_boxes = []

    for box in annotation:
        x_coords = [box['x0'], box['x1'], box['x2'], box['x3']]
        y_coords = [box['y0'], box['y1'], box['y2'], box['y3']]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        drawing_boxes.append([1, 0, x_min, y_min, x_max, y_max])

    draw_box(drawing_boxes, image, write_image_file)


def show_image(images):
    img_path = images[32]
    image = cv2.imread(os.path.join(path, img_path))

    plt.imshow(image)
    plt.show()


def drawBox(boxes, image):
    for i in range(0, len(boxes)):
        # changed color and width to make it visible
        cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cvTest(image):
    # imageToPredict = cv2.imread("img.jpg", 3)
    imageToPredict = cv2.imread(os.path.join(path, image), 3)
    imageToPredict = cv2.imread('/Pictures/Daniel/Images/users/serviceAnalytics/daniel/img_3548.jpg', 3)
    print(imageToPredict.shape)

    # Note: flipped comparing to your original code!
    # x_ = imageToPredict.shape[0]
    # y_ = imageToPredict.shape[1]
    y_ = imageToPredict.shape[0]
    x_ = imageToPredict.shape[1]

    targetSize = 416
    x_scale = targetSize / x_
    y_scale = targetSize / y_
    print(x_scale, y_scale)
    img = cv2.resize(imageToPredict, (targetSize, targetSize));
    print(img.shape)
    img = np.array(img);

    # original frame as named values
    (origLeft, origTop, origRight, origBottom) = (0, 4019, 3018, 1991)

    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))
    # Box.drawBox([[1, 0, x, y, xmax, ymax]], img)
    drawBox([[1, 0, x, y, xmax, ymax]], img)


if __name__ == '__main__':
    a_path = r'E:\BwSyncAndShare\Master\Service Analytics\Groupwork\code\semester-challange\Preprocessing\anot_data'
    i_path = r'E:\BwSyncAndShare\Master\Service Analytics\Groupwork\code\semester-challange\Preprocessing\img_data'
    process_image_data(i_path, a_path, 416, yolo=True)

