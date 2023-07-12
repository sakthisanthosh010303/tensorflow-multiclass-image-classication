# Author: Sakthi Santhosh
# Created on: 16/04/2023
#
# Tensorflow Multi-class Image Classification - Image Inference
def main(argv: list[str]) -> int:
    if not argv:
        print("Error: Program called with no data.")
        return 1

    from os import path

    if not path.exists(argv[0]):
        print("Error: File not found.")
        return 1

    from numpy import array, expand_dims
    from cv2 import (
        FONT_HERSHEY_SIMPLEX,
        imread,
        imshow,
        imwrite,
        putText,
        rectangle,
        waitKey
    )
    from tensorflow.image import resize
    from tensorflow.keras.models import load_model
    from time import time

    from constants import CLASSES, LABELS

    model = load_model("./model.h5")
    image_handle = imread(argv[0])

    start_time = time()
    result = model.predict(
        expand_dims(
            resize(image_handle, (256, 256)) / 255, axis=0
        ), verbose=None
    )
    time_delta = (time() - start_time) * 1000

    ypos = 45
    rectangle(
        image_handle,
        pt1=(0, 0),
        pt2=(265, 180),
        color=(0, 0, 0),
        thickness=-1
    )
    putText(
        image_handle,
        "Prediction (%.2f ms)"%(time_delta),
        (0, 20),
        FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )
    for index in range(CLASSES):
        putText(
            image_handle,
            "%s: %.2f%%"%(LABELS[index], result[0][index] * 100),
            (15, ypos),
            FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2
        )
        ypos += 25
    imshow("Image Classification", image_handle)
    waitKey(5000)

    imwrite("./output.jpg", image_handle)
    return 0

if __name__ == "__main__":
    from sys import argv

    exit(main(argv[1:]))
