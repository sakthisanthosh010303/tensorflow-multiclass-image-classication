# Author: Sakthi Santhosh
# Created on: 16/04/2023
#
# Tensorflow Multi-class Image Classification - Video Inference
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
        destroyAllWindows,
        FONT_HERSHEY_SIMPLEX,
        imread,
        imshow,
        putText,
        rectangle,
        VideoCapture,
        waitKey
    )
    from tensorflow.image import resize
    from tensorflow.keras.models import load_model
    from time import time

    from constants import CLASSES, LABELS

    capture_handle = VideoCapture(argv[0])
    model = load_model("./model.h5")

    while True:
        start_time = time()
        success, frame = capture_handle.read()

        if not success:
            capture_handle.release()
            destroyAllWindows()
            break

        result = model.predict(
            expand_dims(
                resize(frame, (256, 256)) / 255, axis=0
            ), verbose=None
        )
        time_delta = (time() - start_time) * 1000

        ypos = 45
        rectangle(
            frame,
            pt1=(0, 0),
            pt2=(250, 180),
            color=(0, 0, 0),
            thickness=-1
        )
        putText(
            frame,
            "Prediction (%.2f ms)"%(time_delta),
            (0, 20),
            FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        for index in range(CLASSES):
            putText(
                frame,
                "%s: %.2f%%"%(LABELS[index], result[0][index] * 100),
                (15, ypos),
                FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2
            )
            ypos += 25
        imshow("Image Classification", frame)

        if waitKey(10) & 0xFF == ord('q'):
            capture_handle.release()
            destroyAllWindows()
            break
    return 0

if __name__ == "__main__":
    from sys import argv

    exit(main(argv[1:]))
