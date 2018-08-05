#import cv2
import awscam
from tensorflow-human-detection import DetectorAPI

THRESHOLD = 0.7
MODEL_PATH = './faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'


class DataParser:
    def __init__(self, opapi, model_path):
        self.odapi = odapi
        self.model_path = model_path

    def grab_image(self):
        ret, frame = awscam.getLastFrame()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return ret, jpeg

    def count_people(self):
        total_people = 0
        r, img = grab_image()
        img = cv2.resize(img, (1280, 720))
        boxes, scores, classes, num = odapi.processFrame(img)

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                total_people += 1

                # Visualization of the results of a detection.
                # box = boxes[i]
                # cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
        return total_people

    def is_it_lit(self, people):
        if people > 15:
            return "Its lit"
        elif people > 10:
            return "Its kinda lit"
        elif people > 5:
            return "its happening I guess"
        else:
            return "def not lit yo"


if __name__ == "__main__":
    odapi = DetectorAPI(MODEL_PATH)
    dp = DataParser(odapi, MODEL_PATH)
    return dp.count_people()
