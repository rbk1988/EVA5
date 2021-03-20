"""Predict using yolo v3."""
import cv2
import numpy as np


class YoloDetector:
    """."""

    def __init__(self, yolo_weights_path, yolo_cfg_path, coco_names_path):
        """Initialise Yolo Detector."""
        self.net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
        classes = []
        with open(coco_names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        self.classes = classes
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [
            self.layer_names[i[0] - 1]
            for i in self.net.getUnconnectedOutLayers()
        ]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def get_image_predictions(self, image_path):
        """Get predictions using yolo model."""
        # Loading image
        img = cv2.imread(image_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(
            img,
            0.00392,
            (416, 416),
            (0, 0, 0),
            True,
            crop=False
        )

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

        return img

