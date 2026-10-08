from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_name="yolo11n.pt"):
        self.model = YOLO(model_name)

    # Get colour image from camera and send to YOLO for detection
    def detect(self, color_image, confidence=0.5):
        results = self.model(
            color_image,
            conf = confidence,
            verbose = False
        )

        objects = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist() # Bounding box (x1 = left, y1 = top, x2 = right, y2 = bottom)

                confidence_score = float(box.conf[0])
                class_id = int(box.cls[0])

                class_name = self.model.names[class_id]

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                objects.append({
                    "name": class_name,
                    "confidence": confidence_score,
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),
                    "center": (
                        center_x,
                        center_y
                    )
                })

        return objects