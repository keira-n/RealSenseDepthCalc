import cv2

from camera import RealSenseCamera
from detector import ObjectDetector
from distance import DistanceCalculator


def main():
    camera = RealSenseCamera()
    detector = ObjectDetector()
    distance_calculator = DistanceCalculator()

    camera.start()

    print("RealSense camera connected")
    print("Press ESC to quit")

    try:
        while True:
            color_frame, depth_frame = camera.get_frames()

            if color_frame is None or depth_frame is None:
                continue

            # Convert RealSense color frame to a NumPy image
            color_image = (
                __import__("numpy").asanyarray(color_frame.get_data())
            )

            # Detect objects
            objects = detector.detect(color_image)

            # Display results
            for obj in objects:
                name = obj["name"]
                confidence = obj["confidence"]

                x1, y1, x2, y2 = obj["bbox"]
                center_x, center_y = obj["center"]

                # Get RealSense distance
                distance = distance_calculator.get_object_distance(
                    depth_frame,
                    center_x,
                    center_y
                )

                # Draw bounding box
                cv2.rectangle(
                    color_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                if distance is not None:
                    label = (
                        f"{name} "
                        f"{confidence:.2f} "
                        f"{distance:.2f} m"
                    )
                else:
                    label = (
                        f"{name} "
                        f"{confidence:.2f} "
                        f"distance: N/A"
                    )

                # Draw label
                cv2.putText(
                    color_image,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Draw center point
                cv2.circle(
                    color_image,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

            # Show camera footage
            cv2.imshow("Object Distance", color_image)

            ESC_KEY = 27
            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")


if __name__ == "__main__":
    main()