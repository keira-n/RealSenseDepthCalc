import numpy as np


class DistanceCalculator:
    def __init__(self, sample_radius=5):
        self.sample_radius = sample_radius

    # From camera
    def get_object_distance(self, depth_frame, center_x, center_y):
        distances = []

        # Get range of area around object coordinate
        for y in range(
                center_y - self.sample_radius,
                center_y + self.sample_radius + 1
        ):
            for x in range(
                    center_x - self.sample_radius,
                    center_x + self.sample_radius + 1
            ):
                # How far away the surface at pixel (x, y) is
                distance = depth_frame.get_distance(x, y)

                # Ignore invalid depth values
                if distance > 0:
                    distances.append(distance)

        if not distances:
            return None

        # Approx. central value of depth value
        return float(np.median(distances))