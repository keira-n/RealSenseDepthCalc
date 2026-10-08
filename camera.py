import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps
        )

        self.config.enable_stream(
            rs.stream.depth,
            width,
            height,
            rs.format.z16,
            fps
        )

        self.align = rs.align(rs.stream.color)

        self.profile = None
        self.depth_scale = None

    def start(self):
        self.profile = self.pipeline.start(self.config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        print(f"Depth scale: {self.depth_scale} meters")

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        return color_frame, depth_frame

    def get_distance(self, depth_frame, x, y):
        return depth_frame.get_distance(x, y)

    def stop(self):
        self.pipeline.stop()