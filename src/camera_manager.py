import pyrealsense2 as rs
import numpy as np

class CameraManager:
    def __init__(self, color_width=1280, color_height=720, depth_width=1280, depth_height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        self.depth_scale = 0.0
        self.align = rs.align(rs.stream.color)



        self._configure_streams()

    def _configure_streams(self):
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        # Enable color stream
        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, self.fps)
        else:
            self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)

        # Enable depth stream
        self.config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.fps)

    def get_resolution(self):
        return self.color_width, self.color_height, self.depth_width, self.depth_height, self.fps

    def start_stream(self):
        print("Starting RealSense camera stream...")
        profile = self.pipeline.start(self.config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        return True

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        # For MediaPipe, we need to make the array writeable for drawing later.
        # It's set to False by MediaPipe internally, so we set it back to True here.
        color_image.flags.writeable = True

        return color_image, aligned_depth_frame, (aligned_depth_frame.get_width(), aligned_depth_frame.get_height())

    def stop_stream(self):
        print("Stopping RealSense camera stream.")
        self.pipeline.stop()