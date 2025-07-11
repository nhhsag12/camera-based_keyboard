import pyrealsense2
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
        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        self.depth_to_disparity_filter = rs.disparity_transform(True)
        self.disparity_to_depth_filter = rs.disparity_transform(False)


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
        profile:rs.pipeline_profile = self.pipeline.start(self.config)
        device:rs.device = profile.get_device()
        # print(len(device.query_sensors()))
        depth_sensor: rs.depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        return True

    def _filter_depth_data(self, frame: rs.composite_frame)->rs.frame:
        # filtered_frame = self.decimation_filter.process(frame)
        # filtered_frame = self.spatial_filter.process(frame)
        filtered_frame = self.depth_to_disparity_filter.process(frame)
        filtered_frame = self.spatial_filter.process(filtered_frame)
        filtered_frame = self.temporal_filter.process(filtered_frame)
        filtered_frame = self.disparity_to_depth_filter.process(filtered_frame)
        filtered_frame = self.hole_filling_filter.process(filtered_frame)
        return filtered_frame

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames: rs.composite_frame = self.align.process(frames)
        aligned_depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
        color_frame: rs.video_frame = aligned_frames.get_color_frame()
        filtered_depth_frame:rs.frame = self._filter_depth_data(aligned_depth_frame)
        filtered_depth_frame:rs.depth_frame = filtered_depth_frame.as_depth_frame()


        if not aligned_depth_frame or not color_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        # For MediaPipe, we need to make the array writeable for drawing later.
        # It's set to False by MediaPipe internally, so we set it back to True here.
        color_image.flags.writeable = True

        return color_image, filtered_depth_frame, (aligned_depth_frame.get_width(), aligned_depth_frame.get_height())

    def stop_stream(self):
        print("Stopping RealSense camera stream.")
        self.pipeline.stop()