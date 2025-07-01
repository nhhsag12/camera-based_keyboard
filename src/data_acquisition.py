import pyrealsense2 as rs
import numpy as np
import cv2

def initialize_realsense_d405():
    """Initializes the RealSense pipeline for D405, focusing on short-range."""
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams suitable for D405's strengths (short range, high resolution)
    # The D405's RGB comes from the depth sensor, so its FOV is matched.
    # Set resolution and framerate. D405 supports up to 1280x720 at 90fps.
    width, height = 1280, 720
    fps = 30 # Or 60/90 if your system can handle it

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) # D405's RGB is BGR8

    # Align depth frames to color frames (important for combining data)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    profile = pipeline.start(config)

    # Get depth sensor and set ideal depth range for D405
    depth_sensor = profile.get_device().first_depth_sensor()
    # D405 ideal range is 7cm to 50cm.
    # You might want to adjust depth clipping based on your setup.
    depth_scale = depth_sensor.get_depth_scale() # Get depth unit (meters)

    # Configure post-processing filters for better depth data (optional but recommended)
    # This helps with noise reduction and filling small holes
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    hole_filling_filter = rs.hole_filling_filter()

    # You can set filter options, e.g., spatial_filter.set_option(rs.option.holes_fill, 1)

    print(f"RealSense D405 initialized. Depth scale: {depth_scale} meters per unit.")
    return pipeline, align, depth_scale, spatial_filter, temporal_filter, hole_filling_filter

def capture_frames(pipeline, align, spatial_filter, temporal_filter, hole_filling_filter):
    """Captures and processes RealSense frames."""
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None

    # Apply post-processing filters
    # depth_frame = spatial_filter.process(depth_frame)
    # depth_frame = temporal_filter.process(depth_frame)
    # depth_frame = hole_filling_filter.process(depth_frame)

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image

# Example usage:
pipeline, align, depth_scale, sf, tf, hf = initialize_realsense_d405()
try:
    while True:
        color_img, depth_img = capture_frames(pipeline, align, sf, tf, hf)
        if color_img is None:
            continue
        # Display images (optional, for debugging)
        cv2.imshow('Color Image', color_img)
        # Convert depth to a visible colormap for display
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Colormap', depth_colormap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()