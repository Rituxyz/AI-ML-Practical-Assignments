# Assignment 9: Object Detection from CCTV Footage

import cv2
import os
from ultralytics import YOLO

# Step 1: Load the pretrained YOLO model
model = YOLO("yolov8n.pt")

# Step 2: Specify the input and output video paths
video_path = "09-CCTV-Object-Detection/cctv_footage.mp4"
output_path = "09-CCTV-Object-Detection/detected_output.mp4"

# Step 3: Check whether the input video exists
if not os.path.exists(video_path):
    print("Error: CCTV video file not found.")
    print("Please check the video path:", video_path)
    raise SystemExit

# Step 4: Open the input video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open the video.")
    raise SystemExit

# Step 5: Get the original video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25.0

print("Original video width:", width)
print("Original video height:", height)
print("Video FPS:", fps)

# Step 6: Create the output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("Error: Unable to create output video.")
    cap.release()
    raise SystemExit

# Step 7: Create a resizable display window
window_name = "CCTV Object Detection"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Set a smaller window size for display
cv2.resizeWindow(window_name, 640, 700)

print("\nObject detection started!")
print("Press Q to stop the program.")
print("Please wait while the video is processed.\n")

frame_count = 0

try:
    while True:

        # Read the next video frame
        ret, frame = cap.read()

        # Stop when the video ends
        if not ret:
            break

        # Step 8: Detect objects in the original frame
        results = model.predict(
            source=frame,
            conf=0.40,
            verbose=False
        )

        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Step 9: Save the frame at its original resolution
        out.write(annotated_frame)

        # Step 10: Resize only the displayed frame
        display_height = 700
        scale = display_height / height
        display_width = int(width * scale)

        display_frame = cv2.resize(
            annotated_frame,
            (display_width, display_height)
        )

        # Display the processed frame
        cv2.imshow(window_name, display_frame)

        frame_count += 1

        # Print progress every 50 frames
        if frame_count % 50 == 0:
            print("Frames processed:", frame_count)

        # Press Q to stop processing
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Processing stopped by user.")
            break

finally:
    # Step 11: Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Step 12: Display completion message
print("\nObject detection completed!")
print("Total frames processed:", frame_count)
print("Output video saved at:", output_path)