import cv2
import numpy as np

# Load the test image
img = cv2.imread('test_image.jpg')

if img is None:
    print("Error: Could not load test_image.jpg")
    exit(1)

# Get image dimensions
height, width = img.shape[:2]

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (width, height))

# Create a 5-second video (150 frames at 30 fps)
print("Creating test video...")
for i in range(150):
    # Create a copy of the image
    frame = img.copy()
    
    # Add frame number text
    text = f"Frame {i+1}/150"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Write frame to video
    out.write(frame)
    
    if (i + 1) % 30 == 0:
        print(f"  Progress: {i+1}/150 frames")

# Release video writer
out.release()
print("Test video created successfully: test_video.mp4")
print("Video duration: 5 seconds (150 frames at 30 fps)")
