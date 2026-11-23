import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import datetime
import os

print("=" * 60)
print("Facial Analysis Project - Demo Run")
print("=" * 60)

# Check if test files exist
if not os.path.exists("test_image.jpg"):
    print("\nError: test_image.jpg not found!")
    print("Looking for alternative...")
    if os.path.exists("test_image.JPG"):
        print("Found test_image.JPG - using this file")
        reference_img = "test_image.JPG"
    else:
        print("No test image found. Exiting.")
        exit(1)
else:
    reference_img = "test_image.jpg"

if not os.path.exists("test_video.mp4"):
    print("\nError: test_video.mp4 not found!")
    exit(1)

print(f"\nUsing reference image: {reference_img}")
print(f"Using video: test_video.mp4")

def process_frame(frame, reference_img, iteration):
    """
    Runs DeepFace functions on the input video frame.
    """
    try:
        print(f"Iteration {iteration}:")
        
        # Anti-spoofing check
        anti_spoofing = DeepFace.extract_faces(frame, anti_spoofing=True, enforce_detection=False, detector_backend="mtcnn")
        
        # Face analysis (age, gender, emotion, race)
        analysis = DeepFace.analyze(frame, actions=["age", "gender", "emotion", "race"], enforce_detection=False, detector_backend="mtcnn")
        
        # Face verification (compare with reference image)
        verification = DeepFace.verify(frame, reference_img, enforce_detection=False, detector_backend="mtcnn")
        
        return {
            "anti_spoofing": anti_spoofing[0]["antispoof_score"],
            "age": analysis[0]["age"],
            "gender": analysis[0]["dominant_gender"],
            "emotion": analysis[0]["dominant_emotion"],
            "race": analysis[0]["dominant_race"],
            "verification": verification["verified"],
        }
    except Exception as e:
        print(f"  Error: {str(e)}")
        return {"error": str(e)}

def main():
    cap = cv2.VideoCapture("test_video.mp4")
    
    if not cap.isOpened():
        print("\nError: Could not open video file!")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nTotal frames in video: {total_frames}")
    print("\nProcessing first 5 frames for demo...")
    print("(Press 'q' in the video window to stop early)\n")
    
    results_list = []
    iteration = 1
    max_frames = 5  # Process only first 5 frames for demo
    
    while cap.isOpened() and iteration <= max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (300, 300))
        results = process_frame(frame, reference_img, iteration)
        results_list.append(results)
        
        # Display the frame
        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
            
        iteration += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nProcessed {len(results_list)} frames")
    
    # Display results
    print("\nResults Summary:")
    for i, res in enumerate(results_list, 1):
        if "error" not in res:
            print(f"\nFrame {i}:")
            print(f"  Age: {res['age']}")
            print(f"  Gender: {res['gender']}")
            print(f"  Emotion: {res['emotion']}")
            print(f"  Race: {res['race']}")
            print(f"  Anti-spoofing score: {res['anti_spoofing']:.3f}")
            print(f"  Verified: {res['verification']}")
        else:
            print(f"\nFrame {i}: Error - {res['error']}")
    
    # Save results
    with open("demo_results.txt", "w") as file:
        file.write(f"Demo Run - {datetime.datetime.now()}\n")
        file.write(f"Processed {len(results_list)} frames\n\n")
        for i, res in enumerate(results_list, 1):
            file.write(f"Frame {i}: {res}\n")
    
    print(f"\nResults saved to: demo_results.txt")
    print("\nTo run the full analysis on all frames, use: python projectnormal.py")

if __name__ == "__main__":
    main()
