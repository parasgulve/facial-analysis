import cv2
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
import datetime


def process_frame(frame, reference_img, iteration):
    """
    Runs DeepFace functions on the input video frame.
    """
    try:
        print(f"Iteration {iteration}:")  # Print iteration number
        
        # Anti-spoofing check
        anti_spoofing = DeepFace.extract_faces(frame, anti_spoofing = True, enforce_detection=False, detector_backend="mtcnn")
        
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
        return {"error": str(e)}

def plot_results(results_list):
    """
    Plots the extracted values from the video processing.
    """
    ages = [res["age"] for res in results_list if "age" in res]
    emotions = [res["emotion"] for res in results_list if "emotion" in res]
    races = [res["race"] for res in results_list if "race" in res]
    genders = [0 if res["gender"] == "Man" else 1 for res in results_list if "gender" in res]
    anti_spoofing = [res["anti_spoofing"] for res in results_list if "anti_spoofing" in res]
    verification_results = [1 if res["verification"] else 0 for res in results_list if "verification" in res]
    representation = DeepFace.represent(img_path="test_image.jpg", enforce_detection=False, detector_backend="mtcnn")
    embeddings=representation[0]["embedding"]

    plt.figure(figsize=(16, 8))
    
    # Age Graph
    plt.subplot(2, 4, 1)
    plt.plot(ages, marker=None, linestyle='-')
    plt.title("Age Over Time")
    plt.xlabel("Session-->")
    plt.ylabel("Age")
    plt.xticks([])
    
    # Emotion Graph
    plt.subplot(2, 4, 2)
    plt.hist(emotions, bins=len(set(emotions)), alpha=0.7, color='g')
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.xticks(fontsize=10, rotation=45)

    # Race Graph
    plt.subplot(2, 4, 3)
    plt.hist(races, bins=len(set(races)), alpha=0.7, color='r')
    plt.title("Race Distribution")
    plt.xlabel("Race")
    plt.xticks(fontsize=10, rotation=45)
    
    # Gender as Binary Graph
    plt.subplot(2, 4, 4)
    plt.plot(genders, linestyle='-', marker=None, color='blue')
    plt.title("Gender Over Time (0: Man, 1: Woman)")
    plt.xlabel("Session-->")
    plt.ylabel("Gender")
    plt.yticks([0, 1], labels=["Man", "Woman"])
    plt.ylim(-0.25, 1.35)
    plt.xticks([])
    
    # Anti-Spoofing Graph
    plt.subplot(2, 4, 5)
    plt.plot(anti_spoofing, marker=None, linestyle='-', color='purple')
    plt.title("Anti Spoofing")
    plt.xlabel("Session-->")
    plt.ylabel("Blur Score")
    plt.yticks([0, 1], labels=["No", "Yes"])  # Only show 0 and 1 on y-axis
    plt.ylim(-0.25, 1.35)
    plt.xticks([])
    
    # Verification Results Graph
    plt.subplot(2, 4, 6)
    plt.plot(verification_results, marker=None, linestyle='-', color='blue')
    plt.title("Verification Results")
    plt.xlabel("Session-->")
    plt.ylabel("Verified (1: Yes, 0: No)")
    plt.yticks([0, 1])  # Only show 0 and 1 on y-axis
    plt.ylim(-0.25, 1.35)
    plt.xticks([])

    # Face Embedding Graph
    if embeddings:
        plt.subplot2grid((2, 4), (1, 2), colspan=2)
        plt.bar(range(len(embeddings)), embeddings, color='r')
        plt.title("Face Embedding Representation")
        plt.xlabel("Embedding Index")
        plt.ylabel("Embedding Value")
        plt.grid(axis="y")
    
    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"analysis_plot_{timestamp}.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show()
    

def main():
    reference_img = "test_image.jpg"  # Replace with your reference image path
    video_path = "test_video.mp4"  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)
    results_list = []
    iteration = 1  # Initialize iteration counter
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (300, 300))
        results = process_frame(frame, reference_img, iteration)
        results_list.append(results)
        iteration += 1  # Increment iteration counter

        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
       
        
    
    cap.release()
    cv2.destroyAllWindows()
    
    with open("results_list.txt", "a") as file:
        file.write(str(results_list))
        file.write("\n\n")
    # Plot results at the end
    plot_results(results_list)

if __name__ == "__main__":
    main()