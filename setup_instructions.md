# Facial Analysis Project - Setup Instructions

## Prerequisites
- Python 3.12 installed
- (Optional) NVIDIA GPU with CUDA 12.x and cuDNN 8.9 for GPU acceleration

## Setup Steps

### 1. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
cd "d:\Deepak\Paras\Projects\facial analysis"
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
cd "d:\Deepak\Paras\Projects\facial analysis"
.\venv\Scripts\activate.bat
```

### 2. Verify GPU Support (Optional)

Run the GPU verification script to check if TensorFlow can access your GPU:

```powershell
python verify_gpu.py
```

If you see "GPU Available: []" (empty list), TensorFlow will run on CPU. To enable GPU:
1. Install CUDA Toolkit 12.x: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN 8.9: https://developer.nvidia.com/cudnn
3. Restart your computer and run `verify_gpu.py` again

### 3. Prepare Test Files

Before running the project, you need:
- A reference image (e.g., `test_image.jpg`)
- A test video (e.g., `test_video.mp4`)

Place these files in the project directory or update the paths in `projectnormal.py`.

### 4. Update File Paths

Edit `projectnormal.py` and update these lines with your actual file paths:
- Line 46: Path to reference image for face embedding
- Line 120: Path to reference image for verification
- Line 121: Path to test video

### 5. Run the Project

```powershell
python projectnormal.py
```

The script will:
- Process each frame of the video
- Perform face detection and analysis (age, gender, emotion, race)
- Check for anti-spoofing
- Verify faces against the reference image
- Display the processed video
- Generate analysis plots saved as PNG files

Press 'q' to stop the video processing.

## Installed Packages

- **TensorFlow 2.16.1**: Deep learning framework with GPU support
- **DeepFace 0.0.93**: Facial analysis library
- **OpenCV 4.10.0.84**: Computer vision library
- **NumPy 1.26.4**: Numerical computing
- **Matplotlib 3.9.2**: Plotting library
- **Pandas 2.2.3**: Data analysis
- **Pillow 10.4.0**: Image processing

## Troubleshooting

### "No module named 'tensorflow'"
Make sure the virtual environment is activated (you should see `(venv)` in your terminal).

### "Could not find reference image"
Update the file paths in `projectnormal.py` to point to your actual test files.

### GPU not detected
Install CUDA Toolkit 12.x and cuDNN 8.9, then restart your computer.

### Video window not showing
Make sure you have a display connected. The script uses `cv2.imshow()` which requires a GUI.
