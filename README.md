# ⚽ Automated Football Highlight Generator

An intelligent Python pipeline that automatically analyzes full-length football match footage, detects key events, and stitches together a clean highlight reel.

## 🧠 How it Works

Instead of relying on a single data point, this script uses a multi-modal approach to mimic professional video editing:
1. **Audio Peak Detection (Librosa):** Analyzes the RMS energy of the audio track. It dynamically calculates a 99.5th percentile threshold to identify sudden spikes in crowd noise or commentator volume, adapting to quiet matches or loud stadiums automatically.
2. **Visual Context (PySceneDetect):** Audio peaks often occur mid-play. To prevent jarring mid-action cuts, the script uses computer vision to scan backward from the audio peak to find the exact frame of the previous camera cut.
3. **Automated Stitching (MoviePy):** Validated highlight clips are extracted and seamlessly merged into a final MP4 output.

## 🛠️ Tech Stack
* **Python 3.10+**
* **Librosa & SciPy:** Audio signal processing and peak detection.
* **PySceneDetect & OpenCV:** Computer vision and frame analysis.
* **MoviePy:** Automated video editing and rendering.
* **NumPy:** Statistical thresholding.

## 🚀 How to Run Locally

1. Clone this repository
2. Create a virtual environment and activate it
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt