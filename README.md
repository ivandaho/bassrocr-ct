# Video Scroll Stitcher

This script stitches a full-page screenshot from a screen-recorded video of a user scrolling through a mobile app or webpage.

## Prerequisites

- Python 3.x
- FFmpeg (often required by OpenCV for video processing, though not always a strict dependency depending on the OS and video format)

## Installation

1.  **Clone the repository or download the files.**

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script from your terminal, providing the path to the input video and the desired path for the output image.

```bash
python stitcher.py path/to/your/video.mp4 path/to/your/output.png
```

### Advanced Usage

#### Guiding the ROI Detection

If there are multiple moving elements on the screen (like a video playing in the background) and the automatic detection selects the wrong one, you can guide it by providing a coordinate that is *inside* the area you want to stitch.

```bash
# Guide the detection by specifying a point (400, 800) within the scrolling area
python stitcher.py my_video.mp4 my_stitch.png --roi-x 400 --roi-y 800
```

If the automatic detection is not working correctly for your video, you can disable it entirely. This will force the script to use the full video frame for feature matching.

```bash
# Disable the automatic ROI detection
python stitcher.py my_video.mp4 my_stitch.png --no-roi-detection
```

#### Debugging the ROI

If you want to visualize the detected Region of Interest (ROI), you can use the `--debug` flag. This will create a `debug` folder and save each keyframe with a red rectangle drawn around the ROI. This is useful for ensuring the automatic detection is working as expected.

```bash
python stitcher.py my_video.mp4 my_stitch.png --debug
```

### Example

```bash
python stitcher.py sample_data/scrolling_video.mov stitched_screenshot.png
```

## How It Works

1.  **Automatic ROI Detection (Optional):** The script can analyze the video to find the "Region of Interest" (ROI) that is scrolling. It does this by sampling several frames, identifying which pixels are in motion, and calculating a bounding box around them. This prevents static elements like headers from interfering with the stitching process.

2.  **Keyframe Extraction:** The script analyzes the video to pull out "keyframes." It ignores duplicate frames (when the scroll is paused) and only processes frames where significant movement has occurred. This makes the process much more efficient.

3.  **Feature Matching:** It uses the ORB (Oriented FAST and Rotated BRIEF) algorithm from OpenCV to find unique feature points in consecutive keyframes (either in the full frame or within the detected ROI). It then matches these points to determine how the second frame is positioned relative to the first.

4.  **Stitching:** Based on the calculated offset between frames, the script progressively pastes the new, non-overlapping part of each keyframe onto a master canvas, building the final, long screenshot.
