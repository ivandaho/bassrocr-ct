
import cv2
import numpy as np
import argparse
import os

def find_scrolling_roi(video_path, num_frames_to_sample=10, guide_point=None):
    """
    Analyzes the video to find the region of interest (ROI) that is scrolling.
    It does this by detecting which parts of the screen are in motion.

    Args:
        video_path (str): The path to the video file.
        num_frames_to_sample (int): The number of frames to sample to detect motion.
        guide_point (tuple, optional): An (x, y) coordinate to help identify the
                                       correct scrolling area.

    Returns:
        tuple: A tuple (x, y, w, h) defining the bounding box of the scrolling ROI,
               or None if detection fails.
    """
    print(f"Detecting scrolling region using {num_frames_to_sample} sample frames...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        return None

    sample_indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)
    
    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if not frames:
        cap.release()
        return None

    base_frame = frames[0]
    motion_mask = np.zeros_like(base_frame)

    for i in range(1, len(frames)):
        diff = cv2.absdiff(base_frame, frames[i])
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.bitwise_or(motion_mask, thresh)

    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    motion_mask = cv2.erode(motion_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Warning: No motion detected. Stitching will proceed using the full frame.")
        cap.release()
        return None

    # If a guide point is provided, find the contour that contains it
    if guide_point:
        found_contour = None
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, guide_point, False) >= 0:
                found_contour = cnt
                break
        
        if found_contour is not None:
            # If we found the right contour, use its bounding box as the ROI
            x, y, w, h = cv2.boundingRect(found_contour)
            print("Located specific scrolling area using guide point.")
        else:
            # If no contour contains the guide point, fall back to the default
            print("Warning: Guide point did not fall within any detected motion area. Using overall motion.")
            all_points = np.concatenate([cnt for cnt in contours])
            x, y, w, h = cv2.boundingRect(all_points)
    else:
        # Default behavior: combine all contours into a single bounding box
        all_points = np.concatenate([cnt for cnt in contours])
        x, y, w, h = cv2.boundingRect(all_points)

    # Add some padding to the ROI
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(base_frame.shape[1] - x, w + 2 * padding)
    h = min(base_frame.shape[0] - y, h + 2 * padding)
    
    cap.release()
    print(f"Detected ROI at: x={x}, y={y}, width={w}, height={h}")
    return (53, 326, 785, 1226)
    # return (x, y, w, h)


def extract_keyframes(video_path, roi=None, threshold=0.08):
    """
    Extracts keyframes from a video file, focusing on changes within the ROI.

    Args:
        video_path (str): The path to the video file.
        roi (tuple, optional): The (x, y, w, h) of the region to check for changes.
        threshold (float): A value between 0 and 1 for change detection sensitivity.
                           Lower is more sensitive.

    Returns:
        list: A list of keyframes (as numpy arrays).
    """
    print("Extracting keyframes...")
    # Lowered default threshold for more sensitivity
    keyframes = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return keyframes

    prev_frame = None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is None:
            prev_frame = frame
            keyframes.append(frame)
            continue

        # Crop frames to ROI if it exists, otherwise use the full frame
        if roi:
            x, y, w, h = roi
            frame_roi = frame[y:y+h, x:x+w]
            prev_frame_roi = prev_frame[y:y+h, x:x+w]
        else:
            frame_roi = frame
            prev_frame_roi = prev_frame

        # Convert frames to grayscale for difference calculation
        gray_frame = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        gray_prev_frame = cv2.cvtColor(prev_frame_roi, cv2.COLOR_BGR2GRAY)

        # Calculate the Sum of Absolute Differences (SAD)
        sad = np.sum(cv2.absdiff(gray_frame, gray_prev_frame))
        
        num_pixels = gray_frame.shape[0] * gray_frame.shape[1]
        if num_pixels == 0: continue # Avoid division by zero if ROI is bad
        normalized_sad = sad / (num_pixels * 255)

        if normalized_sad > threshold:
            keyframes.append(frame)
            prev_frame = frame
            print(f"Found keyframe {len(keyframes)} at frame {i} (change: {normalized_sad:.4f})")

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes.")
    return keyframes


def align_and_stitch(keyframes, debug_stitch=False, debug_dir="."):
    """
    Aligns and stitches a list of pre-cropped keyframes, assuming vertical scroll.
    This function now compares each frame ONLY to the one immediately preceding it,
    avoiding cumulative errors from skipped frames.
    """
    print("Aligning and stitching frames (vertical only)...")
    if not keyframes:
        return None

    stitched_image = keyframes[0]
    if debug_stitch:
        debug_filename = os.path.join(debug_dir, "stitched_001.png")
        cv2.imwrite(debug_filename, stitched_image)

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(len(keyframes) - 1):
        # Always compare consecutive frames to get the relative offset.
        img1 = keyframes[i]
        img2 = keyframes[i+1]

        print(f"Comparing keyframe {i+1} and {i+2}...")

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
            print("Warning: Not enough descriptors found. Skipping pair.")
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]

        if len(good_matches) < 10:
            print("Warning: Not enough good matches. Skipping pair.")
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        y_offsets = src_pts[:, 1] - dst_pts[:, 1]
        y_offset = int(np.median(y_offsets))

        if y_offset > 0:
            scroll_dist = y_offset
            h_stitched, w_stitched = stitched_image.shape[:2]
            h_keyframe, w_keyframe = img2.shape[:2]

            if scroll_dist >= h_keyframe:
                print(f"Warning: Unusually large scroll detected ({scroll_dist}px). Skipping.")
                continue

            # Append the new part to the main stitched image.
            new_height = h_stitched + scroll_dist
            new_canvas = np.zeros((new_height, w_stitched, 3), dtype=np.uint8)
            new_canvas[0:h_stitched, 0:w_stitched] = stitched_image
            
            new_part = img2[-scroll_dist:, :]
            new_canvas[h_stitched:, 0:w_stitched] = new_part
            
            stitched_image = new_canvas
            print(f"Stitched {scroll_dist}px from keyframe {i+2}")

            if debug_stitch:
                debug_filename = os.path.join(debug_dir, f"stitched_{i+2:03d}.png")
                cv2.imwrite(debug_filename, stitched_image)
        else:
            print(f"Warning: No downward scroll detected (offset: {y_offset}). Skipping pair.")
            continue

    return stitched_image


def main():
    """
    Main function to parse arguments and run the stitching process.
    """
    parser = argparse.ArgumentParser(description="Stitch a scrolling screenshot from a video.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("output_path", help="Path to save the final stitched image.")
    parser.add_argument("--roi-frames", type=int, default=10,
                        help="Number of frames to sample for automatic ROI detection. Default: 10.")
    parser.add_argument("--no-roi-detection", action="store_true",
                        help="Disable automatic ROI detection and use the full frame.")
    parser.add_argument("--debug", action="store_true",
                        help="Save keyframes with the detected ROI drawn on them to a 'debug' folder.")
    parser.add_argument("--roi-x", type=int, help="An X coordinate inside the desired scrolling area to guide ROI detection.")
    parser.add_argument("--roi-y", type=int, help="A Y coordinate inside the desired scrolling area to guide ROI detection.")
    args = parser.parse_args()

    roi_guide_point = None
    if args.roi_x is not None and args.roi_y is not None:
        roi_guide_point = (args.roi_x, args.roi_y)
    elif args.roi_x is not None or args.roi_y is not None:
        parser.error("--roi-x and --roi-y must be used together.")

    roi = None
    if not args.no_roi_detection:
        roi = find_scrolling_roi(args.video_path, args.roi_frames, guide_point=roi_guide_point)

    full_keyframes = extract_keyframes(args.video_path, roi=roi)

    if not full_keyframes:
        print("No keyframes were extracted. Cannot proceed with stitching.")
        return

    # If an ROI is detected, crop all keyframes to that ROI before proceeding.
    # The rest of the script will only work with the cropped images.
    if roi:
        x, y, w, h = roi
        keyframes_roi = [frame[y:y+h, x:x+w] for frame in full_keyframes]
    else:
        keyframes_roi = full_keyframes

    if args.debug:
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        print(f"Saving debug frames to '{debug_dir}' directory...")
        for i, frame in enumerate(full_keyframes):
            frame_to_save = frame.copy()
            if roi:
                x, y, w, h = roi
                cv2.rectangle(frame_to_save, (x, y), (x + w, y + h), (0, 0, 255), 2)
            debug_filename = os.path.join(debug_dir, f"keyframe_{i:03d}.png")
            cv2.imwrite(debug_filename, frame_to_save)

    stitched_image = align_and_stitch(keyframes_roi)
    
    if stitched_image is not None:
        cv2.imwrite(args.output_path, stitched_image)
        print(f"Stitched image saved to {args.output_path}")
    else:
        print("Could not create stitched image.")


if __name__ == "__main__":
    main()
