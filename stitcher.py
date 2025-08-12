import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from config import get_statement_config

def mse(imageA, imageB):
    """Calculates the Mean Squared Error between two images."""
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def extract_keyframes(video_path, diff_threshold, debug=False):
    """
    Extracts keyframes from a video file by comparing consecutive frames.
    A frame is considered a keyframe if it's significantly different from the
    previous keyframe.
    """
    # Limit processing to a max FPS to speed up analysis on high-FPS videos.
    MAX_PROCESSING_FPS = 30

    print(f"Step 1: Extracting keyframes... (Difference Threshold: {diff_threshold})")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = 1
    if video_fps > MAX_PROCESSING_FPS:
        frame_skip = int(round(video_fps / MAX_PROCESSING_FPS))
        print(f"  Video FPS ({video_fps:.2f}) > max FPS ({MAX_PROCESSING_FPS}). Processing 1 of every {frame_skip} frames.")

    if debug:
        debug_path = "debug_keyframes"
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        print(f"  Keyframe debugging is enabled. Saving to '{debug_path}'")

    keyframes = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("  Could not read the first frame.")
        cap.release()
        return keyframes
        
    keyframes.append(prev_frame)
    if debug:
        cv2.imwrite(os.path.join(debug_path, f"keyframe_{len(keyframes)-1:04d}.png"), prev_frame)

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        pbar.update(1)
        frame_count = 1
        while True:
            # Skip frames if necessary. We skip (frame_skip - 1) frames.
            for _ in range(frame_skip - 1):
                ret = cap.grab() # .grab() is faster as it doesn't decode
                if not ret:
                    break
                frame_count += 1
                pbar.update(1)
            
            if not ret:
                break # Reached end of video during skip

            # Now, read the frame we want to process
            ret, current_frame = cap.read()
            if not ret:
                break # Reached end of video
            
            frame_count += 1
            pbar.update(1)
            
            difference = mse(current_frame, keyframes[-1])
            
            if difference > diff_threshold:
                keyframes.append(current_frame)
                if debug:
                    cv2.imwrite(os.path.join(debug_path, f"keyframe_{len(keyframes)-1:04d}.png"), current_frame)
            
    cap.release()
    print(f"  Extracted {len(keyframes)} keyframes from {total_frames} total frames.")
    return keyframes

def determine_next_data(stitched_roi, next_frame, roi, debug=False, frame_num=0):
    """
    Determines the new data to be stitched from the next frame.

    Compares the bottom part of the stitched_roi with the next_frame to find
    the overlap and extract the non-overlapping part.
    """
    # Crop the next frame to the ROI for comparison
    next_frame_roi = next_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

    # Use a larger template for better uniqueness.
    template_height = min(300, int(stitched_roi.shape[0] * 0.8))
    if template_height <= 20: # Template too small to be reliable
        return None
    template = stitched_roi[-template_height:]

    # The template cannot be larger than the image it's searching in.
    if template.shape[0] > next_frame_roi.shape[0] or template.shape[1] > next_frame_roi.shape[1]:
        return None

    # Search for this template in the next frame's ROI
    match_result = cv2.matchTemplate(next_frame_roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(match_result)

    # Use a higher confidence threshold to avoid false matches.
    if max_val > 0.97:
        start_y_in_roi = max_loc[1] + template_height
        
        if start_y_in_roi >= next_frame_roi.shape[0]:
             return None

        new_data = next_frame_roi[start_y_in_roi:]

        # *** Sanity Check ***
        # The amount of new data should not be excessive. If it's almost the full
        # height of the ROI, it's likely a bad match.
        if new_data.shape[0] > roi[3] * 0.9:
            print(f"    Sanity check failed: proposed new data is too large ({new_data.shape[0]} pixels). Skipping.")
            return None

        if new_data.shape[0] > 0:
            if debug:
                debug_path = "debug"
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)
                # Use a unique name for each piece of debug data
                cv2.imwrite(os.path.join(debug_path, f"new_data_from_frame_{frame_num:04d}.png"), new_data)
            return new_data
            
    return None


def stitch_keyframes(keyframes, roi, debug=False):
    """
    Stitches the keyframes together based on the ROI.
    """
    print("Step 2: Stitching keyframes...")
    if not keyframes:
        return None

    # Initialize with the ROI from the first keyframe
    stitched_image = keyframes[0][roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    
    for i in range(1, len(keyframes)):
        print(f"  Processing frame {i+1}/{len(keyframes)}...")
        
        # Directly pass the already stitched image (which is the stitched ROI)
        # and the next full frame.
        new_data = determine_next_data(stitched_image, keyframes[i], roi, debug, i)

        if new_data is not None and new_data.shape[0] > 2: # Avoid adding tiny slivers
            stitched_image = cv2.vconcat([stitched_image, new_data])
            print(f"    Stitched {new_data.shape[0]} new rows of pixels.")
        else:
            print("    No confident match or new data found.")

    return stitched_image


def main():
    parser = argparse.ArgumentParser(
        description="Stitch a scrolling region of interest from a video into a single PNG.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # First, parse only the statement type to load the correct config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-s", "--statement-type",
        default="uobone",
        help="Type of statement to process (e.g., uobone, uobevol, trust). Determines default settings."
    )
    args, _ = pre_parser.parse_known_args()
    
    # Load configuration based on statement type
    try:
        config = get_statement_config(args.statement_type)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Now, define the full parser with defaults from the loaded config
    parser = argparse.ArgumentParser(
        description="Stitch a scrolling region of interest from a video into a single PNG.",
        formatter_class=argparse.RawTextHelpFormatter,
        parents=[pre_parser] # Inherit the --statement-type argument
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("output_path", help="Path to save the output PNG image.")
    parser.add_argument(
        "--roi",
        default=config.get('roi'),
        help="Region of Interest 'x,y,w,h'. Overrides config file."
    )
    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=config.get('stitcher_diff_threshold'),
        help="MSE threshold for keyframe detection. Overrides config file."
    )
    parser.add_argument(
        "--debug-next-data",
        action="store_true",
        help="Save intermediate 'new_data' images to a 'debug/' folder."
    )
    parser.add_argument(
        "--debug-keyframes",
        action="store_true",
        help="Save all extracted keyframes to a 'debug_keyframes/' folder."
    )
    parser.add_argument(
        "--debug-roi",
        action="store_true",
        help="Draw the ROI on the first frame and save it as debug_roi.png, then exit."
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at '{args.video_path}'")
        return

    try:
        roi = tuple(map(int, args.roi.split(',')))
        if len(roi) != 4:
            raise ValueError
    except ValueError:
        print("Error: ROI must be in the format 'x,y,w,h'")
        return

    if args.debug_roi:
        cap = cv2.VideoCapture(args.video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imwrite("debug_roi.png", frame)
            print("Saved ROI debug image to debug_roi.png")
        else:
            print("Error: Could not read the first frame of the video.")
        return

    # Step 1: Extract keyframes
    keyframes = extract_keyframes(args.video_path, args.diff_threshold, args.debug_keyframes)
    if not keyframes:
        print("Error: Could not extract any frames from the video.")
        return

    # If ROI is all zeros, use the full dimensions of the first frame
    if all(v == 0 for v in roi):
        h, w, _ = keyframes[0].shape
        roi = (0, 0, w, h)
        print(f"No ROI specified, using full frame dimensions: {w}x{h}")

    # Step 2: Stitch keyframes
    final_image = stitch_keyframes(keyframes, roi, args.debug_next_data)

    # Step 3: Save the output
    if final_image is not None and final_image.shape[0] > roi[3]:
        print(f"\nStep 3: Saving final stitched image to {args.output_path}")
        cv2.imwrite(args.output_path, final_image)
        print("Done.")
    else:
        print("\nCould not generate a stitched image or no scrolling was detected.")

if __name__ == "__main__":
    main()
