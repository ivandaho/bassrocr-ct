import cv2
import pandas as pd
import argparse
import os
import csv
import pytesseract
from config import get_statement_config
from parsers import get_parser, PARSER_MAPPING

def preprocess_image(image_path, greyscale_threshold, theme='light', debug=False):
    """Load the image and preprocess it for OCR and line detection."""
    print(f"Step 1: Preprocessing image... (Theme: {theme}, Greyscale Threshold: {greyscale_threshold})")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if theme == 'dark':
        threshold_type = cv2.THRESH_BINARY
    else:
        threshold_type = cv2.THRESH_BINARY_INV

    _, thresh = cv2.threshold(gray, greyscale_threshold, 255, threshold_type)
    
    if debug:
        debug_filename = "debug_greyscale_threshold.png"
        cv2.imwrite(debug_filename, thresh)
        print(f"  Greyscale debug image saved to '{debug_filename}'")

    print("  Image preprocessed successfully.")
    return image, gray, thresh

def detect_transaction_dividers(thresh_image):
    """Detect horizontal lines using morphological operations."""
    print("Step 2: Detecting transaction dividers...")
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("  Warning: No lines detected.")
        return []

    y_coords = []
    min_line_width = thresh_image.shape[1] * 0.5
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > min_line_width and w > 5 * h:
            y_coords.append(y + h // 2)

    y_coords = sorted(list(set(y_coords)))

    merged_lines = []
    if y_coords:
        current_line = y_coords[0]
        for y in y_coords[1:]:
            if y - current_line > 10:
                merged_lines.append(current_line)
                current_line = y
        merged_lines.append(current_line)

    print(f"  Detected {len(merged_lines)} distinct horizontal dividers.")
    return merged_lines

def segment_image(original_image, lines, statement_type="uobone"):
    """Segment the image into transaction rows based on detected lines."""
    print(f"Step 3: Segmenting image into transaction rows... (statement_type: {statement_type})")
    segments = []
    height = original_image.shape[0]
    
    line_boundaries = [0] + lines + [height]
    
    for i in range(len(line_boundaries) - 1):
        top = line_boundaries[i]
        bottom = line_boundaries[i+1]
        
        if statement_type == "uobevol":
            segment = original_image[top:bottom, :728]
        else:
            segment = original_image[top:bottom, :]
        
        if segment.shape[0] > 10:
            segments.append(segment)
            
    print(f"  Created {len(segments)} image segments.")
    return segments

def save_to_csv(transactions, output_path):
    """Save the parsed transactions to a CSV file."""
    if not transactions:
        print("Warning: No transactions were parsed. CSV file will be empty.")
        return
        
    print(f"Step 5: Saving data to {output_path}...")
    df = pd.DataFrame(transactions)
    df = df[["date", "description", "amount"]]
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print("Done.")

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-s", "--statement-type",
        default="uobone",
        help="Type of statement to process (e.g., uobone, uobevol, trust)."
    )
    args, _ = pre_parser.parse_known_args()
    
    try:
        config = get_statement_config(args.statement_type)
        parser_class = PARSER_MAPPING.get(args.statement_type)
        if not parser_class:
            raise ValueError(f"Unsupported statement type: {args.statement_type}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    parser = argparse.ArgumentParser(
        description="Parse transactions from a stitched bank app screenshot.",
        parents=[pre_parser]
    )
    parser.add_argument("image_path", help="Path to the input PNG image.")
    parser.add_argument("csv_path", help="Path to save the output CSV file.")
    parser.add_argument("--greyscale-threshold", type=int, default=config.get('parser_greyscale_threshold'), help="Greyscale threshold for image preprocessing.")
    parser.add_argument("--theme", choices=['light', 'dark'], default=config.get('theme', 'light'), help="Image theme.")
    parser.add_argument("--debug-segments", action="store_true", help="Save intermediate image segments.")
    parser.add_argument("--debug-greyscale", action="store_true", help="Save the intermediate greyscale threshold image.")
    args = parser.parse_args()

    try:
        original_image, _, thresh_image = preprocess_image(
            args.image_path, args.greyscale_threshold, args.theme, args.debug_greyscale
        )

        parser_kwargs = {'debug': args.debug_segments}
        if parser_class.requires_dividers:
            dividers = detect_transaction_dividers(thresh_image)
            segments = segment_image(original_image, dividers, args.statement_type)
            parser_kwargs['segments'] = segments
        else:
            # upscale image to help with ocr
            resized_image = cv2.resize(original_image, None, dst=None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            print("Step 2/3/4: Skipping divider detection and segmentation, Performing full-page OCR")
            ocr_data = pytesseract.image_to_data(resized_image, output_type=pytesseract.Output.DATAFRAME)
            if args.debug_segments:
                debug_dir = "debug"
                if not os.path.exists(debug_dir):
                    os.makedirs(debug_dir)
                ocr_data.to_csv(os.path.join(debug_dir, 'ocr_data_full.csv'), index=False)
                print("  DEBUG: Saved full OCR data to debug/ocr_data_full.csv")
            parser_kwargs['ocr_data'] = ocr_data

        parser_instance = get_parser(args.statement_type, **parser_kwargs)
        transactions = parser_instance.parse()
        
        save_to_csv(transactions, args.csv_path)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
