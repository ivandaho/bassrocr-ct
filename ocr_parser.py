import cv2
import pytesseract
import pandas as pd
import numpy as np
import argparse
import os
import re
import csv
from datetime import datetime

def preprocess_image(image_path, debug=False):
    """Load the image and preprocess it for OCR and line detection."""
    # This value is crucial. It's the pixel intensity threshold used to
    # separate the background from the lines and text. For a white background
    # (pixel value ~255) and grey lines, a value around 230 is a good start.
    # You may need to tune this value for your specific screenshot's brightness.
    GREYSCALE_THRESHOLD = 245

    print("Step 1: Preprocessing image...")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply an inverted binary threshold. This makes the white background black (0)
    # and everything darker than the threshold (like grey lines and black text)
    # white (255). This is ideal for the Hough Line Transform to find the dividers.
    _, thresh = cv2.threshold(gray, GREYSCALE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    if debug:
        debug_filename = "debug_greyscale_threshold.png"
        cv2.imwrite(debug_filename, thresh)
        print(f"  Greyscale debug image saved to '{debug_filename}'")

    print("  Image preprocessed successfully.")
    return image, gray, thresh

def detect_transaction_dividers(thresh_image):
    """
    Detect horizontal lines using morphological operations and contour detection.
    This method is more robust to variations in line thickness than Hough Transform.
    """
    print("Step 2: Detecting transaction dividers using morphological operations...")
    
    # 1. Create a long horizontal kernel
    # The width (e.g., 40) is key. It should be long enough to connect broken
    # parts of a line, but not so long that it merges separate text elements.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    
    # 2. Apply morphological opening to isolate horizontal lines
    # This removes most of the text and other non-horizontal elements.
    detected_lines = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 3. Find contours of the resulting shapes
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("  Warning: No lines detected. The script may not work as expected.")
        return []

    # 4. Filter contours to keep only line-like shapes and get their y-coordinates
    y_coords = []
    min_line_width = thresh_image.shape[1] * 0.5 # Line must be at least 50% of image width
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # A line should be very wide and not very tall, and meet a minimum width
        if w > min_line_width and w > 5 * h:
            y_coords.append(y + h // 2) # Use the vertical center of the contour

    y_coords = sorted(list(set(y_coords)))

    # 5. Merge lines that are very close to each other
    merged_lines = []
    if y_coords:
        current_line = y_coords[0]
        for y in y_coords[1:]:
            if y - current_line > 10: # Threshold to consider a new line
                merged_lines.append(current_line)
                current_line = y
        merged_lines.append(current_line)

    print(f"  Detected {len(merged_lines)} distinct horizontal dividers.")
    return merged_lines

def segment_image(original_image, lines):
    """Segment the image into transaction rows based on detected lines."""
    print("Step 3: Segmenting image into transaction rows...")
    segments = []
    height = original_image.shape[0]
    
    # Add top and bottom of image as boundaries
    line_boundaries = [0] + lines + [height]
    
    for i in range(len(line_boundaries) - 1):
        top = line_boundaries[i]
        bottom = line_boundaries[i+1]
        
        # Crop a segment from the original image
        segment = original_image[top:bottom, :]
        
        # Basic sanity check: ignore very small segments
        if segment.shape[0] > 10: # Segment must be at least 10 pixels high
            segments.append(segment)
            
    print(f"  Created {len(segments)} image segments.")
    return segments

def parse_segments(segments, debug=False):
    """
    Perform OCR on each segment and parse the text based on the new format.
    - Date: "DD Mon" (optional, at the start)
    - Amount: "[+|-] XX.XX SGD" (required, at the end)
    - Description: Everything else.
    """
    print("Step 4: Parsing segments with OCR...")

    debug_path = "debug_segments"
    if debug:
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        print(f"  Segment debugging is enabled. Saving to '{debug_path}'")

    transactions = []
    current_date = None

    # Regex to find a date like "14 Jul" or "06 Aug" at the START of the string.
    date_pattern = re.compile(r"^(\d{1,2}\s\w{3})")
    # Regex to find an amount like "+ 14.10 SGD" or "- 2.50 SGD" at the END of the string.
    amount_pattern = re.compile(r"([\+\-]\s\d+\.\d{2}\sSGD)$")

    for i, segment in enumerate(segments):
        if debug:
            cv2.imwrite(os.path.join(debug_path, f"segment_{i:03d}.png"), segment)

        text = pytesseract.image_to_string(segment, config='--psm 6').strip()
        if not text:
            continue

        # Make a copy of the text for parsing
        remaining_text = text
        date_found_in_segment = None

        # 1. Check for a date at the start of the segment
        date_match = date_pattern.search(remaining_text)
        if date_match:
            date_found_in_segment = date_match.group(1)
            current_date = date_found_in_segment
            # Remove the date from the text we need to parse
            remaining_text = date_pattern.sub("", remaining_text).strip()
            print(f"  Found and set date: {current_date}")

        # 2. Handle the first transaction needing a date
        if current_date is None:
            print(f"  Warning: Skipping segment {i+1} because a date has not been found yet. Text: '{text}'")
            continue

        # 3. Check for an amount at the end of the text
        amount_match = amount_pattern.search(remaining_text)
        if amount_match:
            # Clean and format the amount
            raw_amount_str = amount_match.group(1)
            cleaned_amount_str = raw_amount_str.replace("SGD", "").replace("+", "").replace(" ", "")
            
            # Convert to float to ensure correct format, then format to 2 decimal places
            try:
                amount_float = float(cleaned_amount_str)
                formatted_amount = f"{amount_float:.2f}"
            except ValueError:
                print(f"  Warning: Could not convert amount '{cleaned_amount_str}' to a number. Skipping.")
                continue

            # The description is whatever is left after removing the amount.
            # Newlines are replaced with " | " for better readability in the CSV.
            description_raw = amount_pattern.sub("", remaining_text).strip()
            description = " | ".join(line.strip() for line in description_raw.split('\n') if line.strip())

            transactions.append({
                "Date": current_date,
                "Description": description,
                "Amount": formatted_amount
            })
            print(f"  Parsed transaction: {description} | {formatted_amount}")
        else:
            # This segment is not a valid transaction (e.g., it might just be a date header)
            if not date_found_in_segment:
                 # Clean the text for printing to avoid f-string errors
                clean_text = text.replace('\n', ' ')
                print(f"  Skipping segment {i+1} (unrecognized format, no amount found): '{clean_text}'")

    print(f"\nSuccessfully parsed {len(transactions)} transactions.")
    return transactions



def save_to_csv(transactions, output_path):
    """Save the parsed transactions to a CSV file."""
    if not transactions:
        print("Warning: No transactions were parsed. CSV file will be empty.")
        return
        
    print(f"Step 5: Saving data to {output_path}...")
    df = pd.DataFrame(transactions)
    df = df[["Date", "Description", "Amount"]] # Ensure column order
    # Use quoting=csv.QUOTE_ALL to wrap every field in double quotes.
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Parse transactions from a stitched bank app screenshot.")
    parser.add_argument("image_path", help="Path to the input PNG image.")
    parser.add_argument("csv_path", help="Path to save the output CSV file.")
    parser.add_argument(
        "--debug-segments",
        action="store_true",
        help="Save intermediate image segments to a 'debug_segments/' folder."
    )
    parser.add_argument(
        "--debug-greyscale",
        action="store_true",
        help="Save the intermediate greyscale threshold image."
    )
    args = parser.parse_args()

    try:
        # Note: This script requires Tesseract to be installed on your system.
        # Please see: https://tesseract-ocr.github.io/tessdoc/Installation.html
        
        original_image, gray_image, thresh_image = preprocess_image(args.image_path, args.debug_greyscale)
        dividers = detect_transaction_dividers(thresh_image)
        segments = segment_image(original_image, dividers)
        transactions = parse_segments(segments, args.debug_segments)
        save_to_csv(transactions, args.csv_path)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
