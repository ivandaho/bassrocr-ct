import os
import re
import cv2
import pytesseract
from .base_parser import TransactionParser

class UOBParser(TransactionParser):
    """Parses transactions from UOB (uobone, uobevol) statement screenshots."""

    def parse(self):
        """
        Perform OCR on each segment and parse the text based on the UOB format.
        - Date: "DD Mon" (optional, at the start)
        - Amount: "[+|-] XX.XX SGD" (required, at the end)
        - Description: Everything else.
        """
        print("Step 4: Parsing segments with UOBParser...")

        debug_path = "debug_segments"
        if self.debug:
            if not os.path.exists(debug_path):
                os.makedirs(debug_path)
            print(f"  Segment debugging is enabled. Saving to '{debug_path}'")

        current_date = None

        # Regex to find a date like "14 Jul" or "06 Aug" at the START of the string.
        date_pattern = re.compile(r"^(\d{1,2}\s\w{3})")
        # Regex to find an amount like "+ 14.10 SGD" or "- 2.50 SGD" at the END of the string.
        amount_pattern = re.compile(r"([\+\-]\s\d+\.\d{2}\sSGD)$")

        for i, segment in enumerate(self.segments):
            if self.debug:
                cv2.imwrite(os.path.join(debug_path, f"segment_{i:03d}.png"), segment)

            text = pytesseract.image_to_string(segment, config='--psm 6').strip()
            if not text:
                continue

            remaining_text = text
            date_found_in_segment = None

            date_match = date_pattern.search(remaining_text)
            if date_match:
                date_found_in_segment = date_match.group(1)
                current_date = date_found_in_segment
                remaining_text = date_pattern.sub("", remaining_text).strip()
                print(f"  Found and set date: {current_date}")

            if current_date is None:
                print(f"  Warning: Skipping segment {i+1} because a date has not been found yet. Text: '{text}'")
                continue

            amount_match = amount_pattern.search(remaining_text)
            if amount_match:
                raw_amount_str = amount_match.group(1)
                cleaned_amount_str = raw_amount_str.replace("SGD", "").replace("+", "").replace(" ", "")
                
                try:
                    amount_float = float(cleaned_amount_str)
                    formatted_amount = f"{amount_float:.2f}"
                except ValueError:
                    print(f"  Warning: Could not convert amount '{cleaned_amount_str}' to a number. Skipping.")
                    continue

                description_raw = amount_pattern.sub("", remaining_text).strip()
                description = " | ".join(line.strip() for line in description_raw.split('\n') if line.strip())

                self.transactions.append({
                    "Date": current_date,
                    "description": description,
                    "amount": formatted_amount
                })
                print(f"  Parsed transaction: {description} | {formatted_amount}")
            else:
                if not date_found_in_segment:
                    clean_text = text.replace('\n', ' ')
                    print(f"  Skipping segment {i+1} (unrecognized format, no amount found): '{clean_text}'")

        print(f"\nSuccessfully parsed {len(self.transactions)} transactions.")
        return self.transactions
