import re
import pandas as pd
from .base_parser import TransactionParser

class TrustBankParser(TransactionParser):
    """
    Parses transactions from Trust Bank statements using layout-aware OCR data.
    """
    requires_dividers = False

    def _clean_ocr_data(self, ocr_data):
        """Cleans the raw OCR data from Tesseract."""
        # Remove entries with no text or low confidence
        data = ocr_data[ocr_data.conf > 30]
        data = data[data.text.notna()]
        data['text'] = data['text'].str.strip()
        data = data[data.text != '']
        return data

    def _group_by_lines(self, df):
        """Groups words into lines of text based on their coordinates."""
        lines = {}
        for _, row in df.iterrows():
            line_key = (row['block_num'], row['line_num'])
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(row)
        
        # Sort words within each line by their x-coordinate
        for key in lines:
            lines[key].sort(key=lambda x: x['left'])
            
        return lines

    def parse(self):
        """
        Main method to perform OCR on the full image data and extract transactions.
        """
        print("Step 4: Parsing full-page OCR data with TrustBankParser...")
        
        ocr_data = self.data.get('ocr_data')
        if ocr_data is None:
            print("  Error: TrustBankParser requires full OCR data but none was provided.")
            return []

        # 1. Clean and preprocess the OCR data
        df = self._clean_ocr_data(ocr_data)
        lines = self._group_by_lines(df)

        # 2. Define patterns for matching
        # Matches "Sun, 3 Aug 2025" or "Thu, 31 Jul 2025"
        date_pattern = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s\d{1,2}\s\w{3}\s\d{4}$")
        # Matches amounts like "S$12.34" or "-S$5.00"
        price_pattern = re.compile(r"S\$([\d,]+\.\d{2})")
        
        current_date = None
        
        # 3. Iterate through lines to structure the data
        line_texts = [" ".join(word['text'] for word in words) for words in lines.values()]

        for line_text in line_texts:
            # Check if the line is a date header
            if date_pattern.match(line_text):
                # Format date from "Sun, 3 Aug 2025" to "03 Aug"
                parts = line_text.split(' ')
                day = parts[1].zfill(2)
                month = parts[2]
                current_date = f"{day} {month}"
                print(f"  Found date section: {current_date}")
                continue

            # If we haven't found a date yet, skip
            if not current_date:
                continue

            # Check if the line is a transaction
            price_match = price_pattern.search(line_text)
            if price_match:
                # This line is likely a transaction title and its price
                
                # Extract amount
                amount_str = price_match.group(1).replace(',', '')
                # Check for a negative sign in the text preceding the amount
                if '-' in line_text.split(price_match.group(0))[0]:
                    amount = f"-{amount_str}"
                else:
                    amount = amount_str

                # Extract description (everything before the price)
                description = price_pattern.sub("", line_text).strip()

                # Simple approach: Assume the line is the main description
                # More complex logic could analyze the line below for a sub-description
                
                self.transactions.append({
                    "Date": current_date,
                    "description": description,
                    "amount": amount
                })
                print(f"  Parsed transaction: {description} | {amount}")

            # Ignore irrelevant lines (like "Pay over...", "Split now", etc.)
            elif "Pay over" in line_text or "Split now" in line_text:
                print(f"  Ignoring widget: {line_text}")
                continue

        print(f"\nSuccessfully parsed {len(self.transactions)} transactions.")
        return self.transactions
