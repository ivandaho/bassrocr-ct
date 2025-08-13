from datetime import datetime
import re
from .base_parser import TransactionParser
LEFT_THRESHOLD = 82

class TrustBankParser(TransactionParser):
    """
    Parses transactions from Trust Bank statements using layout-aware OCR data.
    This parser does not use dividers and instead analyzes the coordinates
    of the OCR text to group elements into transactions.
    """
    requires_dividers = False

    def _clean_ocr_data(self, ocr_data):
        """Cleans the raw OCR data from Tesseract."""
        data = ocr_data[ocr_data.conf > 14].copy()
        data = data[data.text.notna()]
        data['text'] = data['text'].str.strip()
        data = data[data.text != '']
        return data

    def _group_lines(self, df):
        """Groups words into lines and provides coordinate and text data for each line."""
        lines = {}
        # Group words by block, paragraph, and line number
        for _, row in df.iterrows():
            line_key = (row['block_num'], row['par_num'], row['line_num'])
            if line_key not in lines:
                lines[line_key] = {
                    'words': [],
                    'height_sum': 0,
                    'word_count': 0
                }
            lines[line_key]['words'].append(row)
            lines[line_key]['height_sum'] += row['height']
            lines[line_key]['word_count'] += 1
        
        line_data = []
        for key in sorted(lines.keys()):
            line_info = lines[key]
            words = sorted(line_info['words'], key=lambda x: x['left'])
            
            if not words:
                continue

            text = " ".join(word['text'] for word in words)
            avg_height = line_info['height_sum'] / line_info['word_count'] if line_info['word_count'] > 0 else 0
            
            line_data.append({
                'text': text,
                'words': words,
                'y_pos': words[0]['top'],
                'left': words[0]['left'],
                'avg_height': avg_height,
            })
        return line_data

    def parse(self):
        """
        Main method to perform OCR on the full image data and extract transactions.
        """
        print("Step 4: Parsing full-page OCR data with TrustBankParser...")
        
        ocr_data = self.data.get('ocr_data')
        if ocr_data is None:
            print("  Error: TrustBankParser requires full OCR data but none was provided.")
            return []

        df = self._clean_ocr_data(ocr_data)
        lines = self._group_lines(df)
        
        date_pattern = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s\d{1,2}\s\w{3}\s\d{4}$")
        price_pattern = re.compile(r"(\+?[\d,]+\.\d{2})$")
        
        current_date = None
        self.transactions = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if date_pattern.match(line['text']):
                current_date = line['text']
                i += 1
                continue

            if not current_date:
                # keep searching until a date is found
                i += 1
                continue

            # if "Pay over 18 months" in line['text'] or "Split now" in line['text']:
            #     i += 1
            #     continue

            price_match = price_pattern.search(line['text'])
            if not price_match:
                i += 1
                continue

            amount_str = price_match.group(0)
            title = price_pattern.sub("", line['text']).strip()
            
            description = ""
            
            if i + 1 < len(lines):
                next_line = lines[i+1]
                
                # is_smaller = next_line['avg_height'] < (line['avg_height'] * 0.95)
                is_indented = next_line['left'] > line['left'] # this will also be False if next line is date, or if next line is widget. might be able to refactor and simplify
                is_at_least_x = next_line['left'] > LEFT_THRESHOLD # ignore text in icons
                is_widget = "Pay over 18 months" in next_line['text'] or "Split now" in next_line['text']
                is_date = date_pattern.match(next_line['text'])

                if is_indented and is_at_least_x and not is_widget and not is_date:
                    description = next_line['text']
                    i += 1
            
            hasPlus = amount_str.find("+") > -1
            cleaned_amount = amount_str
            if not hasPlus:
                cleaned_amount = "-" + amount_str

            
            full_description = title
            if description:
                full_description += " | " + description

            if cleaned_amount:
                try:
                    amount_float = float(cleaned_amount)
                    padded_date: str = current_date[5:]
                    is_single_digit_day =  padded_date.find(" ", 1, 2)
                    if not is_single_digit_day:
                        padded_date = "0" + current_date[5:]
                        
                    self.transactions.append({
                        "date": datetime.strptime(padded_date, "%d %b %Y"),
                        "description": full_description,
                        "amount": f"{amount_float:.2f}"
                    })
                except ValueError:
                    print(f"  Warning: Could not convert amount '{cleaned_amount}' to a number. Skipping transaction: '{title}'")
            
            i += 1

        print(f"\nSuccessfully parsed {len(self.transactions)} transactions.")
        return self.transactions
