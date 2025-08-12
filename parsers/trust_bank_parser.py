import os
from .base_parser import TransactionParser

class TrustBankParser(TransactionParser):
    """
    Parses transactions from Trust Bank statement screenshots.
    This is a placeholder and needs to be implemented.
    """

    def parse(self):
        """
        Main method to perform OCR on segments and extract transaction data.
        This method needs to be implemented with the specific logic for Trust Bank.
        """
        print("Step 4: Parsing segments with TrustBankParser...")
        print("  WARNING: TrustBankParser is not yet implemented.")
        print("  This is a placeholder. No transactions will be parsed.")

        # ==================================================================
        # TODO: Implement the parsing logic for Trust Bank statements here.
        #
        # You will need to:
        # 1. Loop through `self.segments` (a list of image crops).
        # 2. Use `pytesseract.image_to_string()` on each segment.
        # 3. Use regular expressions or other string manipulation to find
        #    the 'Date', 'description', and 'amount' for each transaction.
        # 4. Append a dictionary for each transaction to `self.transactions`.
        #    e.g., self.transactions.append({
        #              "Date": "15 Aug",
        #              "description": "FAIRPRICE FINEST",
        #              "amount": "-50.25"
        #          })
        # ==================================================================

        # For now, it returns an empty list.
        self.transactions = []
        return self.transactions
