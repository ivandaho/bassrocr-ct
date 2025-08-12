from abc import ABC, abstractmethod

class TransactionParser(ABC):
    """
    Abstract base class for all statement parsers.
    Ensures that every parser implements a consistent interface.
    """
    
    # Class attribute to indicate if the parser needs pre-segmented images.
    # If False, the parser expects a full OCR data dictionary from Tesseract.
    requires_dividers = True

    def __init__(self, debug=False, **kwargs):
        """
        Initializes the parser.

        Args:
            debug (bool): Flag to enable or disable debugging output.
            **kwargs: Either 'segments' (list of images) or 'ocr_data' (dict).
        """
        self.debug = debug
        self.transactions = []
        # Store whatever data is passed (segments or ocr_data)
        self.data = kwargs

    @abstractmethod
    def parse(self):
        """
        Main method to perform OCR on segments and extract transaction data.
        This method must be implemented by all subclasses.
        It should populate the self.transactions list.
        """
        pass

    def get_transactions(self):
        """Returns the list of parsed transactions."""
        return self.transactions
