from abc import ABC, abstractmethod

class TransactionParser(ABC):
    """
    Abstract base class for all statement parsers.
    Ensures that every parser implements a consistent interface.
    """

    def __init__(self, segments, debug=False):
        """
        Initializes the parser with the image segments.

        Args:
            segments (list): A list of image segments (as numpy arrays) representing potential transactions.
            debug (bool): Flag to enable or disable debugging output.
        """
        self.segments = segments
        self.debug = debug
        self.transactions = []

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
