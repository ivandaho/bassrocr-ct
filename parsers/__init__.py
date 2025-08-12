from .uob_parser import UOBParser
from .trust_bank_parser import TrustBankParser

# A mapping of statement type keys (from config.yml) to their corresponding parser classes.
PARSER_MAPPING = {
    "uobone": UOBParser,
    "uobevol": UOBParser,
    "trust": TrustBankParser,
}

def get_parser(statement_type, **kwargs):
    """
    Factory function to get the appropriate parser instance.

    Args:
        statement_type (str): The type of statement (e.g., 'uobone', 'trust').
        **kwargs: A dictionary of arguments to pass to the parser's constructor
                  (e.g., debug=True, segments=[...], or ocr_data={...}).

    Returns:
        An instance of a TransactionParser subclass.
    
    Raises:
        ValueError: If the statement_type is not supported.
    """
    parser_class = PARSER_MAPPING.get(statement_type)
    
    if not parser_class:
        raise ValueError(f"Unsupported statement type: '{statement_type}'. "
                         f"Supported types are: {list(PARSER_MAPPING.keys())}")
                         
    # Pass the keyword arguments directly to the parser's constructor
    return parser_class(**kwargs)
