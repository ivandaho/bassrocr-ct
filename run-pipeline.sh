screen_recording_file="put_screen_recordings_here/one.mp4"

transactions_file="transactions_uncategorized.csv"
### TODO: add type, evol
python stitcher.py "$screen_recording_file" stitched_output.png
python ocr_parser.py stitched_output.png "$transactions_file" --debug-greyscale --debug-segments
python categorize_transactions.py transaction_model.pkl "$transactions_file" transactions_categorized.csv
