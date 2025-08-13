# BASSROCR-CT
#### Bank App Scrolling Screen Recording -> OCR -> Categorized Transactions

## Introduction

This project is a tool to aid in **extracting, parsing, and categorizing financial transactions from a screen recording of a mobile banking application**. The full pipeline will 1) stitch a scrolling video into a single image, 2) perform OCR to extract transaction data, and 3) use a machine learning model to categorize the results. Each step has it's own python script that can be ran alone.

If you want to automatically categorize transactions, no pre-trained model is included (but you can train one with `trainer.py`), and some small tweaks might be required for this to work for your own use case.

For my own model, I had already categorized 11k transactions over many years over different accounts, and fed it in a .csv file to train my own model. It took only seconds. The results were satisfactory.

This code is provided as-is. Disclaimer: about half of it, including this readme file, is vibe coded, mostly with Google Gemini's free tier. However, as expected, lots of fixes and tweaks were needed and I am in no way advocating AI coding tools as a standalone solution to any problem.

This was created not as a replacement for any existing tool you might be using for your own finances, but merely as a way to export the data hoarded by your bank, usually in proprietary apps, or .pdf reports only at specific intervals, to a useful .csv format.

The idea is that, with this tool, as long as you have access to your bank app, and can provide a screen recording, you should be able to export the data in .csv format.

## Features

- **Video Stitching**: Creates a single, tall screenshot from a scrolling screen recording.
- **OCR Parsing**: Extracts transaction details (date, description, amount) from the stitched image.
- **Automated Categorization**: Uses a pre-trained model to assign a category (e.g., Food, Shopping) to each transaction.
- **Locally Train Categorization Model**: A script is provided to train your own .pkl file for use in the automatic categorization step.
- **Configurable**: (for Video Stitching and Ocr Parsing) ease of configuration for different sources via a YAML configuration file. Currently supports:
    - `uobone` - UOB One Account
    - `uobevol`- UOB Evol Credit Card (might also support other UOB Credit Cards)
    - `trust` - Trust Bank

## Workflow

The entire process is orchestrated by the `run-pipeline.sh` script, which executes the following steps:

1.  **`stitcher.py`**:
  - Takes a source video file as input.
  - Extracts keyframes where significant scrolling occurs.
  - Stitches these frames together based on a configured Region of Interest (ROI) to produce a single, long image (`stitched_output.png`).

2.  **`ocr_parser.py`**:
  - Takes the stitched image as input.
  - Preprocesses the image (e.g., grayscaling, thresholding).
  - Based on the statement type configuration, it either:
    - Detects horizontal lines to segment the image into individual transaction rows.
    - Performs full-page OCR to get text along with layout coordinates.
  - Uses the appropriate parser (e.g. `UOBParser` or `TrustBankParser`) to extract structured data.
  - Saves the extracted, uncategorized transactions to a CSV file (`transactions_uncategorized.csv`).

3.  **`categorize_transactions.py`**:
  - Loads the uncategorized transactions from the CSV file.
  - Uses a pre-trained machine learning model (`transaction_model.pkl`) to predict a category for each transaction. No model is provided, you will need to train your own model - instructions below.
  - Saves the final, categorized data to an output CSV file (default: `transactions_categorized.csv`).

## Prerequisites

- Python 3.x
- Tesseract OCR Engine
- FFmpeg (recommended for video processing)

## Installation

1.  **Clone the repository.**

2.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Tesseract:**
    -   **macOS:** `brew install tesseract`
    -   **Ubuntu:** `sudo apt-get install tesseract-ocr`

## Usage

The main entrypoint is the `run-pipeline.sh` script, but you may run the individual python scripts (see Workflow section above) for each step as needed.

```bash
./run-pipeline.sh <path_to_recording> [output_file] [options]
```

### Arguments & Options:

- `<path_to_recording>`: **(Required)** The path to your screen recording video file.
- `[output_file]`: (Optional) The path for the final categorized CSV file. Defaults to `transactions_categorized.csv`.
- `-s, --statement-type <type>`: Sets the statement type. This is needed for selecting the correct configuration and parser. Supported types are `uobone`, `uobevol`, and `trust`.
- `--debug-roi`: Saves a debug image of the detected Region of Interest and exits.
- `--debug-keyframes`: Saves all extracted video keyframes to the `debug_keyframes/` directory.
- `--debug-segments`: Saves intermediate image segments during OCR (1 segment = 1 transaction) for `uobone`, `uobevol`, and saves full OCR data for `trust`


### Examples:

**Process a UOB One statement video:**
```bash
./run-pipeline.sh input/uob_one_video.mp4 -s uobone
```

**Process a Trust Bank statement video and save to a custom file:**
```bash
./run-pipeline.sh input/trust_video.mov categorized_trust.csv -s trust
```

## Configuration

The file `config.yml` contains all the settings for the pipeline, broken down by statement type. This allows you to adjust parameters without changing the code.

- **`theme`**: `light` or `dark`. Affects image preprocessing.
- **`roi`**: The `x,y,w,h` coordinates of the scrolling area in the video. use `--debug-roi` to find out what works for your phone. The ROI should only cover the main scrollable areas of the screen displaying transactions, and it can cover less area than the full scrollable area and still work, but should not be too small.
- **`stitcher_diff_threshold`**: The sensitivity for detecting new keyframes. Tweak this if images are not stitched correctly.
- **`parser_greyscale_threshold`**: The threshold used for image binarization during OCR. Only used to detect segment lines.

## Parsers

The project uses a factory pattern to handle different statement layouts. The `base_parser.py` defines the interface, and concrete classes implement the specific logic.

- **`UOBParser`**: Uses `requires_dividers = True`. It relies on `ocr_parser.py` to segment the image by horizontal lines and then it parses the text from each segment.
- **`TrustBankParser`**: Uses `requires_dividers = False`. It receives full OCR data (text and coordinates) from `ocr_parser.py` and reconstructs transactions based on the text layout (e.g., indentation, position).

## Training your own model for `categorize_transactions.py`

You will need to train your own model based on your own historical transaction data.

Most (good) financial management tools will have a function to export all (categorized) transactions, which you can use with `trainer.py`. The exported file will need to have these 3 columns for training: `category`, `amount`, and `description`.

```bash
python ./trainer.py my-exported-transactions.csv transaction_model.pkl
```

I highly recommend reading `trainer.py` and modifying the code to suit your own needs.

## Notes
- No warranty provided. I am a webdev. I barely know python and ML related stuff, and I vibe coded the stuff I did not have knowledge of.
- Tested with screen recordings from an iPhone 15 Pro. Trust bank stitcher + OCR was done in dark mode.
- If you already have a way to stitch a long screenshot, don't bother with step 1. `stitcher.py` is because I couldn't find a free and reliable way of doing it on iOS. Same for the other steps, just run the individual scripts as needed.
- for the OCR step, make sure that the input image only contains transaction data, and no leftover additional text or UI elements from the app.
- some bank apps block screenshots and/or screen recordings. In that case, sorry. For Trust bank, I used iPhone Mirroring, and was able to record a screen recording with QuickTime Player.
