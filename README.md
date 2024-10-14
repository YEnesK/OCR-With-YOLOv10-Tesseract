## YOLOv10 and Tesseract OCR Document Analysis

This project combines YOLOv10 object detection and Tesseract OCR to analyze documents, detect layout elements, and extract text from images. It's particularly useful for processing structured documents with text and tables.

## Features

- Uses YOLOv10 for detecting document layout elements (text, tables, pictures)
- Applies Tesseract OCR for text extraction
- Handles table detection and extraction
- Processes multiple images in a folder
- Supports Turkish language OCR

## Requirements

- Python 3.7+
- OpenCV
- Supervision
- Ultralytics YOLOv10
- Pytesseract
- img2table

## Setup

1. Install the required packages:

pip install pytesseract opencv-python supervision ultralytics img2table

pip install -q git+https://github.com/THU-MIG/yolov10.git

2. Install Tesseract OCR:
   
For Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki

For Ubuntu: sudo apt install tesseract-ocr-tur

3. Set up the Tesseract path in the script:

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

## Usage

1. Place your images in the input folder:

folder_path = ''

2. Set the output folder for OCR results:

results_folder_path = ''

3. Run the script:

python main.py

## Key Functions

- main(): Initializes the YOLOv10 model and processes all images in the specified folder
- process_files_in_folder(): Iterates through images in the input folder
- layout_process(): Applies YOLOv10 for layout detection and initiates OCR
- filter_detections(): Refines detected layout elements
- OCR_Result(): Performs OCR on detected text areas
- table_process(): Handles table extraction and OCR

## Note

This project uses a pre-trained YOLOv10 model (yolov10x_best.pt). Ensure you have this model file in your project directory.

## Contributor

Yusuf Enes KURT
