# PDF Question Answering System

An intelligent PDF document analysis tool that allows users to ask questions about PDF content and receive relevant answers using natural language processing and machine learning.

## Features

- **PDF Text Extraction**: Converts PDF documents to text using OCR (Optical Character Recognition)
- **Interactive Question-Answering**: Ask questions about the PDF content in natural language
- **Smart Text Processing**: Breaks down documents into meaningful chunks for better analysis
- **Real-time Progress Tracking**: Shows page-by-page processing progress
- **Modern GUI Interface**: User-friendly interface built with Tkinter
- **Confidence Scoring**: Displays confidence levels for each answer
- **Multi-Threading**: Handles PDF processing in the background while keeping the UI responsive

## Requirements

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`):
  - torch
  - transformers
  - numpy
  - scikit-learn
  - pdf2image
  - pytesseract
  - Pillow

Additionally, you need to have Tesseract OCR installed on your system:
- For macOS: `brew install tesseract`
- For Linux: `sudo apt-get install tesseract-ocr`
- For Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

1. Clone this repository or download the source code
2. Install the required Python packages:
```bash
pip install -r requirements.txt
```
3. Ensure Tesseract OCR is installed on your system
4. Run the application:
```bash
python ui.py
```

## Usage

1. Launch the application by running `ui.py`
2. Click "Select PDF" to choose a PDF document
3. Wait for the document to be processed (progress will be shown)
4. Type your question in the input field
5. Click "Ask Question" to get relevant information from the document
6. View results with confidence scores in the answer display area

## How It Works

1. **PDF Processing**:
   - Converts PDF pages to images
   - Extracts text using OCR
   - Breaks text into meaningful chunks

2. **Question Answering**:
   - Uses BERT-based model (MPNet) for text embeddings
   - Computes similarity between question and document chunks
   - Returns most relevant passages with confidence scores

3. **Text Processing**:
   - Removes noise and duplicates
   - Splits long paragraphs into manageable chunks
   - Filters out low-quality content

## Technical Details

- Uses the `sentence-transformers/all-mpnet-base-v2` model for text embeddings
- Implements cosine similarity for matching relevant content
- Employs multi-threading for background processing
- Minimum similarity threshold of 0.3 for quality control
- Maximum chunk size of 150 words for focused context

## Error Handling

The system includes robust error handling for:
- Invalid PDF files
- OCR processing failures
- File access issues
- Text extraction problems
- Query processing errors

