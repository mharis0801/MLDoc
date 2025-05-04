# PDF Question Answering System

An intelligent PDF document analysis tool that allows users to ask questions about PDF content and receive relevant answers using natural language processing and machine learning, enhanced with AI-powered analysis.

## Features

- **PDF Text Extraction**: Converts PDF documents to text using OCR (Optical Character Recognition)
- **Interactive Question-Answering**: Ask questions about the PDF content in natural language
- **Smart Text Processing**: Breaks down documents into meaningful chunks for better analysis
- **Real-time Progress Tracking**: Shows page-by-page processing progress with speed and ETA
- **Modern GUI Interface**: User-friendly interface built with Tkinter
- **Confidence Scoring**: Displays confidence levels for each answer
- **Multi-Threading**: Handles PDF processing in the background while keeping the UI responsive
- **Gemini AI Integration**: Optional AI-powered analysis of document content
- **Result Caching**: Speeds up repeated queries and PDF processing
- **Parallel Processing**: Utilizes multiple CPU cores for faster processing

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
  - google-generativeai
  - tqdm
  - psutil
  - python-Levenshtein

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
4. (Optional) Get a Google Gemini API key for AI features
5. Run the application:
```bash
python ui.py
```

## Usage

1. Launch the application by running `ui.py`
2. (Optional) Enter your Gemini API key and toggle AI features
3. Click "Select PDF" to choose a PDF document
4. Wait for the document to be processed (progress will be shown)
5. Type your question in the input field
6. Click "Ask Question" to get relevant information from the document
7. View results with confidence scores and optional AI analysis

## How It Works

1. **PDF Processing**:
   - Converts PDF pages to images
   - Extracts text using OCR with optimized parameters
   - Breaks text into meaningful chunks
   - Caches processed text for faster future access

2. **Question Answering**:
   - Uses BERT-based model (MPNet) for text embeddings
   - Computes similarity between question and document chunks
   - Returns most relevant passages with confidence scores
   - Optionally provides AI-powered analysis using Gemini

3. **Text Processing**:
   - Removes noise and duplicates
   - Splits long paragraphs into manageable chunks
   - Filters out low-quality content
   - Processes text in parallel for better performance

## Technical Details

- Uses the `sentence-transformers/all-mpnet-base-v2` model for text embeddings
- Implements cosine similarity for matching relevant content
- Employs multi-threading and parallel processing
- Minimum similarity threshold of 0.3 for quality control
- Maximum chunk size of 150 words for focused context
- Caches embeddings and PDF text for improved performance
- Asynchronous handling of AI responses
- GPU acceleration when available

## Performance Optimizations

- **Parallel Processing**: Utilizes multiple CPU cores for PDF processing and text analysis
- **Caching System**: 
  - PDF text caching to avoid repeated OCR processing
  - Embedding caching for faster similarity matching
  - Question result caching for instant repeated queries
- **GPU Support**: Automatically uses GPU for model inference when available
- **Optimized OCR**: Custom Tesseract configuration for better speed/accuracy trade-off
- **Asynchronous Operations**: Non-blocking AI responses and background processing

## Error Handling

The system includes robust error handling for:
- Invalid PDF files
- OCR processing failures
- File access issues
- Text extraction problems
- Query processing errors
- AI service connectivity issues
- API key validation
- Cache management errors

## Configuration

- **API Key Management**: Secure storage and management of Gemini API keys
- **AI Toggle**: Enable/disable AI features as needed
- **Progress Tracking**: Real-time processing speed and ETA display
- **Cache Location**: Automatically managed in user's home directory



