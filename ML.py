import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import tempfile
import concurrent.futures
import functools
import pickle
from pathlib import Path
import threading

# Load the BERT model and tokenizer
print("Loading the language model...")
model_name = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Cache directory for embeddings
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".pdf_qa_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(pdf_path):
    """Generate a unique cache file path for a PDF"""
    pdf_hash = str(hash(pdf_path + str(os.path.getmtime(pdf_path))))
    return os.path.join(CACHE_DIR, f"{pdf_hash}.pickle")

def process_image(image):
    """Process a single image with OCR using optimized settings"""
    try:
        # Convert image to high contrast black and white
        image = image.convert('L')
        
        # Enhance image contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Set up tesseract parameters with improved quotation handling
        custom_config = r"""--oem 1 --psm 6 
            -c tessedit_char_blacklist=©
            -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?@#$%^&*()_+-=<>[]{}|/\\:;'\"` \n"
            -c preserve_interword_spaces=1
            -c textord_old_xheight=0
            -c textord_fix_xheight_bug=1"""
        
        # Extract text with improved settings
        text = pytesseract.image_to_string(
            image,
            config=custom_config,
            lang='eng'  # Specify English language
        )
        
        # Clean up extracted text with improved handling of quotes
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\x20-\x7E\n]', '', text)  # Keep only printable ASCII and newlines
        text = re.sub(r'(?<=[.!?])\s', '\n', text)  # Add line breaks after sentences
        
        return text.strip()
    except Exception as e:
        print(f"Warning: Could not process image: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path, progress_callback=None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    print(f"Converting PDF to images: {pdf_path}")
    
    # Check cache first
    cache_path = get_cache_path(pdf_path)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                print("Using cached PDF text")
                return cached_data['text']
        except:
            print("Cache invalid, reprocessing PDF")
            pass

    # Create a temporary directory to store images
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Convert PDF to images with optimized parameters
            images = convert_from_path(
                pdf_path,
                dpi=300,  # Increased DPI for better quality
                thread_count=os.cpu_count(),
                use_pdftocairo=True,
                grayscale=True,
                size=(2000, None),  # Set max width while maintaining aspect ratio
                paths_only=False  # Ensure we get PIL Image objects
            )
            
            if not images:
                raise ValueError("No pages could be extracted from the PDF")
            
            total_pages = len(images)
            print(f"Total pages found: {total_pages}")
            
            # Process images in parallel with improved OCR settings
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_to_page = {
                    executor.submit(process_image, image): i 
                    for i, image in enumerate(images)
                }
                
                text_chunks = [""] * total_pages
                
                for future in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        text_chunks[page_num] = future.result()
                        if progress_callback:
                            progress_callback(page_num + 1, total_pages)
                    except Exception as e:
                        print(f"Warning: Could not process page {page_num + 1}: {str(e)}")
                        text_chunks[page_num] = ""  # Ensure empty string for failed pages
                
            # Filter and join text chunks
            text = "\n\n".join(chunk for chunk in text_chunks if chunk.strip())
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            # Cache the results
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({'text': text}, f)
            except Exception as e:
                print(f"Warning: Could not cache PDF text: {str(e)}")
            
            return text
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

@functools.lru_cache(maxsize=1000)
def get_embedding_cached(text):
    """Get embedding with caching"""
    return get_embedding(text)

def get_embedding(text):
    # Tokenize and get embeddings
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling to get text embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def preprocess_text(text):
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into meaningful chunks using a combination of
    # paragraph breaks and sentence boundaries
    chunks = []
    paragraphs = text.split('\n\n')
    
    # Process paragraphs in parallel
    def process_paragraph(para):
        para = para.strip()
        if len(para.split()) < 5:
            return []
            
        if len(para.split()) > 150:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            chunk_list = []
            
            for sentence in sentences:
                if len(current_chunk.split()) + len(sentence.split()) < 150:
                    current_chunk += " " + sentence
                else:
                    if len(current_chunk.split()) > 5:
                        chunk_list.append(current_chunk.strip())
                    current_chunk = sentence
                    
            if current_chunk and len(current_chunk.split()) > 5:
                chunk_list.append(current_chunk.strip())
            return chunk_list
        else:
            return [para]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        chunk_lists = list(executor.map(process_paragraph, paragraphs))
    
    # Flatten the list of chunks
    chunks = [chunk for sublist in chunk_lists for chunk in sublist]
    
    # Remove noise and duplicates
    chunks = list(dict.fromkeys([
        chunk for chunk in chunks 
        if len(chunk.split()) > 5 and
        not chunk.isdigit() and
        not all(c.isdigit() or c.isspace() for c in chunk)
    ]))
    
    print(f"Created {len(chunks)} text chunks for processing")
    return chunks

def find_relevant_info(query, chunks, top_k=3):
    try:
        # Get query embedding
        query_embedding = get_embedding_cached(query)
        
        # Get embeddings for all chunks in parallel
        chunk_data = []
        
        def process_chunk(chunk):
            try:
                embedding = get_embedding_cached(chunk)
                return (chunk, embedding)
            except Exception as e:
                print(f"Warning: Could not process chunk: {str(e)}")
                return None
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            chunk_data = list(executor.map(process_chunk, chunks))
        
        # Filter out failed chunks
        chunk_data = [data for data in chunk_data if data is not None]
        
        if not chunk_data:
            raise ValueError("No valid chunk embeddings generated")
        
        # Calculate similarities in parallel
        def calculate_similarity(chunk_info):
            chunk, embedding = chunk_info
            sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                embedding.reshape(1, -1)
            )[0][0]
            return (chunk, sim)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            similarities = list(executor.map(calculate_similarity, chunk_data))
        
        # Filter and sort results
        min_similarity = 0.3
        filtered_results = [
            (chunk, sim) for chunk, sim in similarities
            if sim > min_similarity
        ]
        
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:top_k]
        
    except Exception as e:
        print(f"Error finding relevant information: {str(e)}")
        return []

def validate_pdf_path(pdf_path):
    """Validate if the given path is a valid PDF file."""
    if not pdf_path.strip():
        return False
    if not os.path.exists(pdf_path):
        return False
    if not pdf_path.lower().endswith('.pdf'):
        return False
    return True

def main():
    while True:
        # Ask user for PDF path
        pdf_path = input("\nEnter the path to your PDF file (or 'quit' to exit): ").strip()
        
        if pdf_path.lower() == 'quit':
            break
            
        if not validate_pdf_path(pdf_path):
            print("\nError: Please provide a valid PDF file path!")
            continue
            
        try:
            # Extract and preprocess text
            print("\nExtracting text from PDF...")
            text = extract_text_from_pdf(pdf_path)
            print("\nPreprocessing text...")
            chunks = preprocess_text(text)
            print(f"\nReady to answer questions! Found {len(chunks)} text segments.")
            
            # Interactive query loop
            while True:
                query = input("\nEnter your question about the document (or 'back' to choose another PDF, 'quit' to exit): ").strip()
                
                if query.lower() == 'quit':
                    return
                if query.lower() == 'back':
                    break
                
                if not query:
                    print("Please enter a question!")
                    continue
                    
                print("\nSearching for relevant information...")
                relevant_info = find_relevant_info(query, chunks)
                
                if not relevant_info:
                    print("No relevant information found. Please try rephrasing your question.")
                    continue
                
                print("\nRelevant information found:\n")
                for i, (chunk, similarity) in enumerate(relevant_info, 1):
                    print(f"Result {i} (Confidence: {similarity:.2f})")
                    print("-" * 80)
                    print(f"{chunk}\n")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please ensure the PDF file exists and is readable.")

if __name__ == "__main__":
    main()