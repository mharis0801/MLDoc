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

# Load the BERT model and tokenizer
print("Loading the language model...")
model_name = 'sentence-transformers/all-mpnet-base-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path, progress_callback=None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    print(f"Converting PDF to images: {pdf_path}")
    
    # Create a temporary directory to store images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert all pages of the PDF
        images = convert_from_path(pdf_path)
        text = ""
        total_pages = len(images)
        print(f"Total pages found: {total_pages}")
        
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{total_pages}")
            if progress_callback:
                progress_callback(i + 1, total_pages)
            try:
                # Extract text from image using OCR
                page_text = pytesseract.image_to_string(image)
                if page_text:
                    text += page_text + "\n\n"
            except Exception as e:
                print(f"Warning: Could not process page {i+1}: {str(e)}")
                continue
                
    if not text.strip():
        raise ValueError("No text could be extracted from the PDF")
        
    return text

def preprocess_text(text):
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into meaningful chunks using a combination of
    # paragraph breaks and sentence boundaries
    chunks = []
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if len(para.split()) < 5:  # Lower threshold for minimum words
            continue
            
        # Split long paragraphs into smaller chunks
        if len(para.split()) > 150:  # Reduced chunk size for more focused context
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk.split()) + len(sentence.split()) < 150:
                    current_chunk += " " + sentence
                else:
                    if len(current_chunk.split()) > 5:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    
            if current_chunk and len(current_chunk.split()) > 5:
                chunks.append(current_chunk.strip())
        else:
            chunks.append(para)
    
    # Remove noise and duplicates
    chunks = list(dict.fromkeys([  # Remove duplicates while preserving order
        chunk for chunk in chunks 
        if len(chunk.split()) > 5 and
        not chunk.isdigit() and
        not all(c.isdigit() or c.isspace() for c in chunk)
    ]))
    
    print(f"Created {len(chunks)} text chunks for processing")
    return chunks

def get_embedding(text):
    # Tokenize and get embeddings
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling to get text embedding
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def find_relevant_info(query, chunks, top_k=3):
    try:
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Get embeddings for all chunks
        chunk_embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            try:
                embedding = get_embedding(chunk)
                chunk_embeddings.append(embedding)
                valid_chunks.append(chunk)
            except Exception as e:
                print(f"Warning: Could not process chunk: {str(e)}")
                continue
        
        if not chunk_embeddings:
            raise ValueError("No valid chunk embeddings generated")
        
        # Calculate similarities with higher precision
        similarities = [cosine_similarity(query_embedding.reshape(1, -1), 
                                        chunk_emb.reshape(1, -1))[0][0] 
                       for chunk_emb in chunk_embeddings]
        
        # Filter out low-confidence results
        min_similarity = 0.3  # Minimum similarity threshold
        filtered_results = [
            (chunk, sim) for chunk, sim in zip(valid_chunks, similarities)
            if sim > min_similarity
        ]
        
        # Sort by similarity and get top k
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