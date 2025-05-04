import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from ML import extract_text_from_pdf, preprocess_text, find_relevant_info, validate_pdf_path
import threading
import os
import queue
import time

class PDFQuestionAnswerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Question Answering System")
        self.root.geometry("800x600")
        
        # Variables
        self.pdf_path = None
        self.chunks = None
        self.processing = False
        self.total_pages = 0
        self.current_page = 0
        self.page_progress = None
        self.processing_queue = queue.Queue()
        self.last_question = None
        self.last_result = None
        
        # Configure threading
        self.max_workers = os.cpu_count()
        self.processing_thread = None
        
        # Style configuration
        self.configure_styles()
        
        # Create main container
        self.main_frame = ttk.Frame(root, style='Modern.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create UI components
        self.create_pdf_selection()
        self.create_question_input()
        self.create_answer_display()
        self.create_status_bar()
        
        # Bind events
        self.question_entry.bind('<Return>', lambda e: self.ask_question())
        
    def configure_styles(self):
        style = ttk.Style()
        style.configure('Modern.TButton', padding=10)
        style.configure('Modern.TFrame', padding=10)
        style.configure('Progress.Horizontal.TProgressbar',
                       troughcolor='#E0E0E0',
                       background='#4CAF50')
        
    def create_pdf_selection(self):
        pdf_frame = ttk.Frame(self.main_frame)
        pdf_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.pdf_label = ttk.Label(pdf_frame, text="No PDF selected", wraplength=500)
        self.pdf_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.select_button = ttk.Button(
            pdf_frame, 
            text="Select PDF", 
            command=self.select_pdf,
            style='Modern.TButton'
        )
        self.select_button.pack(side=tk.RIGHT)
        
    def create_question_input(self):
        question_frame = ttk.Frame(self.main_frame)
        question_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(question_frame, text="Enter your question:").pack(anchor=tk.W)
        
        self.question_entry = ttk.Entry(question_frame)
        self.question_entry.pack(fill=tk.X, pady=(5, 0))
        
        self.ask_button = ttk.Button(
            question_frame,
            text="Ask Question",
            command=self.ask_question,
            style='Modern.TButton'
        )
        self.ask_button.pack(pady=(5, 0), anchor=tk.E)
        
    def create_answer_display(self):
        answer_frame = ttk.Frame(self.main_frame)
        answer_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(answer_frame, text="Answers:").pack(anchor=tk.W)
        
        self.answer_text = scrolledtext.ScrolledText(
            answer_frame,
            wrap=tk.WORD,
            height=15
        )
        self.answer_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
    def create_status_bar(self):
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_bar.pack(fill=tk.X)
        
        # Main progress bar for overall processing
        self.progress = ttk.Progressbar(
            status_frame,
            mode='indeterminate'
        )
        
        # Page progress frame
        self.page_progress_frame = ttk.Frame(status_frame)
        self.page_progress_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Page counter label
        self.page_count_var = tk.StringVar()
        self.page_count_label = ttk.Label(
            self.page_progress_frame,
            textvariable=self.page_count_var
        )
        self.page_count_label.pack(fill=tk.X)
        
        # Page progress bar
        self.page_progress = ttk.Progressbar(
            self.page_progress_frame,
            mode='determinate'
        )
        self.page_progress.pack(fill=tk.X, pady=(2, 0))
        
    def update_page_progress(self, current, total):
        if not self.processing:
            return
        
        self.current_page = current
        self.total_pages = total
        
        # Calculate processing speed
        current_time = time.time()
        if hasattr(self, 'start_time'):
            elapsed_time = current_time - self.start_time
            pages_per_second = current / elapsed_time if elapsed_time > 0 else 0
            eta = (total - current) / pages_per_second if pages_per_second > 0 else 0
            
            # Update progress display with speed and ETA
            self.page_count_var.set(
                f"Processing page {current} of {total} "
                f"({pages_per_second:.1f} pages/sec, ETA: {eta:.0f}s)"
            )
        else:
            self.start_time = current_time
            self.page_count_var.set(f"Processing page {current} of {total}")
        
        self.page_progress["value"] = (current / total) * 100
        self.root.update_idletasks()
        
    def select_pdf(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf")]
        )
        if file_path:
            if validate_pdf_path(file_path):
                self.pdf_path = file_path
                self.pdf_label.config(
                    text=f"Selected: {os.path.basename(file_path)}"
                )
                self.process_pdf()
            else:
                messagebox.showerror(
                    "Error",
                    "Invalid PDF file selected!"
                )
                
    def process_pdf(self):
        def process():
            try:
                self.status_var.set("Processing PDF...")
                self.progress.pack(fill=tk.X, pady=(5, 0))
                self.progress.start()
                self.page_progress_frame.pack(fill=tk.X, pady=(5, 0))
                self.processing = True
                self.start_time = time.time()
                
                # Extract text with progress updates
                text = extract_text_from_pdf(self.pdf_path, self.update_page_progress)
                
                # Process text in chunks for better responsiveness
                self.status_var.set("Analyzing document structure...")
                self.chunks = preprocess_text(text)
                
                process_time = time.time() - self.start_time
                self.status_var.set(
                    f"Ready! Processed {self.total_pages} pages in {process_time:.1f}s. "
                    f"Found {len(self.chunks)} text segments."
                )
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", str(e))
            finally:
                self.processing = False
                self.progress.stop()
                self.progress.pack_forget()
                self.page_progress_frame.pack_forget()
                
        # Use daemon thread for background processing
        self.processing_thread = threading.Thread(target=process, daemon=True)
        self.processing_thread.start()
        
    def ask_question(self):
        if not self.pdf_path or not self.chunks:
            messagebox.showwarning(
                "Warning",
                "Please select a PDF file first!"
            )
            return
            
        if self.processing:
            messagebox.showwarning(
                "Warning",
                "Please wait while the current PDF is being processed!"
            )
            return
            
        query = self.question_entry.get().strip()
        if not query:
            messagebox.showwarning(
                "Warning",
                "Please enter a question!"
            )
            return
            
        # Check if this is the same question as last time
        if query == self.last_question and self.last_result:
            self.display_results(self.last_result)
            return
            
        def process_question():
            self.status_var.set("Searching for relevant information...")
            self.progress.pack(fill=tk.X, pady=(5, 0))
            self.progress.start()
            self.processing = True
            start_time = time.time()
            
            try:
                relevant_info = find_relevant_info(query, self.chunks)
                
                if not relevant_info:
                    self.answer_text.delete(1.0, tk.END)
                    self.answer_text.insert(
                        tk.END,
                        "No relevant information found. Please try rephrasing your question."
                    )
                else:
                    # Cache the results
                    self.last_question = query
                    self.last_result = relevant_info
                    self.display_results(relevant_info)
                
                search_time = time.time() - start_time
                self.status_var.set(f"Found {len(relevant_info)} results in {search_time:.2f}s")
                
            except Exception as e:
                self.answer_text.delete(1.0, tk.END)
                self.answer_text.insert(
                    tk.END,
                    f"An error occurred: {str(e)}"
                )
                self.status_var.set("Error occurred")
                
            finally:
                self.processing = False
                self.progress.stop()
                self.progress.pack_forget()
        
        thread = threading.Thread(target=process_question, daemon=True)
        thread.start()
    
    def display_results(self, results):
        """Display search results with formatting"""
        self.answer_text.delete(1.0, tk.END)
        
        for i, (chunk, similarity) in enumerate(results, 1):
            # Format confidence score as percentage
            confidence = similarity * 100
            
            # Add result header with confidence
            self.answer_text.insert(
                tk.END,
                f"Result {i} (Confidence: {confidence:.1f}%)\n",
                "header"
            )
            self.answer_text.insert(tk.END, "─" * 80 + "\n")
            
            # Format the chunk text for better readability
            formatted_chunk = chunk.strip()
            self.answer_text.insert(tk.END, f"{formatted_chunk}\n\n")
        
        # Configure text tags for styling
        self.answer_text.tag_configure(
            "header",
            font=("Helvetica", 10, "bold")
        )

def main():
    root = tk.Tk()
    app = PDFQuestionAnswerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()