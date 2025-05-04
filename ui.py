import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from ML import extract_text_from_pdf, preprocess_text, find_relevant_info, validate_pdf_path
from gemini_helper import GeminiHelper
import threading
import os
import queue
import time
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

class PDFQuestionAnswerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Question Answering System")
        self.root.geometry("800x600")
        
        # Initialize Gemini AI
        self.gemini = GeminiHelper()
        self.use_ai = tk.BooleanVar(value=False)
        
        # Create async loop in a separate thread
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        self.loop = None
        self.loop_ready = threading.Event()
        
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
        
        # Wait for event loop to be ready
        self.loop_ready.wait()
        
        # Style configuration
        self.configure_styles()
        
        # Create main container
        self.main_frame = ttk.Frame(root, style='Modern.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create UI components
        self.create_pdf_selection()
        self.create_ai_toggle()
        self.create_question_input()
        self.create_answer_display()
        self.create_status_bar()
        
        # Load API key if exists
        self.load_api_key()
        
        # Bind events
        self.question_entry.bind('<Return>', lambda e: self.ask_question())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _run_event_loop(self):
        """Run async event loop in separate thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()
        self.loop.run_forever()
    
    def run_coroutine(self, coro):
        """Run a coroutine in the event loop and return a future"""
        if not self.loop:
            raise RuntimeError("Event loop not initialized")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)
    
    def on_closing(self):
        """Clean up resources when closing the application"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()
    
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
    
    def create_ai_toggle(self):
        """Create Gemini AI toggle switch and API key entry"""
        ai_frame = ttk.Frame(self.main_frame)
        ai_frame.pack(fill=tk.X, pady=(0, 10))
        
        # AI Toggle switch
        self.ai_toggle = ttk.Checkbutton(
            ai_frame,
            text="Use Gemini AI",
            variable=self.use_ai,
            command=self.toggle_ai
        )
        self.ai_toggle.pack(side=tk.LEFT)
        
        # API Key entry
        self.api_key_var = tk.StringVar()
        ttk.Label(ai_frame, text="API Key:").pack(side=tk.LEFT, padx=(20, 5))
        self.api_key_entry = ttk.Entry(ai_frame, textvariable=self.api_key_var, show="*")
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Save API Key button
        self.save_key_button = ttk.Button(
            ai_frame,
            text="Save Key",
            command=self.save_api_key,
            style='Modern.TButton'
        )
        self.save_key_button.pack(side=tk.RIGHT, padx=(5, 0))
    
    def toggle_ai(self):
        """Handle AI toggle switch changes"""
        if self.use_ai.get() and not self.gemini.is_initialized():
            api_key = self.api_key_var.get().strip()
            if not api_key:
                messagebox.showwarning(
                    "API Key Required",
                    "Please enter your Gemini API key to use AI features."
                )
                self.use_ai.set(False)
                return
            
            def init_gemini():
                if not self.gemini.initialize(api_key):
                    self.root.after(0, lambda: messagebox.showerror(
                        "Initialization Error",
                        "Failed to initialize Gemini AI. Please check your API key."
                    ))
                    self.root.after(0, lambda: self.use_ai.set(False))
            
            # Initialize Gemini in a separate thread
            threading.Thread(target=init_gemini, daemon=True).start()
    
    def save_api_key(self):
        """Save API key to configuration file"""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning(
                "Invalid Key",
                "Please enter a valid API key."
            )
            return
        
        try:
            config = {'api_key': api_key}
            with open('config.json', 'w') as f:
                json.dump(config, f)
            messagebox.showinfo(
                "Success",
                "API key saved successfully!"
            )
            # Initialize Gemini if toggle is on
            if self.use_ai.get():
                self.gemini.initialize(api_key)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to save API key: {str(e)}"
            )
    
    def load_api_key(self):
        """Load API key from configuration file"""
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config = json.load(f)
                    api_key = config.get('api_key')
                    if api_key:
                        self.api_key_var.set(api_key)
                        # Don't initialize yet, wait for toggle
        except Exception as e:
            print(f"Error loading API key: {str(e)}")
        
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
        
    async def get_ai_response(self, query, relevant_chunks):
        """Get response from Gemini AI"""
        if not self.gemini.is_initialized():
            return "Gemini AI is not initialized. Please provide an API key and enable AI."
        
        # Extract text from relevant chunks
        context_texts = [chunk for chunk, _ in relevant_chunks]
        return await self.gemini.get_response(query, context_texts)
    
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
                "Please wait while the current process completes!"
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
                    
                    # Display relevant chunks
                    self.display_results(relevant_info)
                    
                    # If AI is enabled, get AI response
                    if self.use_ai.get():
                        self.status_var.set("Getting AI response...")
                        
                        def handle_ai_response(future):
                            try:
                                ai_response = future.result()
                                self.root.after(0, lambda: self.update_ai_response(ai_response))
                            except Exception as e:
                                error_msg = f"\nError getting AI response: {str(e)}\n"
                                self.root.after(0, lambda: self.update_ai_response(error_msg))
                        
                        # Run AI response in background
                        future = self.run_coroutine(self.get_ai_response(query, relevant_info))
                        future.add_done_callback(handle_ai_response)
                
                search_time = time.time() - start_time
                self.status_var.set(
                    f"Found {len(relevant_info)} results in {search_time:.2f}s"
                )
                
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
    
    def update_ai_response(self, response):
        """Update UI with AI response"""
        self.answer_text.insert(tk.END, "\nAI Analysis:\n")
        self.answer_text.insert(tk.END, "─" * 80 + "\n")
        self.answer_text.insert(tk.END, f"{response}\n")
    
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