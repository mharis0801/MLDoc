import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from ttkthemes import ThemedTk
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
        self.root.geometry("1200x800")  # Even larger default size
        
        # Set theme
        self.root.set_theme("arc")  # Modern, clean theme
        
        # Configure colors
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#4CAF50',
            'background': '#F5F5F5',
            'text': '#212121',
            'error': '#F44336',
            'warning': '#FFC107'
        }
        
        # Font configurations
        self.fonts = {
            'header': ('Helvetica', 14, 'bold'),
            'normal': ('Helvetica', 12),
            'large': ('Helvetica', 13),
            'small': ('Helvetica', 11)
        }
        
        # Initialize Gemini AI
        self.gemini = GeminiHelper()
        self.use_ai = tk.BooleanVar(value=False)
        self.api_key_var = tk.StringVar()
        
        # Initialize threading components
        self.loop = None
        self.loop_ready = threading.Event()
        
        # Create async loop in a separate thread
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        
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
        
        # Create main container with padding
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create layout frames
        self.create_header_frame()
        self.create_body_frame()
        self.create_footer_frame()
        
        # Load API key if exists
        self.load_api_key()
        
        # Bind events
        self.question_entry.bind('<Return>', lambda e: self.ask_question())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

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
        """Configure custom styles for widgets"""
        style = ttk.Style()
        
        # Configure frame styles
        style.configure('Header.TFrame', background=self.colors['background'])
        style.configure('Body.TFrame', background=self.colors['background'])
        style.configure('Footer.TFrame', background=self.colors['background'])
        
        # Configure button styles
        style.configure('Primary.TButton',
                       padding=12,
                       font=self.fonts['normal'])
        
        style.configure('Secondary.TButton',
                       padding=10,
                       font=self.fonts['normal'])
        
        # Configure label styles
        style.configure('Header.TLabel',
                       font=self.fonts['header'],
                       padding=8)
        
        style.configure('Info.TLabel',
                       font=self.fonts['normal'],
                       padding=6)
        
        # Configure entry styles
        style.configure('Custom.TEntry',
                       padding=8,
                       font=self.fonts['large'])
        
        # Configure progress bar
        style.configure('Custom.Horizontal.TProgressbar',
                       troughcolor=self.colors['background'],
                       background=self.colors['secondary'])
    
    def create_header_frame(self):
        """Create header section with PDF selection and AI toggle"""
        header = ttk.Frame(self.main_frame, style='Header.TFrame')
        header.grid(row=0, column=0, sticky='nsew', pady=(0, 10))
        header.grid_columnconfigure(1, weight=1)
        
        # PDF selection section
        ttk.Label(header, 
                 text="PDF Document:", 
                 style='Header.TLabel').grid(row=0, column=0, padx=5)
        
        self.pdf_label = ttk.Label(
            header,
            text="No PDF selected",
            style='Info.TLabel',
            wraplength=500
        )
        self.pdf_label.grid(row=0, column=1, sticky='w')
        
        self.select_button = ttk.Button(
            header,
            text="Select PDF",
            command=self.select_pdf,
            style='Primary.TButton'
        )
        self.select_button.grid(row=0, column=2, padx=5)
        
        # AI section
        ai_frame = ttk.Frame(header)
        ai_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=10)
        
        self.ai_toggle = ttk.Checkbutton(
            ai_frame,
            text="Use Gemini AI",
            variable=self.use_ai,
            command=self.toggle_ai,
            style='Custom.TCheckbutton'
        )
        self.ai_toggle.pack(side=tk.LEFT)
        
        ttk.Label(ai_frame, text="API Key:", style='Info.TLabel').pack(
            side=tk.LEFT, padx=(20, 5))
        
        self.api_key_entry = ttk.Entry(
            ai_frame,
            textvariable=self.api_key_var,
            show="*",
            style='Custom.TEntry'
        )
        self.api_key_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.save_key_button = ttk.Button(
            ai_frame,
            text="Save Key",
            command=self.save_api_key,
            style='Secondary.TButton'
        )
        self.save_key_button.pack(side=tk.LEFT, padx=(5, 0))
    
    def create_body_frame(self):
        """Create main content area with question input and answers"""
        body = ttk.Frame(self.main_frame, style='Body.TFrame')
        body.grid(row=1, column=0, sticky='nsew')
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)
        
        # Question input section
        question_frame = ttk.Frame(body)
        question_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        question_frame.grid_columnconfigure(0, weight=1)
        
        ttk.Label(question_frame,
                 text="Enter your question:",
                 style='Header.TLabel').grid(row=0, column=0, sticky='w')
        
        self.question_entry = ttk.Entry(
            question_frame,
            style='Custom.TEntry',
            font=self.fonts['large']
        )
        self.question_entry.grid(row=1, column=0, sticky='ew', pady=(5, 0))
        
        self.ask_button = ttk.Button(
            question_frame,
            text="Ask Question",
            command=self.ask_question,
            style='Primary.TButton'
        )
        self.ask_button.grid(row=1, column=1, padx=(10, 0))
        
        # Answer display section
        answer_frame = ttk.Frame(body)
        answer_frame.grid(row=1, column=0, sticky='nsew')
        answer_frame.grid_columnconfigure(0, weight=1)
        answer_frame.grid_rowconfigure(1, weight=1)
        
        ttk.Label(answer_frame,
                 text="Answers:",
                 style='Header.TLabel').grid(row=0, column=0, sticky='w')
        
        # Custom text widget with improved styling
        self.answer_text = scrolledtext.ScrolledText(
            answer_frame,
            wrap=tk.WORD,
            font=self.fonts['normal'],
            background='white',
            foreground=self.colors['text'],
            padx=15,
            pady=15
        )
        self.answer_text.grid(row=1, column=0, sticky='nsew', pady=(5, 0))
        
        # Configure tags for text styling
        self.answer_text.tag_configure(
            "header",
            font=self.fonts['header'],
            foreground=self.colors['primary']
        )
        self.answer_text.tag_configure(
            "section_header",
            font=('Helvetica', 16, 'bold'),
            foreground=self.colors['primary']
        )
        self.answer_text.tag_configure(
            "confidence_high",
            font=self.fonts['normal'],
            foreground='#2E7D32'  # Dark green
        )
        self.answer_text.tag_configure(
            "confidence_medium",
            font=self.fonts['normal'],
            foreground='#F57C00'  # Orange
        )
        self.answer_text.tag_configure(
            "confidence_low",
            font=self.fonts['normal'],
            foreground='#C62828'  # Dark red
        )
    
    def create_footer_frame(self):
        """Create footer with status and progress information"""
        footer = ttk.Frame(self.main_frame, style='Footer.TFrame')
        footer.grid(row=2, column=0, sticky='ew', pady=(10, 0))
        footer.grid_columnconfigure(0, weight=1)
        
        # Status display
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            footer,
            textvariable=self.status_var,
            style='Info.TLabel'
        )
        self.status_bar.pack(fill=tk.X)
        
        # Progress bars frame
        self.progress_frame = ttk.Frame(footer)
        self.progress_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Main progress bar
        self.progress = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            style='Custom.Horizontal.TProgressbar'
        )
        
        # Page progress section
        self.page_progress_frame = ttk.Frame(self.progress_frame)
        self.page_progress_frame.pack(fill=tk.X)
        
        self.page_count_var = tk.StringVar()
        self.page_count_label = ttk.Label(
            self.page_progress_frame,
            textvariable=self.page_count_var,
            style='Info.TLabel'
        )
        self.page_count_label.pack(fill=tk.X)
        
        self.page_progress = ttk.Progressbar(
            self.page_progress_frame,
            mode='determinate',
            style='Custom.Horizontal.TProgressbar'
        )
        self.page_progress.pack(fill=tk.X, pady=(2, 0))

    def display_results(self, results):
        """Display search results with enhanced formatting"""
        self.answer_text.delete(1.0, tk.END)
        
        # Add title
        self.answer_text.insert(tk.END, "Search Results\n", "section_header")
        self.answer_text.insert(tk.END, "─" * 80 + "\n\n")
        
        for i, (chunk, similarity) in enumerate(results, 1):
            # Format confidence score as percentage with color coding
            confidence = similarity * 100
            confidence_color = self._get_confidence_color(confidence)
            
            # Add result header
            self.answer_text.insert(tk.END, f"Result {i}\n", "header")
            self.answer_text.insert(
                tk.END,
                f"Confidence: {confidence:.1f}%\n",
                f"confidence_{confidence_color}"
            )
            self.answer_text.insert(tk.END, "─" * 40 + "\n")
            
            # Format and insert the content
            formatted_chunk = self._format_chunk(chunk)
            self.answer_text.insert(tk.END, f"{formatted_chunk}\n\n")
        
        # Configure additional text tags for styling
        self.answer_text.tag_configure(
            "section_header",
            font=("Helvetica", 16, "bold"),
            foreground=self.colors['primary']
        )
        self.answer_text.tag_configure(
            "confidence_high",
            foreground='#2E7D32'  # Dark green
        )
        self.answer_text.tag_configure(
            "confidence_medium",
            foreground='#F57C00'  # Orange
        )
        self.answer_text.tag_configure(
            "confidence_low",
            foreground='#C62828'  # Dark red
        )
    
    def _get_confidence_color(self, confidence):
        """Get color coding based on confidence score"""
        if confidence >= 70:
            return "high"
        elif confidence >= 40:
            return "medium"
        return "low"
    
    def _format_chunk(self, text):
        """Format text chunk for better readability"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Add proper paragraph breaks
        text = text.replace('. ', '.\n\n')
        
        return text.strip()
    
    def update_ai_response(self, response):
        """Update UI with enhanced AI response formatting"""
        # Add separator
        self.answer_text.insert(tk.END, "\n" + "═" * 80 + "\n\n")
        
        # Add AI section header
        self.answer_text.insert(tk.END, "AI Analysis\n", "ai_header")
        self.answer_text.insert(tk.END, "─" * 40 + "\n\n")
        
        # Format and insert AI response
        formatted_response = self._format_ai_response(response)
        self.answer_text.insert(tk.END, formatted_response + "\n")
        
        # Scroll to show the AI response
        self.answer_text.see(tk.END)
    
    def _format_ai_response(self, text):
        """Format AI response for better readability"""
        # Preserve bullet points and numbered lists
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            # Skip empty lines
            if not stripped_line:
                formatted_lines.append('')
                continue
                
            # Add extra spacing for lists
            if (stripped_line[0] in ['•', '-', '*'] or
                any(stripped_line.startswith(str(i) + '.') for i in range(10))):
                formatted_lines.append('  ' + line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def update_page_progress(self, current, total):
        """Update progress bar during PDF processing"""
        if not self.processing:
            return
        
        self.current_page = current
        self.total_pages = total
        
        # Calculate processing speed and ETA
        current_time = time.time()
        if hasattr(self, 'start_time'):
            elapsed_time = current_time - self.start_time
            pages_per_second = current / elapsed_time if elapsed_time > 0 else 0
            eta = (total - current) / pages_per_second if pages_per_second > 0 else 0
            
            # Format progress message
            progress_msg = (
                f"Processing page {current:,} of {total:,}\n"
                f"Speed: {pages_per_second:.1f} pages/sec • "
                f"ETA: {eta:.0f} seconds"
            )
            self.page_count_var.set(progress_msg)
        else:
            self.start_time = current_time
            self.page_count_var.set(f"Processing page {current:,} of {total:,}")
        
        # Update progress bar
        self.page_progress["value"] = (current / total) * 100
        self.root.update_idletasks()
    
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
        except Exception as e:
            print(f"Error loading API key: {str(e)}")
    
    def select_pdf(self):
        """Handle PDF file selection"""
        file_path = filedialog.askopenfilename(
            title="Select PDF Document",
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
        """Process the selected PDF file"""
        def process():
            try:
                # Update UI to show processing state
                self.status_var.set("Initializing PDF processing...")
                self.progress.pack(fill=tk.X, pady=(5, 0))
                self.progress.start()
                self.page_progress_frame.pack(fill=tk.X, pady=(5, 0))
                self.processing = True
                self.start_time = time.time()
                
                # Clear any existing warning dialogs
                for widget in self.root.winfo_children():
                    if isinstance(widget, tk.Toplevel):
                        widget.destroy()
                
                # Update UI state
                self.select_button.config(state='disabled')
                self.question_entry.config(state='disabled')
                self.ask_button.config(state='disabled')
                
                # Extract text with progress updates
                text = extract_text_from_pdf(self.pdf_path, self.update_page_progress)
                
                # Process text in chunks
                self.status_var.set("Analyzing document structure...")
                self.chunks = preprocess_text(text)
                
                # Calculate and display processing stats
                process_time = time.time() - self.start_time
                avg_speed = self.total_pages / process_time if process_time > 0 else 0
                
                stats_msg = (
                    f"Ready! Processed {self.total_pages:,} pages in {process_time:.1f}s "
                    f"(avg: {avg_speed:.1f} pages/sec)\n"
                    f"Found {len(self.chunks):,} text segments for analysis"
                )
                self.status_var.set(stats_msg)
                
                # Re-enable UI elements
                self._enable_question_input()
                self.select_button.config(state='normal')
                
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                messagebox.showerror("Error", str(e))
                self.select_button.config(state='normal')
                self.chunks = None  # Clear any partial processing
                self.pdf_path = None  # Reset PDF path on error
                self.pdf_label.config(text="No PDF selected")  # Reset label
                
            finally:
                self.processing = False
                self.progress.stop()
                self.progress.pack_forget()
                self.page_progress_frame.pack_forget()
        
        # Start processing in a separate thread
        threading.Thread(target=process, daemon=True).start()

    def ask_question(self):
        """Handle question submission and result display"""
        # Clear any existing warning dialogs first
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy()
                
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
            self.question_entry.focus_set()
            return
            
        # Check if this is the same question as last time
        if query == self.last_question and self.last_result:
            self.display_results(self.last_result)
            return
            
        def process_question():
            try:
                # Update UI state
                self.status_var.set("Searching for relevant information...")
                self.progress.pack(fill=tk.X, pady=(5, 0))
                self.progress.start()
                self.processing = True
                self.question_entry.config(state='disabled')
                self.ask_button.config(state='disabled')
                start_time = time.time()
                
                # Search for relevant information
                relevant_info = find_relevant_info(query, self.chunks)
                
                if not relevant_info:
                    self.answer_text.delete(1.0, tk.END)
                    self.answer_text.insert(
                        tk.END,
                        "No relevant information found. Please try rephrasing your question.",
                        "header"
                    )
                    # Make sure to re-enable input when no results found
                    self.root.after(0, self._enable_question_input)
                else:
                    # Cache the results
                    self.last_question = query
                    self.last_result = relevant_info
                    
                    # Display relevant chunks
                    self.display_results(relevant_info)
                    
                    # If AI is enabled, get AI response
                    if self.use_ai.get():
                        self.status_var.set("Getting AI analysis...")
                        
                        def handle_ai_response(future):
                            try:
                                ai_response = future.result()
                                self.root.after(0, lambda: self.update_ai_response(ai_response))
                            except Exception as e:
                                error_msg = f"\nError getting AI response: {str(e)}\n"
                                self.root.after(0, lambda: self.update_ai_response(error_msg))
                            finally:
                                self.root.after(0, self._enable_question_input)
                        
                        # Run AI response in background
                        future = self.run_coroutine(self.get_ai_response(query, relevant_info))
                        future.add_done_callback(handle_ai_response)
                    else:
                        # Make sure to re-enable input when not using AI
                        self.root.after(0, self._enable_question_input)
                
                # Update status with timing information
                search_time = time.time() - start_time
                result_count = len(relevant_info) if relevant_info else 0
                self.status_var.set(
                    f"Found {result_count} results in {search_time:.2f}s"
                )
                
            except Exception as e:
                self.answer_text.delete(1.0, tk.END)
                self.answer_text.insert(
                    tk.END,
                    f"An error occurred: {str(e)}",
                    "error"
                )
                self.status_var.set("Error occurred")
                # Make sure to re-enable input on error
                self.root.after(0, self._enable_question_input)
            finally:
                self.processing = False
                self.progress.stop()
                self.progress.pack_forget()
        
        # Run processing in background thread
        thread = threading.Thread(target=process_question, daemon=True)
        thread.start()
    
    async def get_ai_response(self, query, relevant_info):
        """Get AI analysis of the question and relevant chunks"""
        if not self.use_ai.get() or not self.gemini.is_initialized():
            return "AI analysis not available. Please enable AI and provide an API key."
            
        try:
            # Extract text content from relevant info
            chunks = [chunk for chunk, _ in relevant_info]
            
            # Get AI response
            response = await self.gemini.get_response(query, chunks)
            return response
            
        except Exception as e:
            return f"Error getting AI response: {str(e)}"
    
    def _enable_question_input(self):
        """Re-enable question input controls"""
        self.question_entry.config(state='normal')
        self.ask_button.config(state='normal')
        self.question_entry.delete(0, tk.END)
        self.question_entry.focus_set()

def main():
    root = ThemedTk(theme="arc")  # Set modern theme
    root.title("PDF Question Answering System")
    
    # Configure color scheme
    style = ttk.Style()
    style.configure(".", font=('Helvetica', 10))
    style.configure("TButton", padding=6)
    style.configure("TEntry", padding=6)
    
    app = PDFQuestionAnswerUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()