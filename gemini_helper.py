import google.generativeai as genai
from typing import List, Optional
import asyncio

class GeminiHelper:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.model = None
        self.initialized = False
        
    def initialize(self, api_key: str) -> bool:
        """Initialize Gemini with API key"""
        try:
            self.api_key = api_key
            genai.configure(api_key=self.api_key)
            
            # Set up model configuration
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 1024,
            }
            
            # Get the text model
            self.model = genai.GenerativeModel(
                'gemini-2.0-flash-lite',
                generation_config=generation_config
            )
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing Gemini AI: {str(e)}")
            self.initialized = False
            return False
    
    async def get_response(self, question: str, context_chunks: List[str]) -> str:
        """Get AI response for a question with context"""
        if not self.initialized or not self.model:
            return "Gemini AI is not initialized. Please provide an API key."
            
        try:
            # Prepare the prompt with context and format instructions
            context_text = "\n\n".join(
                f"Context {i+1}:\n{chunk}" 
                for i, chunk in enumerate(context_chunks)
            )
            
            prompt = f"""Based on the following context from a document, please answer this question: "{question}"

{context_text}

Instructions:
1. Provide a clear and concise answer based ONLY on the information provided in the context
2. If the context doesn't contain enough information to fully answer the question, say so
3. Use bullet points or numbered lists when appropriate
4. Include specific details and examples from the context to support your answer

Your answer:"""

            # Generate response
            try:
                response = self.model.generate_content(prompt)
                if response and hasattr(response, 'text'):
                    return response.text.strip()
                else:
                    return "Sorry, I couldn't generate a valid response. Please try rephrasing your question."
            except Exception as api_error:
                if "text too long" in str(api_error).lower():
                    # Try with shortened context if text is too long
                    shortened_chunks = [chunk[:1000] for chunk in context_chunks]
                    shortened_context = "\n\n".join(
                        f"Context {i+1}:\n{chunk}" 
                        for i, chunk in enumerate(shortened_chunks)
                    )
                    shortened_prompt = f"""Based on the following context from a document, please answer this question: "{question}"

{shortened_context}

Instructions:
1. Provide a clear and concise answer based ONLY on the information provided
2. If the question asks to EXPLAIN then provide a detailed explanation
3. If you need more context, please indicate this in your response
4. Use bullet points or numbered lists when appropriate

Your answer:"""
                    response = self.model.generate_content(shortened_prompt)
                    if response and hasattr(response, 'text'):
                        return response.text.strip() + "\n\nNote: Response was generated with truncated context due to length limitations."
                    
                raise  # Re-raise if shortening didn't help
                
        except Exception as e:
            error_msg = str(e)
            if "quota exceeded" in error_msg.lower():
                return "API quota exceeded. Please try again later or check your API limits."
            elif "invalid api key" in error_msg.lower():
                return "Invalid API key. Please check your API key and try again."
            else:
                return f"Error getting AI response: {error_msg}"
    
    def is_initialized(self) -> bool:
        """Check if Gemini AI is initialized"""
        return self.initialized