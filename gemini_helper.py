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
            response = await self.model.generate_content_async(prompt)
            
            if not response or not response.text:
                return "Sorry, I couldn't generate a response. Please try rephrasing your question."
                
            return response.text.strip()
            
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