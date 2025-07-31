import os
from typing import List, Dict, Optional, Generator
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaAgriChatbot:
    """Chatbot for agricultural assistance using LLaMA model via NVIDIA API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the chatbot with API credentials.
        
        Args:
            api_key: The API key for NVIDIA API access
        """
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. Please set NVIDIA_API_KEY environment variable or provide key directly.")
            
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        )
        
        self.conversation_history = []
        self.model = "meta/llama-3.2-3b-instruct"
        
    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        
    def get_response(self, user_message: str) -> str:
        """
        Get a response from the LLaMA model (non-streaming).
        
        Args:
            user_message: The user's input message
            
        Returns:
            The model's response text
        """
        if not self.api_key:
            return "API key is not configured. Cannot connect to the model."
        
        # Add user message to history
        self.add_message("user", user_message)
        
        try:
            # Create the completion request
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=False
            )
            
            # Handle non-streaming response
            response = completion.choices[0].message.content
            self.add_message("assistant", response)
            return response
                
        except Exception as e:
            logger.error(f"Error communicating with LLaMA model: {e}")
            return f"Error communicating with LLaMA model: {str(e)}"
    
    def get_streaming_response(self, user_message: str) -> Generator[str, None, None]:
        """
        Get a streaming response from the LLaMA model.
        
        Args:
            user_message: The user's input message
            
        Yields:
            Chunks of the model's response text
        """
        if not self.api_key:
            yield "API key is not configured. Cannot connect to the model."
            return
        
        # Add user message to history
        self.add_message("user", user_message)
        
        try:
            # Create the completion request
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=True
            )
            
            # Handle streaming response
            collected_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    collected_response += chunk_content
                    yield chunk_content
            
            # Add assistant response to history
            self.add_message("assistant", collected_response)
                
        except Exception as e:
            error_msg = f"Error communicating with LLaMA model: {str(e)}"
            logger.error(error_msg)
            yield error_msg
    
    def agricultural_prompt(self, query: str) -> str:
        """
        Format an agricultural domain-specific prompt for better responses.
        
        Args:
            query: The user's agricultural query
            
        Returns:
            A formatted prompt for the LLaMA model
        """
        return f"""You are an agricultural assistant helping farmers with their questions.
Please analyze the following question about agriculture and provide a helpful response:

{query}

Include relevant information about crop management, pest control, or agricultural technology if applicable."""
