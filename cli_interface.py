import os
import sys
import argparse
from .llama_chatbot import LlamaAgriChatbot

def print_streaming_response(generator):
    """Print streaming response from the generator."""
    for text_chunk in generator:
        print(text_chunk, end="", flush=True)
    print()  # Add a newline at the end

def main():
    """Run the command line chatbot interface."""
    parser = argparse.ArgumentParser(description="AgriVision LLaMA Chatbot CLI")
    parser.add_argument("--api-key", help="NVIDIA API key")
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("NVIDIA_API_KEY")
    
    # Initialize chatbot
    chatbot = LlamaAgriChatbot(api_key=api_key)
    
    print("AgriVision LLaMA Chatbot")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "clear":
            chatbot.clear_history()
            print("Conversation history cleared.")
            continue
        
        print("\nAI: ", end="", flush=True)
        
        # Use agricultural prompt formatting
        formatted_input = chatbot.agricultural_prompt(user_input)
        
        # Get streaming response
        response_generator = chatbot.get_response(formatted_input)
        print_streaming_response(response_generator)

if __name__ == "__main__":
    main()
