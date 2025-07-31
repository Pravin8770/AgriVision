import os
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run the AgriVision LLaMA Chatbot")
    parser.add_argument("--mode", choices=["cli", "web"], default="cli", 
                       help="Mode to run the chatbot in (cli or web)")
    parser.add_argument("--port", type=int, default=5000, help="Port for web server (web mode only)")
    parser.add_argument("--api-key", help="NVIDIA API key (overrides environment variable)")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["NVIDIA_API_KEY"] = args.api_key
    
    if args.mode == "cli":
        from chatbot.cli_interface import main as run_cli
        run_cli()
    else:  # web mode
        from chatbot.web_interface import run_server
        run_server(port=args.port, debug=True)

if __name__ == "__main__":
    main()
