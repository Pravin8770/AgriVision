from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
from .llama_chatbot import LlamaAgriChatbot

app = Flask(__name__)
chatbot = LlamaAgriChatbot(api_key=os.environ.get("NVIDIA_API_KEY"))

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for non-streaming chat responses."""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    formatted_message = chatbot.agricultural_prompt(user_message)
    response = chatbot.get_response(formatted_message, stream=False)
    
    return jsonify({"response": response})

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """API endpoint for streaming chat responses."""
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    formatted_message = chatbot.agricultural_prompt(user_message)
    
    def generate():
        for chunk in chatbot.get_response(formatted_message, stream=True):
            yield f"data: {chunk}\n\n"
    
    return Response(stream_with_context(generate()), content_type='text/event-stream')

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_server(debug=True)
