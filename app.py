from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from file2 import ChatBot

app = Flask(__name__)
CORS(app)  # Enable CORS
bot = ChatBot('intents.json', 'words.pkl', 'classes.pkl', 'chatbot_model.keras')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    intents = bot.predict_class(user_message)
    response = bot.get_response(intents)
    return jsonify({'response': response})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
