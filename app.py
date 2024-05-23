from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import chatbot

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.keras')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = chatbot.chatbot_response(userText)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)

