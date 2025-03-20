from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
import joblib
import os
from gtts import gTTS
from google.api_core import exceptions

app = Flask(__name__)
CORS(app)

# Configure Gemini API
genai.configure(api_key="AIzaSyDqpolr-EHZMGLhpIH_snp_A_-Cx3I9lxs")  # Replace with your actual key

# Load NLP model and vectorizer
nlp_model = joblib.load("models/sentiment_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Initialize Gemini Model
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Store the full story globally (for audio later)
full_story = ""

@app.route('/start_interactive_story', methods=['POST'])
def start_interactive_story():
    """Starts a story and ends with a decision point."""
    global full_story
    data = request.json
    topic = data.get("topic", "A mysterious adventure")

    prompt = (
        f"Write the beginning of a detailed and engaging story about '{topic}'. "
        f"Stop at an interesting decision point and end with a specific question about what happens next. "
        f"Provide three clear, distinct options labeled as 'Option 1', 'Option 2', and 'Option 3'. "
        f"Format the question and options like this:\n"
        f"**Question:** [Your question here]\n"
        f"**Option 1:** [First option]\n"
        f"**Option 2:** [Second option]\n"
        f"**Option 3:** [Third option]"
    )
    try:
        response = model.generate_content(prompt)
        full_story = response.text
        story_text, question, options = parse_story_response(response.text)
        return jsonify({
            "story_part": story_text,
            "question": question,
            "options": options,
            "stage": "initial",
            "status": "Generating..."
        })
    except exceptions.ResourceExhausted:
        return jsonify({
            "error": "API quota exceeded. Retrying in a moment..."
        }), 429

@app.route('/continue_story', methods=['POST'])
def continue_story():
    """Continues the story based on user choice or exits."""
    global full_story
    data = request.json
    current_story = data.get("current_story")
    user_choice = data.get("choice")
    choice_text = data.get("choice_text")

    try:
        if user_choice and choice_text:
            new_prompt = (
                f"{current_story}\n\nThe protagonist chooses '{choice_text}'. "
                f"Continue the story from this point with a new development. "
                f"Stop at an interesting decision point and end with a specific question about what happens next. "
                f"Provide three clear, distinct options labeled as 'Option 1', 'Option 2', and 'Option 3'. "
                f"Format like this:\n"
                f"**Question:** [Your question here]\n"
                f"**Option 1:** [First option]\n"
                f"**Option 2:** [Second option]\n"
                f"**Option 3:** [Third option]"
            )
            response = model.generate_content(new_prompt)
            full_story += "\n\n" + response.text
            story_text, question, options = parse_story_response(response.text)
            return jsonify({
                "story_part": story_text,
                "question": question,
                "options": options,
                "stage": "middle"
            })
        elif data.get("continue_or_exit") == "exit":
            new_prompt = f"{current_story}\n\nConclude the story with a satisfying ending."
            response = model.generate_content(new_prompt)
            full_story += "\n\n" + response.text
            return jsonify({
                "story_part": response.text,
                "stage": "end"
            })
        else:
            new_prompt = (
                f"{current_story}\n\nContinue the story with a new development. "
                f"Stop at an interesting decision point and end with a specific question about what happens next. "
                f"Provide three clear, distinct options labeled as 'Option 1', 'Option 2', and 'Option 3'. "
                f"Format like this:\n"
                f"**Question:** [Your question here]\n"
                f"**Option 1:** [First option]\n"
                f"**Option 2:** [Second option]\n"
                f"**Option 3:** [Third option]"
            )
            response = model.generate_content(new_prompt)
            full_story += "\n\n" + response.text
            story_text, question, options = parse_story_response(response.text)
            return jsonify({
                "story_part": story_text,
                "question": question,
                "options": options,
                "stage": "middle"
            })
    except exceptions.ResourceExhausted:
        return jsonify({
            "error": "API quota exceeded. Retrying in a moment..."
        }), 429

def parse_story_response(text):
    """Helper function to extract story, question, and options."""
    story_text = text
    question = "What will happen next?"
    options = {"1": "Option 1", "2": "Option 2", "3": "Option 3"}

    try:
        lines = text.split('\n')
        story_lines = []
        for i, line in enumerate(lines):
            if line.startswith("**Question:**"):
                story_lines = lines[:i]
                question = line.replace("**Question:**", "").strip()
            elif line.startswith("**Option 1:**"):
                options["1"] = line.replace("**Option 1:**", "").strip()
            elif line.startswith("**Option 2:**"):
                options["2"] = line.replace("**Option 2:**", "").strip()
            elif line.startswith("**Option 3:**"):
                options["3"] = line.replace("**Option 3:**", "").strip()
        story_text = "\n".join(story_lines).strip()
    except Exception as e:
        print(f"Error parsing story response: {e}")

    return story_text, question, options

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    global full_story
    data = request.json
    text = data.get("story", full_story)

    if not text:
        return jsonify({"error": "No story to convert"}), 400

    try:
        tts = gTTS(text=text, lang='en')
        audio_path = "static/story_audio.mp3"
        tts.save(audio_path)
        return jsonify({"audio_url": audio_path})
    except Exception as e:
        print(f"Text-to-Speech Error: {str(e)}")
        return jsonify({"error": "Failed to generate audio. Please try again later."}), 500

@app.route('/happy_ending', methods=['POST'])
def happy_ending():
    global full_story
    data = request.json
    original_story = data.get("story", full_story)

    prompt_happy = f"Take the following story and rewrite it entirely to have a happy ending:\n\n{original_story}"
    try:
        response_happy = model.generate_content(prompt_happy)
        full_story = response_happy.text
        return jsonify({"happy_story": response_happy.text})
    except exceptions.ResourceExhausted:
        return jsonify({
            "error": "API quota exceeded. Retrying in a moment..."
        }), 429

@app.route('/analyze_review', methods=['POST'])
def analyze_review():
    data = request.json
    review_text = data.get("review", "")
    if not review_text:
        return jsonify({"error": "No review provided"}), 400
    review_vector = tfidf_vectorizer.transform([review_text])
    prediction = nlp_model.predict(review_vector)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"sentiment": sentiment})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2000, debug=True)