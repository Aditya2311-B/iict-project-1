from flask import Flask, request, jsonify, render_template, send_file
import pickle
import os
import re

app = Flask(__name__)

# Stopwords for text preprocessing (same as week1_preprocessing)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", 
    "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def clean_and_tokenize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    return " ".join(filtered)

# Global variables for models and vectorizer
vectorizer = None
models = {}

def load_resources():
    global vectorizer, models
    vectorizer_path = "tfidf_vectorizer.pkl"
    model_names = ["KNN", "LogReg", "RandomForest", "NeuralNet"]
    
    # Load TF-IDF Vectorizer
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        print("Warning: tfidf_vectorizer.pkl not found.")
        
    # Load all models
    for name in model_names:
        filename = f"{name.lower()}_model.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                models[name] = pickle.load(f)
        else:
            print(f"Warning: {filename} not found.")

@app.route('/')
def home():
    # Render the index.html template
    return render_template('index.html')

@app.route('/download_report')
def download_report():
    report_path = "IEEE_Report.md"
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name="IEEE_Fake_News_Report.md")
    else:
        return "Report file not found.", 404

@app.route('/predict', methods=['POST'])
def predict():
    global vectorizer, models
    
    # Check if models are loaded
    if not vectorizer or not models:
        # Try to load if not initialized
        load_resources()
        if not vectorizer or not models:
            return jsonify({"error": "Models or vectorizer not loaded on server. Please train models first."}), 500
            
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided for classification."}), 400
        
    raw_text = data['text'].strip()
    if not raw_text:
        return jsonify({"error": "Empty text provided."}), 400
        
    # Preprocess
    clean_text = clean_and_tokenize(raw_text)
    
    # Vectorize
    x_vec = vectorizer.transform([clean_text])
    
    predictions = {}
    
    # Predict with all models
    for name, model in models.items():
        pred_label = model.predict(x_vec)[0]
        
        # Calculate probabilities
        try:
            probs = model.predict_proba(x_vec)[0]
            # Classes are ordered alphabetically: ['FAKE', 'REAL']
            prob_fake = float(probs[0])
            prob_real = float(probs[1])
            confidence = prob_real if pred_label == 'REAL' else prob_fake
        except Exception:
            prob_fake = 0.5
            prob_real = 0.5
            confidence = 0.5
            
        predictions[name] = {
            "prediction": pred_label,
            "confidence": round(confidence * 100, 2),
            "prob_real": round(prob_real * 100, 2),
            "prob_fake": round(prob_fake * 100, 2)
        }
        
    # Calculate consensus
    fake_votes = sum(1 for p in predictions.values() if p['prediction'] == 'FAKE')
    real_votes = sum(1 for p in predictions.values() if p['prediction'] == 'REAL')
    
    if fake_votes > real_votes:
        consensus_verdict = 'FAKE'
        consensus_confidence = sum(p['prob_fake'] for p in predictions.values()) / 4
    elif real_votes > fake_votes:
        consensus_verdict = 'REAL'
        consensus_confidence = sum(p['prob_real'] for p in predictions.values()) / 4
    else:
        # Tie break using Neural Network
        consensus_verdict = predictions['NeuralNet']['prediction']
        consensus_confidence = predictions['NeuralNet']['confidence']
        
    return jsonify({
        "predictions": predictions,
        "consensus": {
            "verdict": consensus_verdict,
            "confidence": round(consensus_confidence, 2)
        }
    })

if __name__ == '__main__':
    # Initialize resources on startup
    load_resources()
    # Run server on port 5001 to avoid conflicts
    print("Starting Flask Web Application on http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=True)
