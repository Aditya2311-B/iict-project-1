import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import re

# Stopwords for custom inputs (from week1_preprocessing)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
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

def run_evaluation():
    print("=== WEEK 4: Model Evaluation & Visualizations ===")
    
    # Check for split data
    if not os.path.exists("split_data.pkl"):
        print("Error: split_data.pkl not found! Please run week3_model_training.py first.")
        return
        
    # Load testing split
    with open("split_data.pkl", "rb") as f:
        data_splits = pickle.load(f)
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
    model_names = ["KNN", "LogReg", "RandomForest", "NeuralNet"]
    models = {}
    
    # Load all models
    for name in model_names:
        filename = f"{name.lower()}_model.pkl"
        if not os.path.exists(filename):
            print(f"Error: {filename} not found!")
            return
        with open(filename, "rb") as f:
            models[name] = pickle.load(f)
            
    # Dictionary to store performance metrics
    results = {}
    
    # Create subplots for confusion matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models.items()):
        print(f"\nEvaluating Model: {name}")
        preds = model.predict(X_test)
        
        # Calculate overall metrics
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
        
        results[name] = {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
        
        # Print detailed classification report
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))
        
        # Generate Confusion Matrix
        cm = confusion_matrix(y_test, preds, labels=['REAL', 'FAKE'])
        
        # Plot Confusion Matrix in the grid
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    print("\nSaved plot: confusion_matrices.png")
    
    # Plot Accuracy Comparison Bar Chart
    results_df = pd.DataFrame(results).T
    print("\nSummary of Results:")
    print(results_df)
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=results_df.index, y=results_df['Accuracy'], palette='Set1')
    plt.title('Fake News Classifier Model Comparison (Accuracy)')
    plt.xlabel('Model')
    plt.ylabel('Accuracy Score')
    plt.ylim(0.5, 1.0) # zoom in on range 0.5 to 1.0
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.4f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 8), 
                    textcoords='offset points', 
                    fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison.png')
    plt.close()
    print("Saved plot: model_accuracy_comparison.png")
    
    # Write summary CSV
    results_df.to_csv("model_evaluation_metrics.csv")
    print("Saved metrics summary to: model_evaluation_metrics.csv")

def predict_custom_news():
    vectorizer_path = "tfidf_vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        print("Vectorizer not found! Cannot run custom prediction.")
        return
        
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    model_names = ["KNN", "LogReg", "RandomForest", "NeuralNet"]
    models = {}
    
    for name in model_names:
        filename = f"{name.lower()}_model.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                models[name] = pickle.load(f)
                
    if not models:
        print("No trained models found! Train models first.")
        return
        
    print("\n" + "="*50)
    print("      AI-POWERED FAKE NEWS PREDICTION TOOL")
    print("="*50)
    print("Type a news headline or short article text below to classify it.")
    print("To exit the prediction tool, type 'exit'.")
    print("-"*50)
    
    # Simple hardcoded samples to let user choose or type their own
    samples = [
        "President signs historic executive order to lower prescription drug costs nationwide.",
        "BREAKING: Secret NASA document proves Earth is flat and the moon is a hologram project by government.",
        "Local high school science club wins international competition with revolutionary water purification system."
    ]
    
    print("Sample Headlines to try:")
    for idx, s in enumerate(samples):
        print(f"[{idx + 1}] {s}")
    print("-"*50)
    
    while True:
        user_input = input("\nEnter news text or number [1-3] (or 'exit'): ").strip()
        if user_input.lower() == 'exit':
            print("Exiting prediction tool.")
            break
            
        if not user_input:
            continue
            
        # If user chooses a sample
        if user_input in ['1', '2', '3']:
            chosen_text = samples[int(user_input) - 1]
            print(f"\nTesting Sample {user_input}: \"{chosen_text}\"")
            text_to_test = chosen_text
        else:
            text_to_test = user_input
            
        # Clean and preprocess
        clean_text = clean_and_tokenize(text_to_test)
        
        # Vectorize
        x_vec = vectorizer.transform([clean_text])
        
        # Predict using all models
        print("\nPredictions:")
        print(f"{'Model':<15} | {'Prediction':<10} | {'Probability (Real / Fake)':<30}")
        print("-" * 65)
        for name, model in models.items():
            pred = model.predict(x_vec)[0]
            
            # Try to get prediction probabilities
            try:
                probs = model.predict_proba(x_vec)[0]
                # Class labels mapping in scikit-learn fits alphabetically: ['FAKE', 'REAL']
                fake_prob, real_prob = probs[0], probs[1]
                prob_str = f"Real: {real_prob:.2%}, Fake: {fake_prob:.2%}"
            except AttributeError:
                prob_str = "N/A"
                
            print(f"{name:<15} | {pred:<10} | {prob_str}")
        print("="*50)

if __name__ == "__main__":
    run_evaluation()
    # If this is run interactively or as part of a demonstration, let user input values
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        predict_custom_news()
