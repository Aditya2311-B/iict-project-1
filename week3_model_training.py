import pandas as pd
import pickle
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def main():
    print("=== WEEK 3: Model Building & Training ===")
    
    clean_data_path = "clean_data.csv"
    vectorizer_path = "tfidf_vectorizer.pkl"
    
    if not os.path.exists(clean_data_path) or not os.path.exists(vectorizer_path):
        print("Error: Required files (clean_data.csv or tfidf_vectorizer.pkl) are missing!")
        print("Please run week1_preprocessing.py and week2_feature_engineering.py first.")
        return
        
    # Load dataset and vectorizer
    print("Loading preprocessed dataset...")
    df = pd.read_csv(clean_data_path)
    df = df.dropna(subset=['clean_text'])
    
    print("Loading fitted TfidfVectorizer...")
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    # Transform text to TF-IDF features
    print("Vectorizing text...")
    X = vectorizer.transform(df['clean_text'])
    y = df['label']
    
    # Split into train and test sets (80% train, 20% test)
    print("Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Save the split testing data to disk for evaluation in Week 4
    test_data = {'X_test': X_test, 'y_test': y_test, 'X_train': X_train, 'y_train': y_train}
    with open("split_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    print("Saved train/test split data to split_data.pkl")
    
    # Initialize the models as specified in the PDF skeleton
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
    
    # Train each model and measure training time
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        
        # Fit model
        model.fit(X_train, y_train)
        
        duration = time.time() - start_time
        print(f"Successfully trained {name} in {duration:.2f} seconds.")
        
        # Save model to disk
        model_filename = f"{name.lower()}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved trained {name} model to: {model_filename}")
        
    print("\nAll models have been built and trained successfully!")

if __name__ == "__main__":
    main()
