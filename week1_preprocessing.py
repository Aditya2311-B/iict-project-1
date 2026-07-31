import pandas as pd
import re
import os

# Hardcoded list of standard English stopwords to avoid external package dependencies (like nltk)
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
    """
    Cleans text by lowercasing and removing punctuation/numbers,
    tokenizes by splitting on spaces, and removes stopwords manually.
    """
    if not isinstance(text, str):
        return []
    
    # 1. Lowercase the text
    text_lower = text.lower()
    
    # 2. Remove punctuation and special characters (keep only alphabetic characters and spaces)
    text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text_lower)
    
    # 3. Manual Tokenization (splitting by whitespace)
    tokens = text_clean.split()
    
    # 4. Remove Stopwords
    filtered_tokens = [token for token in tokens if token not in STOPWORDS]
    
    return filtered_tokens

def main():
    dataset_path = "train.csv"
    output_path = "clean_data.csv"
    
    print("=== WEEK 1: Data Loading & Preprocessing ===")
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found! Please run download_dataset.py first.")
        return
        
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape} (Rows, Columns)")
    print("Columns available:", list(df.columns))
    
    # Print label distribution
    print("\nInitial Label distribution:")
    print(df['label'].value_counts())
    
    # Handle missing values
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    # Drop rows with null text or label
    df = df.dropna(subset=['text', 'label'])
    print(f"Shape after removing null rows: {df.shape}")
    
    # Show text cleaning on a sample text
    sample_text = df['text'].iloc[0]
    print("\n--- Preprocessing Demo on Sample Article ---")
    print(f"Original Text (First 150 chars):\n{sample_text[:150]}...")
    
    tokens = clean_and_tokenize(sample_text)
    print(f"\nCleaned and Tokenized list (First 15 tokens):\n{tokens[:15]}")
    
    # Apply preprocessing to all texts in the dataset and join them into clean strings for vectorizer
    print("\nPreprocessing the entire dataset... (this may take a few seconds)")
    df['clean_tokens'] = df['text'].apply(clean_and_tokenize)
    df['clean_text'] = df['clean_tokens'].apply(lambda x: " ".join(x))
    
    # Keep only columns we need for machine learning
    clean_df = df[['title', 'clean_text', 'label']]
    
    # Save the cleaned dataset
    clean_df.to_csv(output_path, index=False)
    print(f"\nPreprocessed dataset successfully saved to: {output_path}")
    print(f"Saved dataset columns: {list(clean_df.columns)}")

if __name__ == "__main__":
    main()
