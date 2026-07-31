import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from collections import Counter
import math

def manual_feature_engineering_demo():
    print("\n--- 1. Manual Feature Engineering Demonstration ---")
    
    # Let's define 3 simple sentences to demonstrate concepts
    docs = [
        "the sky is blue",
        "the sun is bright",
        "the blue sky has sun"
    ]
    print(f"Sample Documents:\nDoc 1: \"{docs[0]}\"\nDoc 2: \"{docs[1]}\"\nDoc 3: \"{docs[2]}\"")
    
    # --- A. Manual Bag-of-Words ---
    # Tokenize and build vocabulary
    vocab = sorted(list(set([word for doc in docs for word in doc.split()])))
    print(f"\nVocabulary ({len(vocab)} unique words):\n{vocab}")
    
    # Create manual Bag-of-Words representation
    bow_matrix = []
    for doc in docs:
        words = doc.split()
        vector = [words.count(w) for w in vocab]
        bow_matrix.append(vector)
    
    print("\nBag-of-Words Count Vectors:")
    for i, vec in enumerate(bow_matrix):
        print(f"Doc {i+1} Vector: {vec}")
        
    # --- B. Manual TF-IDF ---
    # Term Frequency (TF) = count of word / total words in doc
    # Document Frequency (DF) = number of docs containing the word
    # Inverse Document Frequency (IDF) = log(total docs / (1 + DF)) + 1  (standard smoothing)
    N = len(docs)
    df_dict = {word: 0 for word in vocab}
    for doc in docs:
        words = set(doc.split())
        for word in words:
            df_dict[word] += 1
            
    idf_dict = {}
    for word, df in df_dict.items():
        # Standard formula with smoothing: log(N / df)
        idf_dict[word] = round(math.log(N / df) + 1.0, 4)
        
    print("\nDocument Frequencies (DF) & Inverse Document Frequencies (IDF):")
    for word in vocab:
        print(f"  Word: '{word}' -> DF: {df_dict[word]}, IDF: {idf_dict[word]}")
        
    # Compute TF-IDF matrix
    tfidf_matrix = []
    for doc in docs:
        words = doc.split()
        vector = []
        for word in vocab:
            tf = words.count(word) / len(words)
            tfidf = round(tf * idf_dict[word], 4)
            vector.append(tfidf)
        tfidf_matrix.append(vector)
        
    print("\nManual TF-IDF Vectors:")
    for i, vec in enumerate(tfidf_matrix):
        print(f"Doc {i+1} TF-IDF: {vec}")
        
    # --- C. Manual Word Embeddings ---
    # Give a tiny dictionary representing pre-trained 3D word vectors
    word_embeddings = {
        "the":    [0.01, 0.02, 0.05],
        "sky":    [0.85, 0.12, -0.4],
        "is":     [-0.1, 0.05, 0.01],
        "blue":   [0.91, -0.3, 0.15],
        "sun":    [0.78, 0.65, 0.22],
        "bright": [0.62, 0.58, 0.10],
        "has":    [0.03, -0.01, 0.02]
    }
    
    print("\nManual Word Embeddings (3-Dimensional Dense Vectors) sample:")
    for w in ["sky", "sun", "blue"]:
        print(f"  Vector('{w}') = {word_embeddings[w]}")
        
    # Represent sentences as the average of their word vectors
    print("\nDocument Embedding Representation (Average word vectors):")
    for i, doc in enumerate(docs):
        vectors = [word_embeddings[w] for w in doc.split() if w in word_embeddings]
        doc_vector = np.mean(vectors, axis=0)
        print(f"Doc {i+1} Embedding: {np.round(doc_vector, 4)}")
        
    print("----------------------------------------------------\n")

def main():
    print("=== WEEK 2: Feature Engineering & EDA ===")
    
    # Run manual math demo
    manual_feature_engineering_demo()
    
    clean_data_path = "clean_data.csv"
    if not os.path.exists(clean_data_path):
        print(f"Error: {clean_data_path} not found! Please run week1_preprocessing.py first.")
        return
        
    # Load clean data
    print("Loading preprocessed dataset...")
    df = pd.read_csv(clean_data_path)
    df = df.dropna(subset=['clean_text'])
    
    # 2. Exploratory Data Analysis (EDA)
    print("\nPerforming Exploratory Data Analysis (EDA)...")
    
    # A. Label distribution
    label_counts = df['label'].value_counts()
    print("Label Counts:")
    print(label_counts)
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title('Distribution of Real vs Fake Articles')
    plt.xlabel('Label')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.close()
    print("Saved plot: label_distribution.png")
    
    # B. Article Length Analysis
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    
    print("\nArticle length statistics by Label:")
    print(df.groupby('label')['word_count'].describe())
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='word_count', hue='label', kde=True, bins=50, palette='Set2', multiple='stack')
    plt.title('Article Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.xlim(0, 2000)  # limit x axis to focus on main distributions
    plt.tight_layout()
    plt.savefig('article_length_distribution.png')
    plt.close()
    print("Saved plot: article_length_distribution.png")
    
    # C. Most Common Words in Real vs Fake
    real_words = " ".join(df[df['label'] == 'REAL']['clean_text']).split()
    fake_words = " ".join(df[df['label'] == 'FAKE']['clean_text']).split()
    
    real_freq = Counter(real_words).most_common(10)
    fake_freq = Counter(fake_words).most_common(10)
    
    print("\nTop 5 words in REAL articles:", real_freq[:5])
    print("Top 5 words in FAKE articles:", fake_freq[:5])
    
    # Plot most common words
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Real words plot
    words_real, counts_real = zip(*real_freq)
    sns.barplot(x=list(counts_real), y=list(words_real), ax=axes[0], palette='Blues_r')
    axes[0].set_title('Top 10 Words in Real News')
    axes[0].set_xlabel('Count')
    
    # Fake words plot
    words_fake, counts_fake = zip(*fake_freq)
    sns.barplot(x=list(counts_fake), y=list(words_fake), ax=axes[1], palette='Oranges_r')
    axes[1].set_title('Top 10 Words in Fake News')
    axes[1].set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig('most_common_words.png')
    plt.close()
    print("Saved plot: most_common_words.png")
    
    # 3. Vectorization with scikit-learn TfidfVectorizer
    print("\nVectorizing dataset using TF-IDF Vectorizer...")
    # Using parameters from PDF skeleton
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(df['clean_text'])
    
    print(f"Feature matrix shape: {X_vec.shape}")
    
    # Save the vectorizer to disk so Week 3 and 4 can use it
    vectorizer_file = "tfidf_vectorizer.pkl"
    with open(vectorizer_file, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Saved fitted TfidfVectorizer to: {vectorizer_file}")
    print("Feature Engineering completed successfully.")

if __name__ == "__main__":
    main()
