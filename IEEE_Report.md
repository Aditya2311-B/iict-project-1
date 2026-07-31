# AI-Powered Fake News Detection Using Text Classification
**Summer Internship Program in AI & ML Machine Learning 2026**  
*Indian Institute of Computing and Technology (IICT)*

---

## Abstract
This report presents the design and implementation of a complete machine learning pipeline built from scratch to classify news articles as either real or fake. Over a 30-day workflow, we implemented custom text preprocessing, manual tokenization, stopword removal, exploratory data analysis, and mathematical feature extraction. We evaluated four major machine learning algorithms: K-Nearest Neighbors (KNN), Logistic Regression, Random Forest, and a Multi-Layer Perceptron (MLP) Simple Neural Network. Our results show that the MLP Neural Network achieved the highest classification accuracy of **92.06%**, closely followed by Logistic Regression at **90.79%** and Random Forest at **90.40%**, with KNN yielding **85.79%**.

---

## 1. Introduction
### 1.1 Problem Statement
The proliferation of digital news media and social network platforms has drastically increased the speed and volume of information dissemination. While this facilitates rapid global communication, it has also enabled the widespread propagation of "Fake News"—factually incorrect or intentionally misleading articles created to influence opinions or generate clickbait revenue. Automating the detection of fake news is a critical task for maintaining public trust and social stability.

### 1.2 Objective
The goal of this project is to build an end-to-end Machine Learning pipeline to identify fake news. As students, we implemented this system to understand the underlying mathematical concepts behind natural language preprocessing, feature representation (Bag-of-Words, TF-IDF, dense word embeddings), and supervised classification algorithms.

---

## 2. Dataset Description
### 2.1 Data Source and Size
We used the standard public Fake vs. Real News dataset originally compiled by George McIntire. The dataset consists of **6,335 articles**, balanced between two labels:
*   **REAL**: 3,171 articles (50.06%)
*   **FAKE**: 3,164 articles (49.94%)

### 2.2 Features
The dataset contains the following features for each article:
1.  **ID/Index**: Unique identifier.
2.  **Title**: The headline of the news article (Textual).
3.  **Text**: The body text of the news article (Textual, the main input feature).
4.  **Label**: Binary target variable indicating whether the news is `REAL` or `FAKE` (Categorical target).

---

## 3. Methodology
Our implementation follows a structured 4-week machine learning pipeline shown below:

```
[Raw text data] -> [Preprocessing (Lowercase, Punctuation removal)] 
                -> [Manual Tokenization & Stopword filtering] 
                -> [Feature Extraction (TF-IDF Vectorization)] 
                -> [Supervised Classifier Training (KNN, LogReg, RF, NeuralNet)] 
                -> [Evaluation & Prediction]
```

### 3.1 Preprocessing and Manual Tokenization (Week 1)
Text cleaning is essential to remove noise before vectorization. We built a custom preprocessing function:
1.  **Lowercasing**: Standardized all letters to lowercase to ensure consistency (e.g., "News" and "news" are mapped to the same term).
2.  **Punctuation and Numeric Removal**: Removed all special characters, symbols, and numbers using regular expressions (`[^a-zA-Z\s]`) to retain only alphabetical terms.
3.  **Manual Tokenization**: Split text by whitespace into individual elements (tokens) using string manipulation without relying on pre-built libraries.
4.  **Stopwords Filtering**: Filtered out highly frequent but uninformative words (e.g., "the", "is", "and", "in", "to") using a custom English stopword list.

### 3.2 Feature Engineering (Week 2)
Computers cannot directly read text, so we convert string lists into mathematical vectors:
*   **Bag-of-Words (BoW)**: Represents text as a count vector based on word frequency.
*   **TF-IDF (Term Frequency-Inverse Document Frequency)**: Adjusts word counts by their overall document frequency. 
    $$\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total terms in } d}$$
    $$\text{IDF}(t) = \log\left(\frac{N}{1 + \text{DF}(t)}\right) + 1$$
    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
    We utilized `TfidfVectorizer(max_features=5000)` to generate a sparse matrix containing the 5,000 most informative terms across the corpus.
*   **Dense Embeddings**: Demonstrated in code by mapping words to fixed-size vector representations and averaging them to compute document-level semantic embeddings.

### 3.3 Classification Algorithms (Week 3)
We trained four diverse models to compare performance:
1.  **K-Nearest Neighbors (KNN)**: A non-parametric classifier. Classifies an article by finding the majority vote among its $k$ closest neighbors ($k=5$) in the multi-dimensional TF-IDF space.
2.  **Logistic Regression**: A parametric model that calculates the probability of class labels using a sigmoid function:
    $$P(y=1|x) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$
3.  **Random Forest**: An ensemble method consisting of 100 decision trees. It reduces variance and prevents overfitting by voting across multiple trees trained on bootstrapped subsets of the data.
4.  **Simple Neural Network (MLP)**: A Multi-Layer Perceptron containing an input layer (5000 features), a hidden layer (100 neurons), and a binary output layer using backpropagation to optimize classification weights.

---

## 4. Experimental Results (Week 4)

### 4.1 Quantitative Evaluation
We split the dataset into **80% training data** (5,038 articles) and **20% testing data** (1,260 articles). Model performance metrics are summarized below:

| Model Name | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **KNN** | 85.79% | 0.8584 | 0.8579 | 0.8579 |
| **Logistic Regression** | 90.79% | 0.9080 | 0.9079 | 0.9079 |
| **Random Forest** | 90.40% | 0.9040 | 0.9040 | 0.9040 |
| **Simple Neural Net (MLP)** | **92.06%** | **0.9207** | **0.9206** | **0.9206** |

### 4.2 Key Visualizations
*   **`label_distribution.png`**: Confirms that our dataset is balanced (~50% Real vs. ~50% Fake).
*   **`article_length_distribution.png`**: Shows that FAKE news articles in this dataset tend to be slightly shorter on average (~385 words) compared to REAL news articles (~493 words).
*   **`model_accuracy_comparison.png`**: Visually maps the accuracy scores, indicating the neural network out-performed traditional algorithms.
*   **`confusion_matrices.png`**: Plots a 2x2 grid representing the True Positives, False Positives, True Negatives, and False Negatives for each classifier.

---

## 5. Discussion
### 5.1 Parametric vs. Non-Parametric Models
*   **Non-Parametric (KNN)**: KNN makes no assumptions about the distribution of features. However, it suffers in high-dimensional text classification (curse of dimensionality) because distance calculations become less distinguishable, resulting in the lowest accuracy (**85.79%**) and slower test-time inference.
*   **Parametric Models (Logistic Regression, Neural Network)**: These models assume a functional form and optimize coefficients during training. Logistic Regression performed extremely fast and achieved an impressive **90.79%** accuracy, proving highly effective for linear relationships in TF-IDF vectors.
*   **Neural Network (MLP)**: The MLP learned non-linear boundaries by combining feature representations in its hidden layer, achieving the peak score of **92.06%**.

---

## 6. Conclusion
### 6.1 Insights
Text classification using TF-IDF features and simple models is highly viable for detecting misinformation. Logistic regression remains a robust, computationally cheap baseline, while a simple neural net provides the highest accuracy.

### 6.2 Limitations and Future Scope
*   **Word Order**: TF-IDF models ignore word order and context (bag-of-words assumption).
*   **Future Scope**: Implement advanced sequential deep learning models such as LSTMs or pre-trained transformers (e.g., BERT) to understand semantic context better.

---

## 7. Appendix - Python Code Structure
*   `download_dataset.py`: Python script fetching the George McIntire CSV dataset.
*   `week1_preprocessing.py`: Implements text cleaning and manual tokenization.
*   `week2_feature_engineering.py`: Computes BoW, TF-IDF manual concepts, and EDA figures.
*   `week3_model_training.py`: Builds and trains KNN, Logistic Regression, Random Forest, and MLP models.
*   `week4_evaluation.py`: Generates confusion matrices and compares performance scores.
*   `run_pipeline.py`: Wrapper executing the entire week-by-week pipeline seamlessly.
