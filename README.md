# Sentiment_Analysis


## Overview
This project focuses on developing a text classification system for sentiment analysis, classifying movie reviews as positive or negative. Both traditional machine learning (Logistic Regression) and deep learning (LSTM) models were implemented. The 50k IMDB review dataset from Kaggle was used for training and evaluation.
Different experiments were performed using various parameters to find a good combination that would give the best model performance.

## Dataset
The dataset consists of 50,000 movie reviews with two columns: 'review' and 'sentiment'. The sentiment column labels the reviews as either positive or negative. The dataset is balanced, containing 25,000 positive and 25,000 negative reviews.

![sentment_dist](https://github.com/user-attachments/assets/c1f84e61-3f5e-42c9-a146-99f6c555f4a3)

Exploratory Data Analysis (EDA)
- Word clouds were generated to visualize frequent words in positive and negative reviews.
- Histogram analysis helped determine the optimal sequence length for LSTM input.
- No missing values were found in the dataset.

## Preprocessing Steps
The following preprocessing steps were applied to prepare the text data for modeling:
- Converted all text to lowercase.
- Removed HTML tags (e.g., `<br />` which frequently appeared in raw text).
- Eliminated URLs and punctuation marks.
- Tokenized text into individual words.
- Removed stopwords (e.g., "the," "is") to focus on meaningful words.
- Applied padding to standardize input sequence lengths.

These preprocessing steps helped clean the data, enhance model performance, and ensure consistency in input representations.

## Feature Extraction
Two feature extraction techniques were used:
- **TF-IDF Vectorization**: Captured the importance of words in the dataset.
- **Word2Vec Embeddings**: Pre-trained embeddings were used to capture semantic relationships between words. The embedding matrix was created using the tokenizer’s word index and a pre-trained Word2Vec model.


## Model Architecture

### 1. Traditional Machine Learning Model (Logistic Regression)

- Used TF-IDF features.
- Implemented using Scikit-learn’s `LogisticRegression`.
- Trained with a maximum iteration of 1000.

### 2. Deep Learning Model (LSTM)

- Used Word2Vec embeddings.
- **Embedding Layer:** Converted words into dense vector representations.
- **LSTM Layer:** Captured long-term dependencies in text.
- **Dense Output Layer:** Used a sigmoid activation function for binary classification.
- **Dropout & Recurrent Dropout:** Was applied to prevent overfitting.

## Model Training and Hyperparameters
- The dataset was split into 80% training and 20% testing with a fixed random state (42).
- **Logistic Regression** used TF-IDF features and trained using the Adam optimizer.
- **LSTM** was trained using the following hyperparameters:
  - Epochs: 5
  - Batch size: 32
  - Dropout: 0.2
  - Recurrent dropout: 0.2
  - Optimizer: Adam
  - Loss function: Binary cross-entropy

## Execution Steps

### Prerequisites

Have the following installed:

- Python 3. (and above)
- TensorFlow/Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

### Steps to Run the Project

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/cgyireh1/Sentiment_Analysis.git
   cd Sentiment_Analysis
   ```

2. **Open the Notebook in Google Colab:**

   - Upload or open the notebook file ``Sentiment_Analysis_Group_3.ipynb`` in the notebook folder

3. **Mount Youur Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Update File Paths:**

   Check the file paths when loading data. For example:

   ```python
   data = pd.read_csv('/content/drive/MyDrive/ML_Projects 👩‍💻/sentiment_analysis/IMDB Dataset.csv')
   ```

   Modify these paths based on your Google Drive folder structure or where your files are located.
   
- Run through the code to generate the results and experiment with different hyperparameters to observe their impact on the model's performance. This will allow you to fine-tune the model and potentially achieve even better results.

## Results & Evaluation For The Best Models Picked

**Logistic Regression:**
- Accuracy: 88.83%
- Loss: 28.42%
- Faster training time but lacks sequential context handling.

**LSTM Model:**
- Accuracy: 88.83%
- Loss: 27.29%
- Better at capturing word relationships and long-term dependencies.
  
  <img src="https://github.com/user-attachments/assets/00c40481-379f-490e-9fad-606000dde1d8" width="300">
  <img src="https://github.com/user-attachments/assets/14b09e8c-97ad-44d0-b711-ef483e702e90" width="300">

## Conclusion
- The **two-layer LSTM model (128 & 64 units) performed the best with an accuracy of 88.83% and a ROC score of 0.96**.
- **Logistic Regression provided a simpler and faster alternative** with similar accuracy but lacked the ability to capture sequential dependencies.
- The two-layer LSTM model demonstrated superior performance, validating the choice of LSTM for sentiment analysis tasks especially when capturing long-term text dependencies

