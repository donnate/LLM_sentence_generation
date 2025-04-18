import nltk

import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
# Make sure you've downloaded the necessary NLTK data:
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')

# 1) Create a custom tokenizer that:
#    - tokenizes
#    - lowercases
#    - removes non-alpha tokens
#    - removes English stopwords
#    - lemmatizes


def convert_to_dtm(data):
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(stopwords.words('english'))
    def custom_tokenizer(text):
        # Tokenize
        tokens = word_tokenize(text)
        # Keep only alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]
        # Lowercase
        tokens = [word.lower() for word in tokens]
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) 
                for word in tokens 
                if word not in english_stopwords]
        return tokens

    # 2) Create CountVectorizer with the custom tokenizer
    vectorizer = CountVectorizer(
        tokenizer=custom_tokenizer, 
        stop_words=None,  # we handle stopwords inside tokenizer
        min_df=2,         # optionally ignore words that appear in fewer than 2 docs
        max_df=0.5        # optionally ignore words that appear in >50% of docs
    )

    # -------------------------------------------------------
    # Assume you already have your DataFrame 'df' from the ArXiv data:
    # df = pd.DataFrame(test)  # test from your original code
    # df['text'] = df['title'] + ' ' + df['abstract']

    # 3) Fit the vectorizer on your text data and transform
    X_count = vectorizer.fit_transform(data)

    # 4) Check results
    print("Shape of the document-term matrix:", X_count.shape)
    print("Sample features:", vectorizer.get_feature_names_out()[:20])

    # Suppose X_count_filtered is an n×m sparse matrix.
    # sum(axis=0) returns a 1×m matrix of column sums.
    col_sums = np.array(X_count.sum(axis=1)).ravel()  # make it a 1D NumPy array
    inv_col_sums = 1.0 / col_sums                              

    # Build the m×m sparse diagonal matrix of inverse sums.
    D_inv = sp.diags(inv_col_sums, offsets=0)  # shape (m, m)

    # Multiply from the correct side. If you want to scale columns, multiply on the right:
    freq = D_inv @ X_count   

    return X_count, vectorizer, freq, col_sums


def align_on_confusion(y_true, clusters):
    m = confusion_matrix(y_true, clusters)

    row_sum = m.sum(axis=1)
    cost_matrix = (1-m / row_sum[:, np.newaxis])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1

    #print(confusion_matrix(y_true[index[0]],clusters  ))
    #### Train classif
    m = confusion_matrix(y_true,clusters)@P.T

    return m, np.sum(np.diag(m))/len(clusters) 



    


