# 🛒🚀Recommender-System-Ecommerce
## Overview
It is a real-time recommendation system for an e-commerce platform. The system provides product recommendations based on the user’s recent history and ratings using Hybrid Filtering, which leverages collaborative and content-based filtering.
# Implementation
## Hybrid Recommendation Implementation (hybrid_recommendations())
The hybrid recommendation system combines Content-Based Filtering and Collaborative Filtering to provide accurate and personalized product recommendations.
### How It Works?
1. Content-Based Filtering: Finds similar products using TF-IDF and cosine similarity based on product tags.
2. Collaborative Filtering: Uses SVD (Singular Value Decomposition) to recommend products based on user interactions.
3. Hybrid Approach: Merges results from both methods, removes duplicates, and returns the top N recommendations.

## Content-Based Filtering (content_based_filter())
This method recommends products similar to a given item based on text similarity (TF-IDF).

### Function Steps:
1. Extract product tags (e.g., categories, descriptions).
2. Uses TF-IDF Vectorization to convert textual data into numerical features.
3. Computes cosine similarity between items. 
Cosine similarity between two vectors **A** and **B** is calculated as:

**cosine_similarity(A, B) = (A · B) / (‖A‖ × ‖B‖)**

Where:  
- **A · B** is the **dot product** of vectors **A** and **B**.  
- **‖A‖** and **‖B‖** are the **Euclidean norms** (magnitudes) of the vectors.  

4. Cosine similarity measures the **angle between two vectors**, helping identify product similarity in **content-based filtering**.
5. Recommends the top N most similar products based on the selected item.

## Collaborative Filtering (collaborative_based_filter())
This method recommends products based on user-item interactions, using Singular Value Decomposition (SVD).

### Function Steps:
1. Trains an SVD model on user-product ratings.
2. Predicts missing ratings for products a user hasn’t interacted with.
3. Sorts products based on predicted scores and recommends top N products.

## Streamlit Application
The Streamlit-based web application allows users to: 

✔ Select User ID

✔ Select Product Name

✔ Choose Number of Recommendations

✔ View recommended products in a grid layout with images

### How to run the App in your system
1. Download the Application folder in the repository
2. Install the requirements
3. In CLI, Give the command **streamlit run app.py**

Note: Change the path of dataset according to your system

## Requirements
Python
Scikit learn
Surprise package
pandas
numpy

Thank you for all the references and authors listed below:
* https://bgiri-gcloud.medium.com/building-an-effective-recommendation-system-for-e-commerce-a-step-by-step-guide-bafae59862e1
* https://medium.com/towards-data-science/various-implementations-of-collaborative-filtering-100385c6dfe0
* https://naomy-gomes.medium.com/the-cosine-similarity-and-its-use-in-recommendation-systems-cb2ebd811ce1
* https://surprise.readthedocs.io/en/stable/index.html
* https://scikit-learn.org/0.21/documentation.html

