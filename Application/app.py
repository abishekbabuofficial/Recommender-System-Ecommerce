import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("wallmart_data.csv")
data['Category'].fillna('', inplace=True)
data['Brand'].fillna('', inplace=True)
data['Description'].fillna('', inplace=True)

def content_based_filter(dataset, item_name, top_n=10):
    if item_name not in dataset['Name'].values:
        print("Item", item_name," not found in the data.")
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(dataset['Tags'])
    consine = cosine_similarity(tfidf_matrix, tfidf_matrix)
    item_index = dataset[dataset['Name'] == item_name].index[0]
    similar_items = list(enumerate(consine[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = dataset.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
    return recommended_items_details

reader = Reader(rating_scale=(0, 5))
new_data = Dataset.load_from_df(data[['ID', 
                                  'ProdID', 
                                  'Rating']], reader)

algo = SVD()
trainset = new_data.build_full_trainset()
algo.fit(trainset)

def collaborative_based_filter(user_id, top_n):
    testset = trainset.build_anti_testset()
    testset = filter(lambda x: x[0] == user_id, testset)
    predictions = algo.test(testset)
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommendations = [prediction.iid for prediction in predictions[:top_n]]
    recommended_items = data[data['ProdID'].isin(recommendations)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
    return recommended_items.head(top_n)

def hybrid_recommendations(train_data,target_user_id, item_name, top_n):

    content = content_based_filter(train_data,item_name, top_n)

    collaborative = collaborative_based_filter(target_user_id, top_n)
    
    hybrid_rec = pd.concat([content, collaborative]).drop_duplicates()
    
    return hybrid_rec.head(top_n)

# Streamlit UI
st.title("🛒E-commerce Product Recommendation System")

# User selection
target_user_id = st.selectbox("Select User ID", data["ID"].unique())
top_n = st.selectbox("Select Number of Recommendations", [5, 10, 15, 20])
item_name = st.selectbox("Select Item", data["Name"].unique())

# Generate recommendations
if st.button("Get Recommendations"):
    recommendations = hybrid_recommendations(data, target_user_id, item_name, top_n)
    st.subheader("Recommended Products")
    cols = st.columns(3)  # Creating a grid layout with 3 columns
    
    for index, row in recommendations.iterrows():
        with cols[index % 3]: 
            st.image(row['ImageURL'], width=200)
            st.write(f"**{row['Name']}**")
            st.write(f"Brand: {row['Brand']}")
            st.write(f"Rating: {row['Rating']} ⭐ | Reviews: {row['ReviewCount']}")
            st.markdown("---")