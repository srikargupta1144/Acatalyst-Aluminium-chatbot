import streamlit as st
import pandas as pd
import numpy as np
import openai
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Function to retrieve the API key
def get_api_key(email):
    url = "http://52.66.239.27:8504/get_keys"
    payload = {"email": email}
    response = requests.post(url, json=payload)
    return response.json().get('key')

# Fetch and set OpenAI API key
email = os.getenv('EMAIL')
openai.api_key = get_api_key(email)

# Load the scraped data with embeddings
df = pd.read_csv('embeddings.csv')

# Convert string representation of lists back to lists using json.loads
def safe_load_embedding(embedding_str):
    try:
        # Debugging print statement
        print(f"Loading embedding: {embedding_str}")
        return np.array(json.loads(embedding_str), dtype=np.float32)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return np.array([])  # Handle or log errors as needed

df['embedding'] = df['embedding'].apply(safe_load_embedding)

# Function to get embeddings using the latest OpenAI API
def get_embedding(text):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        # Debugging print statement
        print(f"OpenAI response: {response}")
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Error retrieving embedding: {e}")
        return np.array([])  # Handle or log errors as needed

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)
    if vec1.size == 0 or vec2.size == 0:
        return 0  # Handle cases where embeddings might be empty or invalid
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

st.title("Aluminium Industry News Chatbot")

# User input
user_query = st.text_input("Enter your query:")

# Add a submit button
if st.button("Submit"):
    if user_query:
        # Convert query to embedding
        query_embedding = get_embedding(user_query)

        # Find the most similar article
        df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, query_embedding))
        most_similar_article = df.loc[df['similarity'].idxmax()]

        st.write("### Most Relevant Article")
        st.write("**Title**: ", most_similar_article['title'])
        st.write("**Summary**: ", most_similar_article['summary'])
        st.write("**Date**: ", most_similar_article['date'])
    else:
        st.write("Please enter a query.")
