import os
import json
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize Pinecone
pc = PineconeClient(api_key=pinecone_api_key)

# Define index name and check if it exists
index_name = "food-nutrition-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Load the JSON data from file and upsert to Pinecone if index is empty
if index.describe_index_stats()['total_vector_count'] == 0:
    with open("data.json", "r") as file:
        data = json.load(file)

    # Function to convert food name and nutrition to vector
    def get_vector(food_name, nutrition):
        text = f"Food: {food_name}. Nutrition: {nutrition}"
        return embeddings.embed_query(text)

    # Prepare vectors and metadata
    vectors = []
    for i, item in enumerate(data):
        food_name = item['food_name']
        nutrition = item['nutrition']
        vector = get_vector(food_name, nutrition)
        vectors.append({
            "id": str(i),
            "values": vector,
            "metadata": {"food_name": food_name, "nutrition": nutrition}
        })

    # Upsert vectors to Pinecone with namespace
    namespace = "food-nutrition-namespace"
    index.upsert(vectors=vectors, namespace=namespace)

# Helper function to format nutrition information
def format_nutrition(nutrition_str):
    nutrition_items = nutrition_str.split(', ')
    return '\n'.join(nutrition_items)

# Streamlit app
st.title("Food Nutrition Recommendation System")
st.write("Enter a food name below to get nutrition information.")

# User input
user_input = st.text_input("Food Name")

if user_input:
    # Query Pinecone
    query_vector = embeddings.embed_query(user_input)
    results = index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="food-nutrition-namespace")
    
    # Set a similarity threshold
    similarity_threshold = 0.75

    if results['matches'] and results['matches'][0]['score'] > similarity_threshold:
        food_info = results['matches'][0]['metadata']
        food_name = food_info.get('food_name', 'Unknown food')
        nutrition = food_info.get('nutrition', 'No nutrition information found.')
        
        st.write(f"Food Name: {food_name}")
        st.write("Nutrition:")
        st.write(format_nutrition(nutrition))
    else:
        st.write("Food not found in data")
