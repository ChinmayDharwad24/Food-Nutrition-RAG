import os
import json
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
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
        return embeddings.embed_documents([text])[0]

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

# Initialize LangChain components
vectorstore = Pinecone(index=index, embedding=embeddings, text_key='food_name', namespace="food-nutrition-namespace")

# Create a retriever
retriever = vectorstore.as_retriever()

# Helper function to format nutrition information
def format_nutrition(nutrition_str):
    nutrition_items = nutrition_str.split(', ')
    return '\n'.join(nutrition_items)

# Helper function to check relevance of the query
def check_query_relevance(query):
    try:
        docs = retriever.get_relevant_documents(query)
        return len(docs) > 0
    except Exception as e:
        st.error(f"An error occurred in relevance check: {str(e)}")
    return False

# Streamlit app
st.title("Food Nutrition Recommendation System")
st.write("Enter a food name below to get nutrition information.")

# User input
user_input = st.text_input("Food Name")

if user_input:
    try:
        # Check relevance of the query
        if check_query_relevance(user_input):
            # Retrieve documents based on the food name
            docs = retriever.get_relevant_documents(user_input)
            
            if docs:
                st.write("Relevant Information:")
                for doc in docs:
                    food_name = doc.metadata.get('food_name', 'Unknown food')
                    nutrition = doc.metadata.get('nutrition', 'No nutrition information found.')
                    # st.write(f"Food Name: {food_name}")
                    st.write("Nutrition:")
                    st.write(format_nutrition(nutrition))
            else:
                st.write("Response: I'm sorry, but I couldn't find any nutrition information for that food.")
        else:
            st.write("I am sorry, but the system does not contain relevant information for your query.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
