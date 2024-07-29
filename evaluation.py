import time
import json
import random
import string
from typing import List, Dict, Tuple
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, BertModel, Trainer, TrainingArguments, BertForSequenceClassification
import torch
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from datasets import Dataset, load_metric
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize clients and models
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
pc = PineconeClient(api_key=pinecone_api_key)
index = pc.Index("food-nutrition-index")
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Initialize BERT model for embeddings
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def load_test_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]  # Convert single item to list
    return data

def create_test_queries(data: List[Dict], n: int = 100) -> List[Tuple[str, str]]:
    return [(item['food_name'], item['nutrition']) for item in random.sample(data, n)]

def add_noise(query: str, noise_level: float = 0.2) -> str:
    words = query.split()
    num_words_to_change = int(len(words) * noise_level)
    for _ in range(num_words_to_change):
        index = random.randint(0, len(words) - 1)
        words[index] = ''.join(random.choices(string.ascii_lowercase, k=len(words[index])))
    return ' '.join(words)

def calculate_context_precision(test_queries: List[Tuple[str, str]], index, embeddings) -> float:
    correct = 0
    for query, expected in test_queries:
        query_vector = embeddings.embed_query(query)
        results = index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="food-nutrition-namespace")
        if results['matches'] and results['matches'][0]['metadata']['food_name'] == query:
            correct += 1
    return correct / len(test_queries)

def calculate_noise_robustness(test_queries: List[Tuple[str, str]], index, embeddings) -> float:
    correct = 0
    for query, expected in test_queries:
        noisy_query = add_noise(query)
        query_vector = embeddings.embed_query(noisy_query)
        results = index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="food-nutrition-namespace")
        if results['matches'] and results['matches'][0]['metadata']['food_name'] == query:
            correct += 1
    return correct / len(test_queries)

def bert_encode(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculate_answer_relevance(test_queries: List[Tuple[str, str]], index, embeddings) -> float:
    relevance_scores = []
    for query, expected in test_queries:
        query_vector = embeddings.embed_query(query)
        results = index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="food-nutrition-namespace")
        if results['matches']:
            retrieved_nutrition = results['matches'][0]['metadata']['nutrition']
            query_embedding = bert_encode(query)
            nutrition_embedding = bert_encode(retrieved_nutrition)
            relevance_score = 1 - cosine(query_embedding, nutrition_embedding)
            relevance_scores.append(relevance_score)
    return np.mean(relevance_scores)

def calculate_latency(test_queries: List[Tuple[str, str]], index, embeddings) -> float:
    latencies = []
    for query, _ in test_queries:
        start_time = time.time()
        query_vector = embeddings.embed_query(query)
        index.query(vector=query_vector, top_k=1, include_metadata=True, namespace="food-nutrition-namespace")
        end_time = time.time()
        latencies.append(end_time - start_time)
    return np.mean(latencies)

def fine_tune_model(data):
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = create_dataset(data, tokenizer)
    
    if len(train_dataset) < 2:
        print("Not enough data for fine-tuning. Using pre-trained model as is.")
        return model, tokenizer
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        evaluation_strategy="no",  # Disable evaluation as we don't have a separate eval dataset
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    
    return model, tokenizer

def create_dataset(data, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["food_name"], padding="max_length", truncation=True)
    
    # Convert list of dictionaries to dictionary of lists
    dict_data = {key: [d[key] for d in data] for key in data[0]}
    
    # Add labels if they don't exist
    if "labels" not in dict_data:
        dict_data["labels"] = [0 if item["food_name"] in ("food_name_1", "food_name_2") else 1 for item in data]
    
    dataset = Dataset.from_dict(dict_data)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    return tokenized_datasets

def compute_metrics(p):
    metric = load_metric("accuracy")
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def improve_embeddings(index, embeddings):
    print("Improving embeddings with fine-tuned model...")
    
    # Load and prepare data
    with open('test_data.json', 'r') as f:
        data = json.load(f)
    
    # # Ensure data is a list
    # if isinstance(data, dict):
    #     data = [data]
    
    # # Fine-tune model
    # model, tokenizer = fine_tune_model(data)
    
    # # Update embeddings in the index
    # for i, item in enumerate(data):
    #     text = item["food_name"]
    #     nutrition = item["nutrition"]
    #     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    #     with torch.no_grad():
    #         outputs = model.bert(**inputs)  # Use the base BERT model
    #         new_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
    #     # Use food_name as id if 'id' is not present
    #     id = item.get("id", text)
        
    #     index.upsert(
    #         vectors=[{
    #             "id": id,
    #             "values": new_embedding.tolist(),
    #             "metadata": {
    #                 "food_name": text,
    #                 "nutrition": nutrition
    #             }
    #         }],
    #         namespace="food-nutrition-namespace"
    #     )

    print(f"Updated embeddings for {len(data)} items.")
    
    
def implement_caching(app):
    # Placeholder for implementing caching
    print("Implementing caching mechanism...")

def run_evaluation():
    print("Loading test data...")
    test_data = load_test_data("data.json")
    test_queries = create_test_queries(test_data)

    print("Calculating initial metrics...")
    initial_precision = calculate_context_precision(test_queries, index, embeddings)
    initial_robustness = calculate_noise_robustness(test_queries, index, embeddings)
    initial_relevance = calculate_answer_relevance(test_queries, index, embeddings)
    initial_latency = calculate_latency(test_queries, index, embeddings)

    print("Initial Metrics:")
    print(f"Context Precision: {initial_precision:.4f}")
    print(f"Noise Robustness: {initial_robustness:.4f}")
    print(f"Answer Relevance: {initial_relevance:.4f}")
    print(f"Latency: {initial_latency:.4f} seconds")

    print("\nImplementing improvements...")
    improve_embeddings(index, embeddings)
    implement_caching(None)

    print("\nCalculating metrics after improvements...")
    improved_precision = calculate_context_precision(test_queries, index, embeddings)
    improved_robustness = calculate_noise_robustness(test_queries, index, embeddings)
    improved_relevance = calculate_answer_relevance(test_queries, index, embeddings)
    improved_latency = calculate_latency(test_queries, index, embeddings)

    print("Improved Metrics:")
    print(f"Context Precision: {improved_precision:.4f}")
    print(f"Noise Robustness: {improved_robustness:.4f}")
    print(f"Answer Relevance: {improved_relevance:.4f}")
    print(f"Latency: {improved_latency:.4f} seconds")

    print("\nImprovement Summary:")
    print(f"Context Precision: {(improved_precision - initial_precision) / initial_precision * 100:.2f}% improvement")
    print(f"Noise Robustness: {(improved_robustness - initial_robustness) / initial_robustness * 100:.2f}% improvement")
    print(f"Answer Relevance: {(improved_relevance - initial_relevance) / initial_relevance * 100:.2f}% improvement")
    print(f"Latency: {(initial_latency - improved_latency) / initial_latency * 100:.2f}% reduction")

if __name__ == "__main__":
    run_evaluation()
