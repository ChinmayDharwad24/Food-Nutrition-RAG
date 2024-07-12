# Food Nutrition Recommendation System

This project implements a Food Nutrition Recommendation System using Streamlit, LangChain, Pinecone, and OpenAI embeddings. Users can input a food name to retrieve detailed nutrition information.

## Demo Video Link

    https://www.loom.com/share/f1f1137bca4048eb974cf2d1d7db8f07?sid=39aa0cdc-5708-4e79-bf54-2154eb6836fe

    
## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Keys and Environment Variables](#api-keys-and-environment-variables)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- Enter a food name to get corresponding nutrition information.
- Utilizes vector embeddings to efficiently retrieve relevant data.
- Dynamic integration with Pinecone for fast and scalable storage.
- User-friendly interface powered by Streamlit.

## Technologies Used

- **Python**: The programming language used for the backend logic.
- **Streamlit**: A framework for creating web applications for data science and machine learning.
- **LangChain**: A framework for building applications with language models.
- **Pinecone**: A vector database for storing and querying high-dimensional vectors.
- **OpenAI API**: Used for generating embeddings from food names and nutrition data.
- **dotenv**: For loading environment variables from a `.env` file.

## Setup and Installation

### Prerequisites

1. Python 3.7 or higher
2. pip (Python package installer)

### Step 1: Clone the Repository

```sh
git clone https://github.com/yourusername/food-nutrition-recommendation-system.git
cd food-nutrition-recommendation-system
```


### Step 2: Install Dependencies

```sh
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables
Create a .env file in the root directory of the project and add your Pinecone and OpenAI API keys:

PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key

### Step 4: Prepare the Data
Make sure you have a data.json file containing the food and nutrition information in the same directory. The format should be as follows:

```sh
[
    {
        "food_name": "Apple",
        "nutrition": "Calories: 52, Carbs: 14g, Fiber: 2.4g, Vitamin C: 14%"
    },
    {
        "food_name": "Banana",
        "nutrition": "Calories: 96, Carbs: 27g, Fiber: 2.6g, Vitamin C: 17%"
    }
]
```

### Step 5: Run the Application


```sh
streamlit run streamlit_app.py
```

Open your browser and go to http://localhost:8501 to access the application.

### Usage
1. Launch the Streamlit app as mentioned above.
2. Enter a food name in the input field.
3. Click "Enter" to retrieve nutrition information.
4. View the relevant nutrition details displayed below.

### Project Structure

```sh
food-nutrition-recommendation-system/
│
├── app.py              # Main application file
├── data.json           # JSON file containing food and nutrition data
├── requirements.txt     # List of dependencies
└── .env                # Environment variables
```

### API Keys and Environment Variables
To run this application, you need valid API keys for Pinecone and OpenAI. These should be set in the .env file as described above.

### Troubleshooting
No relevant information found: Ensure that the food name you are querying matches exactly with entries in data.json. Check for case sensitivity and whitespace.
API errors: Double-check your API keys in the .env file and ensure they are correctly set up.


### Contributing
Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

