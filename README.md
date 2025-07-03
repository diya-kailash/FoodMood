# 🍽️ FoodMood - AI Food Recommender

## Project Overview

FoodMood is an intelligent food recommendation system that uses advanced AI and machine learning to suggest personalized dishes based on your mood, preferences, and constraints. The system analyzes your natural language input to understand your cravings and matches them with the perfect food options from a curated dataset of Indian and international cuisines.

The project combines semantic similarity matching with AI-powered constraint analysis to provide highly personalized recommendations that consider factors like preparation time, budget, health consciousness, comfort seeking, and social context.

---

## 🚀 How to Run the Project

### Option 1: Command Line
```bash
python run.py
```

### Option 2: Streamlit App

```bash
streamlit run app.py
```

### Option 3: API Server

```bash
python api.py
```

Access the API at: `http://localhost:8000`

### First-Time Dependencies

```bash
pip install streamlit sentence-transformers pandas numpy scikit-learn plotly fastapi uvicorn groq python-dotenv nltk scipy statsmodels matplotlib seaborn requests python-multipart
```

### Environment Setup

1. Create a `.env` file in the project root.
2. Add your Groq API key:

```
GROQ_API_KEY=your-api-key-here
```

Get your API key from: [Groq Console](https://console.groq.com/keys)

---

## 🛠️ Tech Stack

### Core AI & Machine Learning

* Sentence Transformers (`all-MiniLM-L6-v2`) for semantic embeddings
* Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for reranking recommendations
* Scikit-learn for cosine similarity
* NumPy & Pandas for data processing

### AI Language Model

* Groq API (Llama3-70B-8192) for constraint analysis and explanations

### Web Frameworks

* Streamlit for interactive web application
* FastAPI for RESTful API
* Uvicorn as the ASGI server
* Plotly for interactive charts and graphs

### Performance & Optimization

* ThreadPoolExecutor for parallel processing
* LRU Cache for improved response times
* Dataclasses for structured data handling

---

## 📊 Dataset Overview

The project uses a comprehensive food dataset (`Dataset.csv`) with over 65 food items spanning multiple cuisines.

### Dataset Columns

* **Food Item**: Name of the dish (e.g., Idli, Pizza, Burger)
* **Food Type**: Category (Meal, Snack, Dessert)
* **Cuisine**: Regional or international (South Indian, Italian, Chinese)
* **Taste Notes**: Flavor profile (Spicy, Sweet, Tangy)
* **Texture**: Physical texture (Soft, Crispy, Creamy)
* **Prep Time**: Cooking duration (Quick, Moderate, Elaborate)
* **Budget**: Cost category (Cheap, Moderate, Expensive)
* **Social Context**: Dining situation (Family, Alone, Group, Date)
* **User Mood**: Suitable moods (Usual, Cravings, Comfort)

---

## ✨ Key Features

### AI-Powered Recommendations

* Natural language understanding of user mood
* Semantic similarity matching with transformer models
* Cross-encoder reranking for higher accuracy
* Constraint extraction and analysis using large language models
* Tradeoff analysis for balanced recommendations
* Multi-factor scoring combining similarity and constraints
* AI-generated explanations for recommendations

### Performance Highlights

* Parallel processing for faster responses
* Intelligent caching for quicker results
* Batch embedding for efficiency
* Real-time performance monitoring

### User Interfaces

* Streamlit web app with real-time recommendations
* FastAPI REST API with automatic documentation
* Responsive design and dark theme
* Interactive dataset visualizations

### User Experience

* Prompt suggestions for inspiration
* Expandable explanations for each recommendation
* Adjustable number of suggestions (5-12)
* Visual food cards with emojis
* Loading animations and progress indicators

---

