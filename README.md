# 🍽️ FoodMood - AI Food Recommender

## Project Overview

FoodMood is an intelligent food recommendation system that uses advanced AI and machine learning to suggest personalized dishes based on your mood, preferences, and constraints. The system analyzes your natural language input to understand your cravings and matches them with the perfect food options from a curated dataset of Indian and international cuisines.

The project combines semantic similarity matching with AI-powered constraint analysis to provide highly personalized recommendations that consider factors like preparation time, budget, health consciousness, comfort seeking, and social context.

---

## 🚀 How to Run the Project

### Option 1 — Command Line

```bash
python run.py
```

### Option 2 — Streamlit App

```bash
streamlit run app.py
```

### Option 3 — API Server

```bash
python api.py
```

Access the API at: [http://localhost:8000](http://localhost:8000)

---

### 📦 First-Time Dependencies

Install the necessary packages (one-time setup):

```bash
pip install streamlit sentence-transformers pandas numpy scikit-learn plotly fastapi uvicorn groq python-dotenv nltk scipy statsmodels matplotlib seaborn requests python-multipart
```

---

### 🔑 Environment Setup

1. Create a `.env` file in the project root.
2. Add your Groq API key:

   ```
   GROQ_API_KEY=your-api-key-here
   ```

   You can get your API key from: [https://console.groq.com/keys](https://console.groq.com/keys)

---

## 🛠️ Tech Stack Used

### Core AI & Machine Learning

* Sentence Transformers (`all-MiniLM-L6-v2`) for semantic embeddings
* Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) for reranking recommendations
* Scikit-learn for similarity calculations
* NumPy and Pandas for data processing

### AI Language Model

* Groq API (Llama3-70B-8192) for intelligent constraint analysis and explanations

### Web Technologies

* Streamlit for interactive web app
* FastAPI for RESTful API backend
* Uvicorn as ASGI server
* Plotly for interactive charts

### Performance & Optimization

* ThreadPoolExecutor for parallel processing
* LRU Caching for faster responses
* Dataclasses for structured constraint handling

---

## 📊 Dataset Overview

FoodMood uses a curated dataset (`Dataset.csv`) with 100+ food items from Indian and international cuisines.

### Dataset Columns:

| Column         | Description                                            |
| -------------- | ------------------------------------------------------ |
| Food Item      | Name of the dish (e.g., Idli, Pizza, Burger)           |
| Food Type      | Meal category (Meal, Snack, Dessert)                   |
| Cuisine        | Regional/International cuisine (South Indian, Italian) |
| Taste Notes    | Flavor profile (Spicy, Sweet, Tangy)                   |
| Texture        | Physical texture (Soft, Crispy, Crunchy)               |
| Prep Time      | Cooking duration (Quick, Moderate, Elaborate)          |
| Budget         | Cost level (Cheap, Moderate, Expensive)                |
| Social Context | Dining setting (Family, Alone, Group, Date)            |
| User Mood      | Applicable moods (Usual, Cravings, Comfort)            |

---

## ✨ Key Features Summary

### 🧠 AI-Powered Recommendations

* Understands mood through natural language
* Semantic similarity matching using transformer embeddings
* Reranking for higher accuracy via cross-encoder
* Automated constraint extraction using LLMs
* Trade-off analysis for more balanced suggestions
* Multi-factor scoring combining similarity and constraints
* AI-generated explanations for each recommendation

### ⚡ Performance Features

* Parallel processing for fast recommendations
* Intelligent caching system
* Batch embedding creation for efficiency
* Real-time speed monitoring and optimization

### 🖥️ User Interfaces

* Streamlit Web App with real-time interactivity
* FastAPI REST API with automatic Swagger docs
* Dark-themed responsive UI with visualizations

### 📱 User Experience

* Sample prompt suggestions
* Expandable AI explanations
* Customizable number of recommendations (5–12)
* Visual food cards with emojis
* Loading animations and progress bars

---

