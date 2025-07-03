import streamlit as st
import pandas as pd
from src.food_recommender import FoodRecommender
import time
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title=" FoodMood - AI Food Recommender",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main theme colors */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #2d2d2d 100%);
        color: #ffffff;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #ff6b35, #f7931e, #ffc845);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
    }
    
    .main-header h1 {
        color: #1a1a2e;
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #1a1a2e;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    /* Input section styling */
    .input-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 107, 53, 0.3);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    /* Recommendation cards */
    .food-card {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(247, 147, 30, 0.1));
        border: 2px solid rgba(255, 107, 53, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .food-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.4);
        border-color: rgba(255, 107, 53, 0.6);
    }
    
    .food-title {
        color: #ffc845;
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .food-details {
        color: #e0e0e0;
        margin: 0.3rem 0;
    }
    
    .food-score {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: #1a1a2e;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.95) !important;
    }
    
    /* Additional sidebar styling */
    .css-1l02zno {
        background: rgba(0, 0, 0, 0.95) !important;
    }
    
    .css-17eq0hr {
        background: rgba(0, 0, 0, 0.95) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: #1a1a2e;
        border: none;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.7rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #f7931e, #ffc845);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 107, 53, 0.4);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 107, 53, 0.3);
        border-radius: 10px;
        color: #ffffff;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(255, 107, 53, 0.6);
        box-shadow: 0 0 10px rgba(255, 107, 53, 0.3);
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 204, 69, 0.1);
        border: 1px solid rgba(255, 204, 69, 0.3);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #ffc845;
    }
    
    .stat-label {
        color: #e0e0e0;
        font-size: 0.9rem;
    }
    
    /* Loading animation */
    .loading-text {
        text-align: center;
        color: #ffc845;
        font-size: 1.2rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

FOOD_EMOJIS = {
    'Meal': '🍽️',
    'Light Meal': '🥗',
    'Snack': '🍿',
    'Dessert': '🍰',
    'Beverage': '🥤',
}


@st.cache_resource
def load_recommender():
    try:
        return FoodRecommender()
    except Exception as e:
        st.error(f"Error loading recommender system: {str(e)}")
        return None

def display_food_card(food_item, index):
    food_emoji = FOOD_EMOJIS.get(food_item['food_type'], '🍽️')
    combined_score = food_item.get('combined_score', 'N/A')
    score_html = f'<div class="food-score">⭐ Match Score: {combined_score}</div>'
    
    card_html = f"""
    <div class="food-card">
        <div class="food-title">
            {food_emoji} {food_item['food_name']} 
        </div>
        <div class="food-details">
            <strong>🏛️ Cuisine:</strong> {food_item['cuisine']}
        </div>
        <div class="food-details">
            <strong>🍽️ Type:</strong> {food_item['food_type']}
        </div>
        <div class="food-details">
            <strong>👅 Taste:</strong> {food_item['taste_notes']}
        </div>
        <div class="food-details">
            <strong>🤚 Texture:</strong> {food_item['texture']}
        </div>
        <div class="food-details">
            <strong>⏱️ Prep Time:</strong> {food_item['prep_time']}
        </div>
        <div class="food-details">
            <strong>💰 Budget:</strong> {food_item['budget']}
        </div>
        <div class="food-details">
            <strong>👥 Social Context:</strong> {food_item['social_context']}
        </div>
        {score_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def main(): 
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ FoodMood - AI Food Recommender</h1>
        <p>Discover perfect dishes based on your unique mood and preferences!🎯</p>
    </div>
    """, unsafe_allow_html=True)
    
    recommender = load_recommender()
    if recommender is None:
        st.error("Failed to load the food recommender system. Please check your setup.")
        return
    with st.sidebar:
        if hasattr(recommender, 'groq_client') and recommender.groq_client:
            st.success("🧠 AI Analysis: Enabled")
            st.markdown("*Powered by Groq AI (Llama3-70B)*")
        else:
            st.error("❌ Groq AI Required")
            st.markdown("*Please configure Groq API key*")
            st.stop() 
    with st.sidebar:
        st.markdown("## 📊 Dataset Overview")
        
        try:
            stats = recommender.get_dataset_stats()
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['total_items']}</div>
                <div class="stat-label">Total Food Items</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['cuisines']}</div>
                <div class="stat-label">Different Cuisines</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{stats['food_types']}</div>
                <div class="stat-label">Food Categories</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Food type distribution pie chart
            st.markdown("### 🍽️ Food Type Distribution")
            food_type_counts = {}
            for food_type in stats['unique_food_types']:
                if 'Meal' in food_type:
                    food_type_counts['Meals'] = food_type_counts.get('Meals', 0) + 1
                elif 'Snack' in food_type or 'Chaat' in food_type:
                    food_type_counts['Snacks'] = food_type_counts.get('Snacks', 0) + 1
                elif 'Dessert' in food_type:
                    food_type_counts['Desserts'] = food_type_counts.get('Desserts', 0) + 1
                elif 'Beverage' in food_type:
                    food_type_counts['Beverages'] = food_type_counts.get('Beverages', 0) + 1
                else:
                    food_type_counts['Others'] = food_type_counts.get('Others', 0) + 1
            fig = go.Figure(data=[go.Pie(
                labels=list(food_type_counts.keys()),
                values=list(food_type_counts.values()),
                hole=0.4,
                marker_colors=['#ff6b35', '#f7931e', '#ffc845', '#ffaa00', '#ff8c42']
            )])
            
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.01,
                    font=dict(size=10)
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
                        
            # Regional distribution
            regional_cuisines = ['South Indian', 'North Indian', 'Punjabi', 'Gujarati', 'Maharashtrian', 'Bengali', 'Rajasthani']
            indian_count = sum(1 for cuisine in stats['unique_cuisines'] if any(region in cuisine for region in regional_cuisines))
            international_count = stats['cuisines'] - indian_count
            
            st.markdown("### 🌍 Cuisine Distribution")
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{indian_count}</div>
                <div class="stat-label">Indian Cuisines</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{international_count}</div>
                <div class="stat-label">International Cuisines</div>
            </div>
            """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    
    st.markdown("### 💭 Tell us about your food mood...")
    user_input = st.text_area(
        "Describe how you're feeling and what kind of food you're craving:",
        placeholder="Example: I'm feeling stressed and want something comforting and sweet to cheer me up...",
        height=100
    )
    num_recommendations = st.selectbox(
        "🔢 Number of recommendations:",
        options=[5, 8, 10, 12],
        index=1
    )
    if st.button("🔍 Get Food Recommendations", type="primary"):
        if user_input.strip():
            with st.spinner("AI is analyzing your mood and finding perfect food matches..."):
                try:
                    time.sleep(1)
                    recommendations = recommender.get_recommendations(
                        user_input, 
                        top_k=num_recommendations
                    )
                    if recommendations:
                        st.markdown("## 🧠 AI-Enhanced Personalized Food Recommendations")
                        st.markdown(f"*Based on: \"{user_input}\"*")
                        for i in range(0, len(recommendations), 2):
                            col1, col2 = st.columns(2)
                            with col1:
                                if i < len(recommendations):
                                    display_food_card(recommendations[i], i+1)
                                    with st.expander(f"🤖 Why {recommendations[i]['food_name']}?"):
                                        explanation = recommender.explain_recommendation_with_groq(
                                            user_input, recommendations[i]['food_name']
                                        )
                                        st.write(explanation)                            
                            with col2:
                                if i+1 < len(recommendations):
                                    display_food_card(recommendations[i+1], i+2)
                                    with st.expander(f"🤖 Why {recommendations[i+1]['food_name']}?"):
                                        explanation = recommender.explain_recommendation_with_groq(
                                            user_input, recommendations[i+1]['food_name']
                                        )
                                        st.write(explanation)
                        st.markdown("---")
                        st.markdown("### 💡 Tips:")
                        st.markdown("""
                        - **Try different descriptions** to discover new dishes
                        - **Be specific** about your mood, taste preferences, or occasion
                        - **Explore different cuisines** by mentioning them in your description
                        """)
                        
                    else:
                        st.warning("No recommendations found. Try describing your mood differently!")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
        else:
            st.warning("⚠️ Please describe your food mood to get recommendations!")
    
    st.markdown("---")
    st.markdown("### 💡 Need inspiration? Try these sample prompts:")
    sample_prompts = [
        "🎉 I'm celebrating and want something festive and sweet",
        "😴 I'm tired after work and need comfort food",
        "🌶️ I'm in the mood for something spicy and tangy",
        "🥗 I want something healthy and refreshing",
        "👨‍👩‍👧‍👦 Planning a family dinner with mild flavors",
        "🏠 Staying home alone and want easy comfort food",
        "🎂 Having a party and need crowd-pleasing snacks",
        "☔ It's raining and I want something warm and cozy"
    ]
    cols = st.columns(2)
    for i, prompt in enumerate(sample_prompts):
        col = cols[i % 2]
        with col:
            if st.button(prompt, key=f"sample_{i}"):
                st.session_state.sample_prompt = prompt.split(' ', 1)[1]  
                st.rerun()
    
    if 'sample_prompt' in st.session_state:
        user_input = st.session_state.sample_prompt
        del st.session_state.sample_prompt
        st.rerun()
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 2rem;">
        <p>🍽️ FoodMood - AI Food Recommender powered by Sentence Transformers & Streamlit</p>
        <p>Discover perfect dishes based on your unique mood and preferences!🎯</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
