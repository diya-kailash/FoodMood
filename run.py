import subprocess
import sys
import os

def run_app():
    try:
        print("🍽️ Starting FoodMood AI Recommender System...")
        print("📱 The web app will open in your default browser")
        print("🔗 URL: http://localhost:8501")
        print("\n" + "="*50)
        print("💡 To stop the app: Press Ctrl+C in this terminal")
        print("="*50 + "\n")
        
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "dark"
        ])
        
    except KeyboardInterrupt:
        print("\n\n🛑 FoodMood AI stopped by user")
        print("Thanks for using FoodMood AI! 🍽️")
    except Exception as e:
        print(f"❌ Error running the app: {str(e)}")
        print("\n💡 Make sure you have installed all dependencies:")
        print("   pip install streamlit sentence-transformers pandas numpy scikit-learn")

if __name__ == "__main__":
    run_app()
