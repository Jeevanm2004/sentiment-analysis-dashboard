import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .sentiment-neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_model(model_name):
    """Load and cache the sentiment analysis model"""
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            return_all_scores=True
        )
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def analyze_sentiment(text, classifier):
    """Analyze sentiment of input text"""
    if not text or not text.strip():
        return None
    
    try:
        # Get predictions for all labels
        results = classifier(text)
        
        # Debug: Print raw results to understand structure
        if isinstance(results, list) and len(results) > 0:
            raw_results = results[0] if isinstance(results[0], list) else results
        else:
            raw_results = results
        
        # Convert to a more readable format
        sentiment_scores = {}
        
        for result in raw_results:
            label = result['label'].upper()
            score = float(result['score'])
            
            # Map different label formats to standard format
            if label in ['LABEL_2', 'POSITIVE', 'POS']:
                sentiment_scores['Positive'] = score
            elif label in ['LABEL_0', 'NEGATIVE', 'NEG']:
                sentiment_scores['Negative'] = score
            elif label in ['LABEL_1', 'NEUTRAL', 'NEU']:
                sentiment_scores['Neutral'] = score
            else:
                # For unknown labels, use the label as-is
                sentiment_scores[label.title()] = score
        
        # Ensure we have at least some scores
        if not sentiment_scores:
            st.error("No valid sentiment scores found from model output")
            return None
        
        # Get the dominant sentiment
        dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
        
        return {
            'dominant_sentiment': dominant_sentiment[0],
            'confidence': dominant_sentiment[1],
            'all_scores': sentiment_scores
        }
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        # Debug information
        st.write("Debug info - Raw classifier output:", results if 'results' in locals() else "No results")
        return None

def create_sentiment_chart(sentiment_data):
    """Create a bar chart showing sentiment confidence scores"""
    if not sentiment_data or not sentiment_data.get('all_scores'):
        return None
    
    scores = sentiment_data['all_scores']
    
    # Create color mapping
    colors = {'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#ffc107'}
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(scores.keys()),
            y=list(scores.values()),
            marker_color=[colors.get(sentiment, '#6c757d') for sentiment in scores.keys()],
            text=[f'{score:.2%}' for score in scores.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Confidence Scores",
        xaxis_title="Sentiment",
        yaxis_title="Confidence Score",
        yaxis=dict(tickformat='.1%', range=[0, 1]),
        height=400,
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def display_sentiment_result(sentiment_data, text):
    """Display sentiment analysis results with styling"""
    if not sentiment_data:
        return
    
    dominant = sentiment_data['dominant_sentiment']
    confidence = sentiment_data['confidence']
    
    # Choose CSS class based on sentiment
    css_class = f"sentiment-{dominant.lower()}"
    
    # Display main result
    st.markdown(f"""
    <div class="{css_class}">
        <h3>🎭 Sentiment Analysis Result</h3>
        <p><strong>Text analyzed:</strong> "{text[:100]}{'...' if len(text) > 100 else ''}"</p>
        <p><strong>Dominant Sentiment:</strong> {dominant}</p>
        <p><strong>Confidence:</strong> {confidence:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Analyze the sentiment of any text using state-of-the-art AI models from Hugging Face!**
    
    This dashboard uses deep learning to classify text sentiment as Positive, Negative, or Neutral with confidence scores.
    """)
    
    # Sidebar for model selection and settings
    st.sidebar.header("🛠️ Model Settings")
    
    # Model selection
    available_models = {
        "cardiffnlp/twitter-roberta-base-sentiment-latest": "Twitter RoBERTa (Recommended)",
        "nlptown/bert-base-multilingual-uncased-sentiment": "Multilingual BERT",
        "distilbert-base-uncased-finetuned-sst-2-english": "DistilBERT SST-2"
    }
    
    selected_model = st.sidebar.selectbox(
        "Choose Sentiment Model",
        options=list(available_models.keys()),
        format_func=lambda x: available_models[x],
        help="Different models may perform better on different types of text"
    )
    
    # Load model
    with st.spinner("Loading AI model... This may take a moment on first run."):
        classifier = load_sentiment_model(selected_model)
    
    if classifier is None:
        st.error("Failed to load the sentiment analysis model. Please try again or select a different model.")
        return
    
    st.sidebar.success("✅ Model loaded successfully!")
    
    # Model info
    st.sidebar.markdown(f"""
    **Current Model:** {available_models[selected_model]}
    
    **Model Details:**
    - Architecture: Transformer-based
    - Training: Social media & review data
    - Languages: English (primarily)
    """)
    
    # Main input area
    st.header("📝 Text Input")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Type/Paste Text", "Upload Text File"],
        horizontal=True
    )
    
    text_to_analyze = ""
    
    if input_method == "Type/Paste Text":
        text_to_analyze = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here... (e.g., product reviews, social media posts, feedback, etc.)",
            help="Enter any text you'd like to analyze for sentiment"
        )
    
    else:  # Upload file
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=['txt'],
            help="Upload a .txt file containing the text to analyze"
        )
        
        if uploaded_file is not None:
            text_to_analyze = str(uploaded_file.read(), "utf-8")
            st.text_area("Uploaded text:", value=text_to_analyze, height=150, disabled=True)
    
    # Sample texts for quick testing
    with st.expander("📋 Try Sample Texts"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("😊 Positive Sample"):
                text_to_analyze = "I absolutely love this product! The quality is outstanding and the customer service was amazing. Highly recommended!"
                st.rerun()
        
        with col2:
            if st.button("😐 Neutral Sample"):
                text_to_analyze = "The product arrived on time. It works as expected. The packaging was adequate."
                st.rerun()
        
        with col3:
            if st.button("😞 Negative Sample"):
                text_to_analyze = "I'm very disappointed with this purchase. The quality is poor and it broke after just one day. Waste of money!"
                st.rerun()
    
    # Analyze button
    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if text_to_analyze and text_to_analyze.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = analyze_sentiment(text_to_analyze, classifier)
                
                if sentiment_result:
                    # Store results in session state for persistence
                    st.session_state.last_result = sentiment_result
                    st.session_state.last_text = text_to_analyze
                    st.session_state.analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.rerun()
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Display results if available
    if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
        st.header("📊 Analysis Results")
        
        # Display main result
        display_sentiment_result(st.session_state.last_result, st.session_state.last_text)
        
        # Create two columns for metrics and chart
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📈 Detailed Scores")
            scores = st.session_state.last_result['all_scores']
            
            for sentiment, score in scores.items():
                # Create color-coded metric
                if sentiment == 'Positive':
                    st.success(f"**{sentiment}**: {score:.2%}")
                elif sentiment == 'Negative':
                    st.error(f"**{sentiment}**: {score:.2%}")
                else:
                    st.warning(f"**{sentiment}**: {score:.2%}")
        
        with col2:
            # Display chart
            chart = create_sentiment_chart(st.session_state.last_result)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
        
        # Additional information
        st.info(f"📅 Analysis completed at: {st.session_state.analysis_time}")
        
        # Export results
        with st.expander("💾 Export Results"):
            result_data = {
                'text': st.session_state.last_text,
                'dominant_sentiment': st.session_state.last_result['dominant_sentiment'],
                'confidence': st.session_state.last_result['confidence'],
                'positive_score': st.session_state.last_result['all_scores'].get('Positive', 0),
                'negative_score': st.session_state.last_result['all_scores'].get('Negative', 0),
                'neutral_score': st.session_state.last_result['all_scores'].get('Neutral', 0),
                'analysis_time': st.session_state.analysis_time
            }
            
            st.json(result_data)
            
            # Download as CSV
            df = pd.DataFrame([result_data])
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this Dashboard:**
    - Built with Streamlit and Hugging Face Transformers
    - Uses state-of-the-art deep learning models for sentiment analysis
    - Real-time analysis with confidence scores
    - Export results for further analysis
    
    **Tips for better results:**
    - Longer texts generally provide more accurate sentiment analysis
    - The model works best with English text
    - Try different models for various types of content
    """)

if __name__ == "__main__":
    main()
