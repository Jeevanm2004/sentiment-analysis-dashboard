# 🎭 Sentiment Analysis Dashboard

A modern, AI-powered web application for analyzing text sentiment using state-of-the-art deep learning models from Hugging Face Transformers. Built with Streamlit for an intuitive user experience.

## ✨ Features

- **🤖 AI-Powered Analysis**: Uses pre-trained transformer models from Hugging Face
- **📊 Interactive Visualizations**: Real-time charts showing confidence scores
- **🔄 Multiple Model Support**: Easy switching between different sentiment models
- **📁 File Upload Support**: Analyze text from uploaded files
- **💾 Export Results**: Download analysis results as CSV
- **🎨 Modern UI**: Clean, responsive interface with custom styling
- **⚡ Real-time Processing**: Instant sentiment analysis with confidence scores

## 🚀 Live Demo

[View Live Dashboard](your-streamlit-app-url-here) *(Update this after deployment)*

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Hugging Face Transformers, PyTorch
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Deployment**: Streamlit Community Cloud

## 📋 Requirements

- Python 3.8+
- Internet connection (for downloading models)
- 2GB+ RAM (for model loading)

## 🔧 Local Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-dashboard.git
cd sentiment-analysis-dashboard
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv sentiment_env

# Activate virtual environment
# On Windows:
sentiment_env\Scripts\activate
# On macOS/Linux:
source sentiment_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Open in Browser

The dashboard will automatically open in your default browser at:
```
http://localhost:8501
```

## 🌐 Online Deployment on Streamlit Community Cloud

### Prerequisites
- GitHub account
- Streamlit Community Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

### Step-by-Step Deployment Guide

#### Step 1: Prepare Your Repository

1. **Create a new GitHub repository**:
   - Go to [GitHub](https://github.com) and click "New repository"
   - Name it `sentiment-analysis-dashboard`
   - Make it public
   - Initialize with README (optional, you'll replace it)

2. **Upload your project files**:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-dashboard.git
   cd sentiment-analysis-dashboard
   
   # Copy these files to your repository:
   # - app.py
   # - requirements.txt
   # - README.md
   
   git add .
   git commit -m "Initial commit: Sentiment Analysis Dashboard"
   git push origin main
   ```

#### Step 2: Deploy on Streamlit Community Cloud

1. **Access Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy your app**:
   - Click "New app"
   - Select your repository: `your-username/sentiment-analysis-dashboard`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL (optional): Choose a custom URL or use the auto-generated one

3. **Configure app settings**:
   - Click "Advanced settings" if needed
   - Python version: `3.11` (recommended)
   - Click "Deploy!"

4. **Wait for deployment**:
   - Initial deployment may take 5-10 minutes
   - Models will be downloaded on first run
   - Your app will be available at the provided URL

#### Step 3: Update Your README

After deployment, update the live demo link in your README.md:

```markdown
## 🚀 Live Demo

[View Live Dashboard](https://your-app-name.streamlit.app)
```

### Deployment Tips

- **Model Caching**: Models are automatically cached after first load
- **Resource Limits**: Community Cloud has resource limits; the app is optimized to work within them
- **Updates**: Push changes to GitHub to automatically update your deployed app
- **Custom Domain**: Available with Streamlit Team plans

## 📖 Usage Guide

### Basic Usage

1. **Enter Text**: Type or paste text in the input area
2. **Choose Model**: Select from available AI models in the sidebar
3. **Analyze**: Click "Analyze Sentiment" to get results
4. **View Results**: See sentiment classification, confidence scores, and visualizations

### Advanced Features

#### Model Selection
- **Twitter RoBERTa** (Recommended): Best for social media content
- **Multilingual BERT**: Supports multiple languages
- **DistilBERT SST-2**: Fast and efficient for English text

#### File Upload
- Upload `.txt` files for analysis
- Supports files up to 200MB
- Automatically processes uploaded content

#### Export Results
- Download analysis results as CSV
- Includes all sentiment scores and metadata
- Perfect for further analysis or reporting

### Sample Text Examples

Try these examples to test different sentiments:

**Positive**: "I absolutely love this product! The quality is outstanding and the customer service was amazing."

**Negative**: "I'm very disappointed with this purchase. The quality is poor and it broke after one day."

**Neutral**: "The product arrived on time. It works as expected. The packaging was adequate."

## 🔍 Model Information

### Default Model: cardiffnlp/twitter-roberta-base-sentiment-latest

- **Architecture**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Training Data**: 124M tweets from January 2018 to December 2021
- **Languages**: English
- **Output**: Negative, Neutral, Positive with confidence scores
- **Performance**: State-of-the-art accuracy on social media text

### Alternative Models

The dashboard supports easy model swapping. Other available models include:

- `nlptown/bert-base-multilingual-uncased-sentiment`: Multilingual support
- `distilbert-base-uncased-finetuned-sst-2-english`: Faster processing

## 🎨 Customization

### Adding New Models

To add new Hugging Face models:

1. Update the `available_models` dictionary in `app.py`:
```python
available_models = {
    "your-model-name": "Display Name",
    # ... existing models
}
```

2. Ensure the model outputs sentiment labels compatible with the app

### Styling Customization

Modify the CSS in the `st.markdown` section of `app.py` to change:
- Colors and themes
- Layout and spacing
- Font styles and sizes

### Adding Features

The modular code structure makes it easy to add:
- Batch processing for multiple texts
- Historical analysis tracking
- Additional visualization types
- Integration with external APIs

## 📊 Performance Considerations

### Memory Usage
- Models require 400MB-1GB RAM
- Cached after first load
- Optimized for Streamlit Community Cloud limits

### Processing Speed
- First run: 30-60 seconds (model download)
- Subsequent runs: 1-3 seconds per analysis
- File uploads: Depends on file size

### Resource Optimization
- Models are cached using `@st.cache_resource`
- Efficient text processing with transformers
- Minimal memory footprint for visualizations

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**:
- Ensure internet connectivity for model download
- Try switching to a different model
- Check if Hugging Face services are available

**Slow Performance**:
- Models are cached after first use
- Large texts may take longer to process
- Consider using DistilBERT for faster results

**Memory Issues**:
- Restart the app if memory usage is high
- Use shorter texts for analysis
- Clear browser cache if needed

### Error Messages

**"Error loading model"**: 
- Check internet connection
- Try a different model from the dropdown
- Refresh the page

**"Error analyzing sentiment"**:
- Ensure text input is not empty
- Check for special characters that might cause issues
- Try with simpler text first

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Commit changes: `git commit -m 'Add feature'`
6. Push to branch: `git push origin feature-name`
7. Submit a Pull Request

### Areas for Contribution

- Additional model integrations
- New visualization types
- Performance optimizations
- UI/UX improvements
- Multi-language support
- Batch processing features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for providing pre-trained transformer models
- **Streamlit** for the amazing web app framework
- **Cardiff NLP** for the excellent sentiment analysis models
- **Plotly** for interactive visualizations

## 📞 Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-username/sentiment-analysis-dashboard/issues)
- **Documentation**: This README and inline code comments
- **Community**: Streamlit Community Forum

## 🔄 Updates & Changelog

### Version 1.0.0 (Current)
- Initial release
- Multiple model support
- Interactive dashboard
- File upload capability
- Export functionality
- Streamlit Community Cloud deployment

---

**Built with ❤️ using Streamlit and Hugging Face Transformers**
