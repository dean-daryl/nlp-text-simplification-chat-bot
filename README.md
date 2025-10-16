# 🤖 NLP Text Simplification Bot

A Streamlit-powered chatbot that simplifies complex text using Microsoft's Phi-3-mini-4k-instruct model. Transform academic papers, technical documents, and complex content into easy-to-understand language.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **AI-Powered Simplification**: Uses Microsoft Phi-3-mini-4k-instruct for high-quality text simplification
- **Interactive Chat Interface**: Streamlit-based conversational UI for natural interaction
- **Real-time Processing**: Instant text simplification with processing time metrics
- **Word Count Analytics**: Track original vs simplified word counts
- **Quick Examples**: Pre-loaded examples for testing
- **GPU Acceleration**: Automatic GPU detection and utilization when available
- **Fallback Mode**: Mock mode for deployment testing and demo purposes

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended (for model loading)
- GPU optional but recommended for faster inference

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nlp-text-simplification
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit torch transformers
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If port 8501 is busy, try: `streamlit run app.py --server.port 8502`

## 🎯 Usage

### Basic Text Simplification

1. **Start the app** and wait for the model to load (first run takes longer)
2. **Type or paste complex text** into the chat input
3. **Receive simplified version** with word count statistics
4. **Try the examples** in the sidebar for quick testing

### Example Transformations

**Input:**
> "The implementation of sophisticated algorithms requires extensive computational resources and meticulous optimization procedures."

**Output:**
> "Using advanced computer programs needs a lot of computing power and careful fine-tuning steps."

### Chat Interface Features

- **Conversation History**: All simplifications are saved in the chat
- **Quick Examples**: Click sidebar examples for instant testing
- **Clear Chat**: Reset conversation with the clear button
- **Real-time Metrics**: See processing time and word count changes

## 🛠️ Technical Details

### Model Architecture

- **Base Model**: Microsoft Phi-3-mini-4k-instruct
- **Model Size**: ~7GB (downloaded on first run)
- **Context Length**: 4,096 tokens
- **Architecture**: Transformer-based causal language model

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| Storage | 10GB free | 20GB+ free |
| CPU | Multi-core | Intel i5/AMD Ryzen 5+ |
| GPU | None | NVIDIA RTX series |

### Performance

- **CPU Mode**: 5-15 seconds per simplification
- **GPU Mode**: 1-3 seconds per simplification
- **First Run**: Additional time for model download (~10-15 minutes)
- **Subsequent Runs**: Model cached locally for faster startup

## 📁 Project Structure

```
nlp-text-simplification/
├── app.py                    # Main Streamlit application
├── history_bot.py           # Alternative bot implementation
├── phi-3-mini-4k-instruct.Q4_K_M.gguf  # GGUF model file
├── Kaggle Archive/          # Training datasets
├── training_ready/          # Processed training data
└── README.md               # This file
```

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Set custom model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Optional: Force CPU usage (disable GPU)
export CUDA_VISIBLE_DEVICES=""
```

### Model Customization

To use a different model, modify `app.py`:

```python
# Change this line in load_model() function
model_name = "your-preferred-model-name"
```

## 🔧 Troubleshooting

### Common Issues

**Model Loading Errors**
- Ensure sufficient RAM (8GB minimum)
- Check internet connection for model download
- Verify transformers library version: `pip install transformers>=4.35.0`

**Port Already in Use**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

**Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade torch transformers streamlit
```

**Out of Memory (OOM)**
- Close other applications
- Use CPU mode: set `CUDA_VISIBLE_DEVICES=""`
- Consider using a smaller model

### Performance Optimization

1. **Use GPU**: Install CUDA-compatible PyTorch
2. **Increase RAM**: Close unnecessary applications
3. **SSD Storage**: Store model cache on SSD for faster loading

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

### Development Setup

```bash
# Install development dependencies
pip install streamlit torch transformers black pytest

# Run tests (if available)
python -m pytest

# Format code
black app.py
```

## 📊 Model Information

### Phi-3-Mini-4K-Instruct Details

- **Developer**: Microsoft
- **Release Date**: 2024
- **Parameters**: ~3.8B
- **Training Data**: High-quality web data, academic papers, code
- **Capabilities**: Text generation, summarization, simplification
- **Languages**: Primarily English, some multilingual support

### Prompt Engineering

The model uses a specific chat format:
```
<|user|>
Please simplify this text: [YOUR TEXT]
<|end|>
<|assistant|>
[SIMPLIFIED TEXT]
```

## 🔮 Future Enhancements

- [ ] **Multiple Models**: Support for different simplification models
- [ ] **Batch Processing**: Simplify multiple texts at once
- [ ] **Reading Level**: Target specific reading levels (grade 3, 5, 8, etc.)
- [ ] **Export Options**: Save results as PDF, Word, or text files
- [ ] **API Integration**: RESTful API for programmatic access
- [ ] **Custom Training**: Fine-tune models on domain-specific data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft** for the Phi-3-mini-4k-instruct model
- **Streamlit** for the amazing web app framework
- **Hugging Face** for the transformers library
- **PyTorch** for the deep learning framework

## 📞 Support

- **Issues**: Report bugs on GitHub Issues
- **Questions**: Start a GitHub Discussion
- **Documentation**: Check this README and code comments

---

**Made with ❤️ for accessible communication**

*Simplifying complex text, one sentence at a time.*