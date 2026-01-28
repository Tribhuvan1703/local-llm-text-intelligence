# Local LLM Text Intelligence

A powerful NLP text analysis application using local Llama 3.1 models via Ollama. Performs multilingual language detection, translation, sentiment analysis, named entity recognition, and domain classification.

## Features

- 🌍 **Multilingual Support**: Detects and processes Hindi, Marathi, Bengali, Punjabi, Tamil, Telugu, and more
- 🔄 **Translation**: Automatic translation to English for non-English text
- 🎯 **Domain Classification**: Categorizes text into Politics, Crime, Military, Law & Order, etc.
- 😊 **Sentiment Analysis**: Determines sentiment (Positive, Negative, Neutral, Anti-National)
- 🏷️ **Named Entity Recognition (NER)**: Extracts Persons, Locations, Organizations, Events, Products
- 📅 **Event Calendar**: Identifies dates and events mentioned in text
- 🌐 **Country Identification**: Detects country references
- 📊 **Fact Checker**: Analyzes relevance and provides confidence scores
- 📝 **Summarization**: Generates concise summaries

## Project Structure

```
d:/cdac project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment template
├── .env                         # Environment variables
├── run_app.bat                  # Quick launcher script
│
├── src/                         # Core NLP logic
│   ├── __init__.py
│   ├── nlp_orchestrator.py      # Main NLP pipeline
│   └── utils/
│       └── __init__.py
│
├── streamlit_app/               # Web UI
│   ├── __init__.py
│   ├── app.py                   # Streamlit application
│   └── components/
│       └── __init__.py
│
├── tests/                       # Unit tests
│   └── __init__.py
│
└── docs/                        # Documentation
    └── usage.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- Llama 3.1 model pulled in Ollama

### Setup

1. **Clone or navigate to the project directory**:
   ```bash
   cd "d:\cdac project"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   ```bash
   # Windows
   .\venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Pull the Llama model in Ollama**:
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```

6. **Configure environment variables** (optional):
   ```bash
   copy .env.example .env
   # Edit .env if needed
   ```

## Usage

### Running the Streamlit Web App

**Option 1: Use the launcher script**
```bash
.\run_app.bat
```

**Option 2: Run manually**
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run Streamlit
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the NLP Orchestrator Programmatically

```python
from src.nlp_orchestrator import NLPOrchestrator

# Initialize
nlp = NLPOrchestrator(model_id="llama3.1:8b-instruct-q4_K_M")

# Process text
text = "Your text here..."
result = nlp.process(text)

# Access results
print(result["lang_det"])           # Detected language
print(result["sentiment"])          # Sentiment
print(result["Summary"])            # Summary
print(result["NER"])                # Named entities
```

## Features in Detail

### Language Detection
- Unicode script detection for Indic languages
- Romanized Hindi detection
- Marathi-specific markers
- Fallback to LLM-based detection

### Translation
- Automatic translation to English
- Preserves proper nouns
- Confidence scoring

### Domain Classification
Classifies text into:
- Politics
- Crime
- Military
- Terrorism
- Law and Order
- Narcotics
- General (default for technology, business, sports, etc.)

### Named Entity Recognition
Extracts:
- **Person**: Names of individuals
- **Location**: Places, cities, countries
- **Organisation**: Companies, government bodies, agencies
- **Event**: Named events, conferences
- **Product**: Product names, models

## Configuration

Edit `.env` file to configure:
```bash
# Add any configuration variables here
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Adding New Features
1. Add core logic to `src/nlp_orchestrator.py` or create new modules in `src/utils/`
2. Update UI components in `streamlit_app/`
3. Add tests in `tests/`
4. Update documentation

## Troubleshooting

**Issue**: "Could not import NLPOrchestrator"
- **Solution**: Ensure you're running from the project root and virtual environment is activated

**Issue**: Ollama connection error
- **Solution**: Make sure Ollama is running (`ollama serve`)

**Issue**: Model not found
- **Solution**: Pull the model with `ollama pull llama3.1:8b-instruct-q4_K_M`

## License

This project is for educational and research purposes.

## Contributors

CDAC Project Team
