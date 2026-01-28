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
```
