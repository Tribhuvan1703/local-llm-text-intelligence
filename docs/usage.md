# Usage Guide

## NLP Orchestrator API Reference

### Class: `NLPOrchestrator`

Main class for performing NLP analysis on text.

#### Initialization

```python
from src.nlp_orchestrator import NLPOrchestrator

nlp = NLPOrchestrator(model_id="llama3.1:8b-instruct-q4_K_M")
```

**Parameters:**
- `model_id` (str): Ollama model identifier. Default: `"llama3.1:8b-instruct-q4_K_M"`

#### Methods

##### `process(text: str) -> dict`

Main method to process text through the complete NLP pipeline.

**Parameters:**
- `text` (str): Input text to analyze

**Returns:**
- `dict`: Analysis results containing:
  - `Cleaned_content`: Normalized text
  - `lang_det`: Detected language
  - `lang_det_confidence_score`: Language detection confidence
  - `Translation`: English translation
  - `Translation_confidence_score`: Translation confidence
  - `domain_ident`: Domain classification
  - `domain_ident_confidence_score`: Domain confidence
  - `sentiment`: Sentiment label
  - `sentiment_confidence_score`: Sentiment confidence
  - `NER`: Named entities (Person, Location, Organisation, Event, Product)
  - `NER_confidence_score`: NER confidence
  - `Event_calender`: Extracted events/dates
  - `Event_calender_confidence_score`: Event extraction confidence
  - `Country_iden`: Identified country
  - `Country_iden_confidence_score`: Country identification confidence
  - `Summary`: Text summary
  - `Summary_confidence_score`: Summary confidence
  - `Fact_checker`: Relevant topics
  - `Fact_checker_relevance_rating`: Relevance rating
  - `Fact_checker_confidence_score`: Fact checker confidence

**Example:**

```python
text = "मुंबई में आज बारिश हुई।"
result = nlp.process(text)

print(f"Language: {result['lang_det']}")
print(f"Translation: {result['Translation']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Summary: {result['Summary']}")
```

##### `detectlanguage(text: str) -> dict`

Detect the language of input text.

**Returns:**
- `dict`: `{"primary_lang": str, "confidence": float, "method": str}`

##### `translate(text: str, lang: str) -> tuple`

Translate text to English.

**Returns:**
- `tuple`: `(translated_text, confidence)`

##### `analyze(text: str) -> AnalysisResult`

Perform comprehensive analysis (domain, sentiment, NER, etc.).

**Returns:**
- `AnalysisResult`: Pydantic model with all analysis fields

## Streamlit App Usage

### File Upload

The app supports three file types:

1. **TXT files**: Plain text files
2. **CSV files**: Tabular data (all cells concatenated)
3. **PDF files**: Extracted text from PDF documents

### Analysis Workflow

1. Upload a file or paste text in the text area
2. Click "🚀 Analyze Text"
3. View results in three tabs:
   - **📊 Dashboard**: Visual metrics and entities
   - **📝 Clean & Translate**: Original and translated text
   - **🧾 Raw JSON**: Complete JSON response (downloadable)

### Interpreting Results

#### Language Detection
- Shows detected language with confidence percentage
- Supports: Hindi, Marathi, English, Bengali, Punjabi, Tamil, Telugu, etc.

#### Domain Classification
- **Politics**: Elections, political parties, government
- **Crime**: Criminal activities
- **Law and Order**: Riots, protests, public safety
- **General**: Technology, business, sports, entertainment

#### Sentiment Analysis
- **Positive**: Favorable, optimistic content
- **Negative**: Unfavorable, pessimistic content
- **Neutral**: Objective, balanced content
- **Anti-National**: Content flagged as anti-national

#### Named Entities
Entities are displayed as colored badges grouped by type:
- **Person**: Individual names
- **Location**: Geographic locations
- **Organisation**: Companies, agencies, government bodies
- **Event**: Named events
- **Product**: Product names and models

## Advanced Usage

### Custom Model Configuration

```python
# Use a different Ollama model
nlp = NLPOrchestrator(model_id="llama3.2:latest")
```

### Batch Processing

```python
texts = [
    "Text 1...",
    "Text 2...",
    "Text 3..."
]

results = [nlp.process(text) for text in texts]
```

### Caching

The orchestrator includes LRU caching for LLM calls to improve performance:

```python
# Clear cache if needed
nlp._llm_cache.clear()
```

### Error Handling

```python
try:
    result = nlp.process(text)
except Exception as e:
    print(f"Error processing text: {e}")
```

## Performance Tips

1. **Text Length**: Keep input under 10,000 characters for optimal performance
2. **Batch Processing**: Process multiple texts sequentially to leverage caching
3. **Model Selection**: Smaller quantized models (q4_K_M) are faster but less accurate
4. **Ollama Configuration**: Ensure Ollama has sufficient RAM allocated

## API Integration

To integrate the NLP orchestrator into your own application:

```python
from src.nlp_orchestrator import NLPOrchestrator
import json

# Initialize once (reuse across requests)
nlp = NLPOrchestrator()

def analyze_text_api(text: str) -> str:
    """API endpoint for text analysis"""
    result = nlp.process(text)
    return json.dumps(result, ensure_ascii=False)

# Example usage
response = analyze_text_api("Your text here")
print(response)
```

## Troubleshooting

### Common Issues

**Slow Processing**
- Reduce text length
- Use a smaller/faster model
- Ensure Ollama is running locally (not remote)

**Incorrect Language Detection**
- Ensure text has sufficient length (>20 characters)
- Check for mixed-language content
- Verify Unicode encoding

**Low Confidence Scores**
- Ambiguous or unclear text
- Mixed domains or sentiments
- Short text snippets

**Import Errors**
- Verify virtual environment is activated
- Check that you're running from project root
- Ensure all dependencies are installed
