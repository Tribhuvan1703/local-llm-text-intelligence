# Contributing to Local LLM Text Intelligence

Thank you for your interest in contributing! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs
- Check if the bug has already been reported in Issues
- Create a new issue with a clear title and description
- Include steps to reproduce, expected behavior, and actual behavior
- Add relevant logs or screenshots

### Suggesting Features
- Open an issue with the `enhancement` label
- Clearly describe the feature and its benefits
- Discuss implementation approach if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   # Run the app
   streamlit run streamlit_app/app.py
   
   # Test imports
   python -c "from src.nlp_orchestrator import NLPOrchestrator; print('OK')"
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add: brief description of changes"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

## Project Structure

```
src/              # Core NLP logic
streamlit_app/    # UI components
tests/            # Unit tests
docs/             # Documentation
```

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/cdac-nlp-project.git
cd cdac-nlp-project

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app/app.py
```

## Questions?

Feel free to open an issue for any questions or clarifications!
