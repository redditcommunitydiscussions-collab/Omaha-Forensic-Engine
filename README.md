# The Omaha Forensic Engine

An automated forensic analyst that generates investment memos in the style of Warren Buffett and Charlie Munger.

## Quick Start

1. **Set up the environment:**
   ```bash
   ./setup.sh
   ```

2. **Configure your API key:**
   - Copy `.env.template` to `.env` (if not done automatically)
   - Add your `GOOGLE_API_KEY` to `.env`
   - Get your API key from: https://makersuite.google.com/app/apikey

3. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Architecture

See `project_structure.md` for detailed architecture documentation.

The system consists of three core agents:
- **Librarian Agent**: Extracts data from financial documents
- **Quant Agent**: Performs financial calculations
- **Writer Agent**: Generates the investment memo using chained prompts

## Requirements

- Python 3.11
- Google Gemini API key

## Secrets (Streamlit Cloud)

Set these in Streamlit Cloud → App Settings → Secrets:

```
GOOGLE_API_KEY="your_key_here"
```

Optional tuning:

```
GEMINI_MODEL="gemini-1.5-flash"
RETRY_ATTEMPTS="4"
RETRY_BACKOFF_SECONDS="30"
SLEEP_BETWEEN_STEPS="10"
ENABLE_REDUCE="true"
MAX_SECTION_CHARS="120000"
```

## Project Status

🚧 **In Development** - Currently in scaffolding phase.
