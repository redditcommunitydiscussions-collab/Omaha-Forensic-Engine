# The Omaha Forensic Engine - Project Architecture

## Overview

This document outlines the architecture for "The Omaha Forensic Engine" - an automated forensic analyst that generates investment memos in the style of Warren Buffett and Charlie Munger.

## Core Philosophy

The system is built on three distinct agent types that work together in a chained workflow:
1. **Librarian Agent**: Extracts and organizes data from financial documents
2. **Quant Agent**: Performs financial calculations and validates numbers
3. **Writer Agent**: Generates the narrative analysis using the "Omaha" voice

## Project Structure

```
omaha_forensic_engine/
├── app.py                          # Streamlit UI entry point
├── .env                            # Environment variables (not in git)
├── .env.template                   # Template for environment variables
├── requirements.txt                # Python dependencies
├── setup.sh                        # Virtual environment setup script
│
├── agents/                         # Core agent modules
│   ├── __init__.py
│   ├── librarian/                  # Librarian Agent
│   │   ├── __init__.py
│   │   ├── document_parser.py     # PDF parsing logic
│   │   ├── data_extractor.py      # Extract financial data from text
│   │   └── metadata_extractor.py  # Extract fiscal year, quarter, etc.
│   │
│   ├── quant/                      # Quant Agent
│   │   ├── __init__.py
│   │   ├── financial_calculator.py # Core financial calculations
│   │   ├── metrics_validator.py   # Validate extracted numbers
│   │   └── formulas.py            # Financial formula definitions
│   │
│   └── writer/                     # Writer Agent
│       ├── __init__.py
│       ├── prompt_templates.py    # All 10 prompt templates
│       ├── context_manager.py     # Manages context injection
│       ├── style_enforcer.py      # Enforces "Omaha" voice
│       └── chain_orchestrator.py  # Orchestrates the 10-prompt chain
│
├── core/                           # Core system components
│   ├── __init__.py
│   ├── lens_router.py             # IP Component 1: Sector detection & routing
│   ├── omaha_rubric.py            # IP Component 2: Quality validation
│   ├── quant_sandbox.py           # IP Component 3: Financial calculation library
│   └── config.py                  # Configuration management
│
├── prompts/                        # Prompt templates (JSON/YAML)
│   ├── executive_summary.json
│   ├── supply_chain.json
│   ├── competition.json
│   ├── management.json
│   ├── product_economics.json
│   ├── regulatory.json
│   ├── intangibles.json
│   ├── footnotes.json
│   ├── customers.json
│   └── ytd_review.json
│
├── utils/                          # Utility functions
│   ├── __init__.py
│   ├── file_handlers.py           # File I/O utilities
│   └── logging_config.py          # Logging setup
│
└── tests/                          # Test suite
    ├── __init__.py
    ├── test_librarian.py
    ├── test_quant.py
    └── test_writer.py
```

## Agent Architecture

### 1. Librarian Agent (`agents/librarian/`)

**Purpose**: Extract structured data from unstructured financial documents.

**Responsibilities**:
- Parse PDF files (10-K, 10-Q) using PyPDF2
- Extract text and identify document sections
- Extract key financial metrics (revenue, cash, debt, etc.)
- Identify fiscal year and quarter metadata
- Extract footnotes and specific note references
- Store extracted data in structured format for downstream agents

**Key Components**:
- `document_parser.py`: Handles PDF parsing and text extraction
- `data_extractor.py`: Uses regex/LLM to extract specific financial numbers
- `metadata_extractor.py`: Identifies document type, fiscal period, company name

**Output**: Structured data dictionary with financial metrics, metadata, and raw text sections.

---

### 2. Quant Agent (`agents/quant/`)

**Purpose**: Perform accurate financial calculations and validate extracted numbers.

**Responsibilities**:
- Calculate "Omaha Math" metrics (Real Net Cash, SBC ratios, etc.)
- Validate that extracted numbers are consistent across document
- Perform trend analysis (YoY, QoQ comparisons)
- Calculate operating leverage, free cash flow, etc.
- Flag inconsistencies or missing data

**Key Components**:
- `financial_calculator.py`: Core calculation engine (the "Quant Sandbox" IP)
- `metrics_validator.py`: Cross-validates numbers from different sections
- `formulas.py`: Centralized definitions of all financial formulas

**Key Calculations** (from PRD):
- Real Net Cash = Cash + Equivalents - Total Debt - Operating Lease Liabilities
- SBC as % of Net Income
- Free Cash Flow = Operating Cash Flow - CapEx
- Operating Leverage = Profit Growth / Revenue Growth

**Output**: Validated financial metrics dictionary with calculated values and flags.

---

### 3. Writer Agent (`agents/writer/`)

**Purpose**: Generate the investment memo using the "Omaha" voice and chained prompts.

**Responsibilities**:
- Execute a sequential, non-parallel chain (see Context Injection requirement)
- Inject accumulated context from previous steps into each step
- Enforce the "Buffett & Munger" voice (cynical, metaphor-rich)
- Generate each section according to the strict format rules
- Validate output against the "Omaha Rubric" (quality gate)

**Key Components**:
- `prompt_templates.py`: All 10 prompt templates from the guide
- `context_manager.py`: Manages context injection between prompts
- `style_enforcer.py`: Validates tone and metaphor usage
- `chain_orchestrator.py`: Orchestrates the sequential execution

**Sequential Chain Flow (Backend)**:
1. Executive Summary → Store as `context_step_1`
2. Financial Analysis → System prompt receives `context_step_1`
3. Risk & Competition Analysis → System prompt receives accumulated context (Steps 1–2)
4. Management & Governance Analysis → System prompt receives accumulated context (Steps 1–3)
5. YTD Review & Final Verdict → System prompt receives accumulated context (Steps 1–4)

**Note**: The prompt library still preserves the 10 detailed templates from the Prompt Logic. The backend may group these into 5 execution steps while maintaining a strict sequential chain and full context injection.

**Output**: Complete investment memo in markdown format.

---

## Core IP Components

### 1. Lens Router (`core/lens_router.py`)

**Purpose**: Detect company sector and apply sector-specific analysis rules.

**Logic**:
- Analyzes company name, industry codes, or document content
- Routes to appropriate "lens" (Tech, Finance, REIT, etc.)
- Injects sector-specific rules (e.g., "SBC_is_Expense = True" for Tech)
- **Note**: MVP focuses on Tech sector only (per PRD limitations)

**Future Enhancement**: Expand to support multiple sectors with different analysis frameworks.

---

### 2. Omaha Rubric (`core/omaha_rubric.py`)

**Purpose**: Quality gate that validates output meets "Omaha" standards.

**Validation Rules**:
- Must contain specific metaphors ("Toll Bridge", "Red Queen", etc.)
- Must perform required calculations (Real Net Cash, SBC %, etc.)
- Must use cynical tone (reject generic phrases)
- Must cite specific page numbers or note references
- Must distinguish between "Buffett" and "Munger" voices

**Implementation**: JSON schema with validation logic that can reject drafts and request regeneration.

---

### 3. Quant Sandbox (`core/quant_sandbox.py`)

**Purpose**: Centralized Python library for financial calculations (ensures consistency).

**Key Functions**:
- `calculate_real_net_cash(cash, debt, leases)`
- `calculate_sbc_ratio(sbc_expense, net_income)`
- `calculate_free_cash_flow(ocf, capex)`
- `calculate_operating_leverage(profit_growth, revenue_growth)`

**Why IP**: These calculations are standardized and don't rely on LLM math (which can be inconsistent).

---

## Workflow: The "Chained" Method

```
User Input (10-K PDF)
    ↓
[Librarian Agent]
    ├─ Parse PDF
    ├─ Extract Financial Data
    └─ Extract Metadata
    ↓
[Quant Agent]
    ├─ Validate Numbers
    ├─ Calculate Metrics
    └─ Flag Inconsistencies
    ↓
[Writer Agent - Step 1]
    ├─ Generate Executive Summary
    └─ Store as context_step_1
    ↓
[Writer Agent - Step 2]
    ├─ Financial Analysis
    └─ Inject context_step_1 into system prompt
    ↓
[Writer Agent - Step 3]
    ├─ Risk & Competition Analysis
    └─ Inject accumulated context (Steps 1–2)
    ↓
[Writer Agent - Step 4]
    ├─ Management & Governance Analysis
    └─ Inject accumulated context (Steps 1–3)
    ↓
[Writer Agent - Step 5]
    ├─ YTD Review & Final Verdict
    └─ Inject accumulated context (Steps 1–4)
    ↓
Final Investment Memo (Markdown)
```

## Technology Stack

- **Language**: Python 3.11
- **UI**: Streamlit (simple, text-dense interface)
- **AI Framework**: LangChain
- **LLM**: Google Gemini 1.5 Pro
- **PDF Parsing**: PyPDF2
- **Configuration**: python-dotenv

## Key Design Decisions

1. **Separation of Concerns**: Three distinct agents prevent mixing data extraction, calculation, and narrative generation.

2. **Context Injection**: Each step receives the full accumulated history of prior steps (no parallel execution).

3. **Deterministic Math**: Financial calculations are done in Python, not by LLM, ensuring accuracy.

4. **Quality Gates**: Omaha Rubric acts as a validation layer before final output.

5. **Modular Prompts**: Each of the 10 prompts is a separate template, making them easy to update.

## Next Steps (Implementation Order)

1. ✅ **Scaffold Project** (Current Step)
2. **Implement Librarian Agent**: PDF parsing and data extraction
3. **Implement Quant Agent**: Financial calculation library
4. **Implement Writer Agent**: Prompt templates and chain orchestration
5. **Implement Core IP Components**: Lens Router, Omaha Rubric, Quant Sandbox
6. **Build Streamlit UI**: Simple interface for file upload and report display
7. **Integration & Testing**: End-to-end testing with sample 10-K

## Notes

- MVP scope: Tech sector only (per PRD)
- Latency acceptable: 60-90 seconds for deep analysis
- Output format: Markdown text (can be exported to PDF later)
