# Real Estate Multi-Agent System

A sophisticated multi-agent system built with LangGraph for real estate asset management and financial analysis.

## Features

- **Intent Classification**: Automatically detects query type (P&L, property, tenant, general)
- **Entity Extraction**: Identifies properties, tenants, and timeframes from natural language
- **Fuzzy Matching**: Handles typos and variations in property/tenant names
- **Financial Analysis**: Calculates P&L, revenue, expenses with detailed breakdowns
- **Property Comparison**: Compare performance across multiple properties
- **Tenant Analytics**: Revenue analysis and tenant rankings
- **Error Handling**: Graceful handling of missing data and ambiguous queries

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Orchestration | LangGraph | 0.2.0+ |
| LLM | OpenAI GPT-4o-mini | Latest |
| Data Processing | Pandas + PyArrow | Latest |
| UI | Streamlit | 1.30+ |
| Fuzzy Matching | rapidfuzz | 3.0+ |
| Python | 3.11+ | Required |

## Project Structure

```
real_estate_agent/
├── app.py                 # Streamlit entry point
├── graph.py               # LangGraph workflow
├── state.py               # State TypedDict
├── config.py              # Settings, constants
├── agents/
│   ├── __init__.py
│   ├── router.py          # Intent + entity extraction
│   ├── pnl_agent.py       # P&L queries
│   ├── property_agent.py  # Property queries
│   ├── tenant_agent.py    # Tenant queries
│   ├── general_agent.py   # General knowledge queries
│   └── fallback.py        # Error/unclear handling
├── data/
│   ├── __init__.py
│   ├── loader.py          # Parquet loader (singleton)
│   └── queries.py         # All query functions
├── prompts/
│   └── templates.py       # LLM prompts as constants
├── requirements.txt
├── cortex.parquet         # Dataset (3,924 rows)
└── README.md
```

## Installation

### 1. Clone or Download

```bash
cd EstateAgent
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

### 5. Verify Data File

Ensure `cortex.parquet` is in the project root directory.

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Programmatic Usage

```python
from graph import run_query

# Run a query
result = run_query("What's the total P&L for 2024?")
print(result["response"])
```

## Example Queries

### P&L Analysis
- "What's the total P&L for 2024?"
- "Show me the expense breakdown"
- "P&L for Q1 2024"
- "What are our biggest expenses?"

### Property Queries
- "How is Building 180 performing?"
- "Compare Building 17 and Building 120"
- "Show Building 140 revenue for 2024"
- "bldg 180" (fuzzy match to "Building 180")

### Tenant Analysis
- "Revenue from Tenant 8"
- "Top 5 tenants by revenue"
- "How much does Tenant 12 pay in 2024?"

### General Knowledge
- "How many tenants do I have?"
- "List all properties"
- "Give me an overview of the portfolio"

### Fallback Handling
- "Tell me about Building 999" (Graceful "not found")
- "What's the weather?" (Helpful suggestions)

## Architecture

### LangGraph Workflow

```
START -> router -> [conditional routing] -> agent -> END
                         |
         +---------------+---------------+---------------+
         |               |               |               |
    pnl_agent    property_agent    tenant_agent    general_agent
                         |
                    fallback
```

### Intent Types

| Intent | Description | Example |
|--------|-------------|---------|
| `pnl_summary` | Total P&L for period | "Total P&L for 2024" |
| `pnl_breakdown` | Expense/revenue breakdown | "Show expense breakdown" |
| `property_details` | Property info & comparisons | "How is Building 180 doing?" or "Compare Building 17 vs 120" |
| `tenant_details` | Tenant revenue | "Revenue from Tenant 8" |
| `tenant_ranking` | Top tenants | "Top 5 tenants" |
| `general_knowledge` | General portfolio queries | "How many tenants?" |
| `fallback` | Unclear or off-topic | Anything else |

### State Schema

```python
class AgentState(TypedDict):
    user_query: str           # Original question
    intent: str              # Classified intent
    entities: dict           # Extracted entities
    data: dict              # Query results
    response: str           # Final answer
    error: Optional[str]    # Error message if any
```

## Design Decisions

### 1. Singleton Data Loader
- **Why**: Load parquet file once, cache in memory
- **Benefit**: Fast queries, no repeated I/O
- **Trade-off**: Memory usage for large datasets

### 2. Fuzzy Matching
- **Why**: Handle user typos and variations
- **Implementation**: rapidfuzz with 80% threshold
- **Example**: "bldg 180" matches "Building 180"

### 3. Separate Agent Modules
- **Why**: Single responsibility, easy testing
- **Benefit**: Can develop/test agents independently
- **Pattern**: Each agent is a pure function returning state updates

### 4. TypedDict State
- **Why**: LangGraph best practice
- **Benefit**: Type hints, clear contracts
- **Trade-off**: Less flexible than dict

### 5. Template-Based Prompts
- **Why**: Centralized prompt management
- **Benefit**: Easy to update, version control
- **Location**: `prompts/templates.py`

## Code Quality Standards

### Type Hints
All functions use type hints:
```python
def get_total_pnl(year: Optional[str] = None) -> float:
    ...
```

### Docstrings
Google-style docstrings for all public functions:
```python
def fuzzy_match_property(query: str) -> Optional[str]:
    """Find the best matching property name using fuzzy matching.

    Args:
        query: Property name query (can be partial or misspelled).

    Returns:
        Best matching property name, or None if no good match found.
    """
```

### Error Handling
- No crashes: all errors return user-friendly messages
- Logging: all errors logged for debugging
- Fuzzy matching before "not found" errors

### Modern Patterns
- No deprecated imports
- Enums for constants
- DRY principle
- Small, focused functions

## Dataset Schema

File: `cortex.parquet` (3,924 rows)

| Column | Type | Description |
|--------|------|-------------|
| entity_name | str | Company name ("PropCo") |
| property_name | str | Building 120, 140, 160, 17, 180 |
| tenant_name | str | Tenant 1-18 |
| ledger_type | str | "expenses", "revenue" |
| ledger_group | str | Category grouping |
| ledger_category | str | Subcategories |
| ledger_code | int | Numeric codes |
| ledger_description | str | Bilingual descriptions |
| month | str | "2024-M01" format |
| quarter | str | "2024-Q1" format |
| year | str | "2024", "2025" |
| profit | float | Positive=revenue, Negative=expense |

## Testing

### Manual Testing
Run through example queries in `AGENT_SPEC.md`:
```bash
streamlit run app.py
```

### Programmatic Testing
```python
from graph import run_query

# Test P&L
result = run_query("What's the total P&L for 2024?")
assert "2024" in result["response"]

# Test fuzzy matching
result = run_query("bldg 180")
assert "Building 180" in result["response"]

# Test fallback
result = run_query("What's the weather?")
assert "help you with" in result["response"]
```

## Troubleshooting

### API Key Issues
```
Error: OpenAI API key not found
Solution: Set OPENAI_API_KEY in .env file
```

### Data File Not Found
```
Error: Data file not found: cortex.parquet
Solution: Ensure cortex.parquet is in project root
```

### Module Import Errors
```
Error: ModuleNotFoundError
Solution: Activate virtual environment and reinstall requirements
```

### Streamlit Port Already in Use
```
Error: Port 8501 already in use
Solution: streamlit run app.py --server.port 8502
```

## Performance Considerations

- **Data Loading**: Singleton pattern loads once
- **LLM Calls**: Router + all agents (pnl, property, tenant, general) use LLM for flexible responses
- **Caching**: Pandas operations are fast on 3,924 rows
- **Fuzzy Matching**: rapidfuzz is highly optimized

## Future Enhancements

- [ ] Add caching for repeated queries
- [ ] Implement streaming responses
- [ ] Add visualization charts
- [ ] Support multi-year comparisons
- [ ] Add export to CSV/Excel
- [ ] Implement user authentication
- [ ] Add query history persistence

## License

This project is proprietary and confidential.

## Support

For issues or questions, please contact the development team.

---

**Built with LangGraph, OpenAI, and Streamlit**
