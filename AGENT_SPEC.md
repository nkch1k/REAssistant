# Real Estate Multi-Agent System — Implementation Spec

## Task Requirements (Original)

Build a multi-agent system using **LangGraph** for real estate asset management:
- Detect request type (price comparison, P&L, asset details, general)
- Extract entities (properties, tenants, timeframes)
- Retrieve data from provided dataset
- Perform calculations and return clear responses
- Handle errors: missing data, ambiguous input, unknown entities

---

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| Orchestration | LangGraph ≥0.2.0 | Required |
| LLM | OpenAI GPT-4o-mini | Cost-effective |
| Data | Pandas + Parquet | In-memory queries |
| UI | Streamlit ≥1.30 | Simple interface |
| Fuzzy match | rapidfuzz ≥3.0 | Entity matching |
| Python | 3.11+ | Modern syntax |

---

## ⚠️ CRITICAL INSTRUCTIONS

### Code Quality — Senior Level
- **Type hints** everywhere (use `typing` module)
- **Pydantic** for data validation where appropriate
- **Docstrings** for all public functions (Google style)
- **No magic strings** — use Enums or constants
- **Single responsibility** — small focused functions
- **DRY** — extract common logic

### Avoid Deprecated Patterns
```python
# ❌ BAD
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
dict.get("key", None)  # redundant None

# ✅ GOOD  
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
dict.get("key")
```

### LangGraph Specifics
- Use `StateGraph` with `TypedDict` state
- Use `add_conditional_edges` for routing
- Nodes are **functions**, not classes
- Return state updates as `dict`

### Error Handling
- Never crash — always return user-friendly message
- Use fuzzy matching before "not found" errors
- Log errors, don't expose internals to user

---

## Dataset Schema

File: `cortex.parquet` (3,924 rows)

| Column | Type | Values |
|--------|------|--------|
| entity_name | str | "PropCo" |
| property_name | str | Building 120, 140, 160, 17, 180 |
| tenant_name | str | Tenant 1–18 |
| ledger_type | str | "expenses", "revenue" |
| ledger_group | str | rental_income, general_expenses, management_fees, taxes_and_insurances, sales_discounts |
| ledger_category | str | Subcategories |
| ledger_code | int | Numeric codes |
| ledger_description | str | Bilingual descriptions |
| month | str | "2024-M01" format |
| quarter | str | "2024-Q1" format |
| year | str | "2024", "2025" |
| profit | float | Positive=revenue, Negative=expense |

---

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
│   └── fallback.py        # Error/unclear handling
├── data/
│   ├── __init__.py
│   ├── loader.py          # Parquet loader (singleton)
│   └── queries.py         # All query functions
├── prompts/
│   └── templates.py       # LLM prompts as constants
├── requirements.txt
└── README.md
```

---

## State Schema

```python
from typing import TypedDict, Optional

class AgentState(TypedDict):
    user_query: str
    intent: str  # pnl | property | tenant | fallback
    entities: dict  # extracted entities
    data: dict  # query results
    response: str  # final answer
    error: Optional[str]
```

---

## LangGraph Flow

```
START → router → [conditional] → agent → END
                      │
         ┌────────────┼────────────┐
         ↓            ↓            ↓
     pnl_agent   property_agent  tenant_agent
                      │
                  fallback (default)
```

---

## Intent Types

| Intent | Example Query |
|--------|---------------|
| `pnl_summary` | "Total P&L for 2024" |
| `pnl_breakdown` | "Show expense breakdown" |
| `property_details` | "How is Building 180 doing?" |
| `property_compare` | "Compare Building 17 vs 120" |
| `tenant_details` | "Revenue from Tenant 8" |
| `tenant_ranking` | "Top 5 tenants" |
| `fallback` | Unclear or off-topic |

---

## User Flow Example

```
User: "What's the total P&L for Building 180 in 2024?"
                    │
                    ▼
Router Agent:
  → intent: "property_details"
  → entities: {property: "Building 180", year: "2024"}
                    │
                    ▼
Property Agent:
  → queries.get_property_pnl("Building 180", year="2024")
  → formats response
                    │
                    ▼
Response: "Building 180 P&L for 2024: Revenue $X, Expenses $Y, Net $Z"
```

---

## Key Query Functions

```python
# data/queries.py — implement these

def get_total_pnl(year: str | None, quarter: str | None) -> float
def get_pnl_breakdown(year: str | None) -> dict[str, float]
def get_property_summary(name: str) -> dict
def get_property_pnl(name: str, year: str | None) -> dict
def compare_properties(p1: str, p2: str) -> dict
def get_tenant_revenue(name: str, year: str | None) -> float
def get_top_tenants(n: int = 5) -> list[dict]
def fuzzy_match_property(query: str) -> str | None
def fuzzy_match_tenant(query: str) -> str | None
```

---

## Prompts Strategy

Keep prompts in `prompts/templates.py`:

```python
ROUTER_SYSTEM_PROMPT = """..."""
ROUTER_USER_TEMPLATE = """..."""
```

Router prompt should:
- Classify intent (one of defined types)
- Extract entities as JSON
- Return structured output

---

## Streamlit UI Requirements

Minimal but functional:
- Text input for queries
- Chat-like message display
- Show processing steps (optional)
- Error messages styled appropriately

---

## Implementation Order

1. `config.py` + `requirements.txt`
2. `data/loader.py` — load parquet
3. `data/queries.py` — all query functions
4. `state.py` — state definition
5. `prompts/templates.py`
6. `agents/router.py`
7. `agents/pnl_agent.py`, `property_agent.py`, `tenant_agent.py`
8. `agents/fallback.py`
9. `graph.py` — wire everything
10. `app.py` — Streamlit UI
11. `README.md`

---

## Testing Queries

System must handle:
```
✓ "What's the total P&L for 2024?"
✓ "Show me Building 180 performance"
✓ "Compare Building 17 and Building 120"
✓ "Top 5 tenants by revenue"
✓ "Revenue from Tenant 8 in Q1 2024"
✓ "What are our biggest expenses?"
✓ "bldg 180" → fuzzy match to "Building 180"
✓ "Tell me about Building 999" → graceful "not found"
✓ "What's the weather?" → fallback response
```

---

## Final Checklist

- [ ] All type hints present
- [ ] No deprecated imports
- [ ] Fuzzy matching works
- [ ] Errors handled gracefully
- [ ] README documents design decisions
- [ ] Code runs without warnings
