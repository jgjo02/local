# Legal Advisor Agent

A production-quality LangGraph-based legal consultation agent implementing Adaptive RAG, Self-RAG, and Corrective RAG with Human-in-the-Loop verification. Designed for Korean legal queries with support for civil law (민법), labor law (근로기준법), and tenancy law (임대차보호법).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Legal Advisor Agent                          │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ Adaptive RAG │    │  Self-RAG    │    │   Corrective RAG     │  │
│  │    Graph     │    │    Graph     │    │       Graph          │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                  │                        │              │
│         └──────────────────┼────────────────────────┘              │
│                            │                                       │
│                    ┌───────▼────────┐                              │
│                    │  Main Router   │                              │
│                    │  (Query Type)  │                              │
│                    └───────┬────────┘                              │
│                            │                                       │
│              ┌─────────────┼─────────────┐                        │
│              ▼             ▼             ▼                         │
│       ┌─────────┐   ┌──────────┐  ┌──────────┐                   │
│       │  Legal  │   │  Vector  │  │   Web    │                   │
│       │  Tools  │   │  Store   │  │  Search  │                   │
│       └─────────┘   └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **Adaptive RAG**: Dynamically routes queries to the most appropriate retrieval strategy based on query complexity and type
- **Self-RAG**: Performs self-reflection on retrieved documents and generated answers, iterating until quality thresholds are met
- **Corrective RAG**: Evaluates retrieved documents and triggers web search fallback when local knowledge is insufficient
- **Human-in-the-Loop**: Pauses at critical decision points for human review and approval before proceeding
- **Korean Law Support**: Parses Korean legal structure (조/항/호), handles Hanja and Hangul citations
- **Legal Document Processing**: Intelligent chunking by legal article (조문) with metadata extraction
- **Multi-turn Conversations**: Thread-based conversation management with state persistence
- **Legal Citations**: Answers include specific law citations with article numbers
- **Disclaimer Generation**: Automatic legal disclaimer appended to all responses
- **Gradio UI**: Interactive web interface with RAG type selection and HITL approval panel

## RAG Strategy Selection

| Query Type | RAG Strategy | Reasoning |
|---|---|---|
| Simple factual (법령 조회) | Adaptive RAG (direct) | Fast vector lookup sufficient |
| Complex legal analysis | Self-RAG | Requires multiple iterations |
| Current events / recent law | Corrective RAG + Web | Local docs may be outdated |
| Case law / court decisions | Corrective RAG | Needs external sources |
| Calculation (기간, 기한) | Adaptive RAG (direct) | Tool-based answer |

## Directory Structure

```
project3_legal_advisor/
├── README.md
├── requirements.txt
├── .env.example
├── app.py                        # Gradio web application
├── src/
│   ├── config.py                 # Configuration management
│   ├── agent_state.py            # LangGraph state definitions
│   ├── legal_document_processor.py  # Document loading & chunking
│   ├── legal_vector_store.py     # Vector store management
│   ├── legal_tools.py            # LangChain legal tools
│   ├── graders.py                # LLM-based RAG quality graders
│   ├── adaptive_rag_graph.py     # Adaptive RAG LangGraph
│   ├── self_rag_graph.py         # Self-RAG LangGraph
│   ├── corrective_rag_graph.py   # Corrective RAG LangGraph
│   └── main_agent.py             # Main orchestrator agent
├── data/
│   └── sample_legal_docs.txt     # Sample Korean legal documents
└── docs/
    └── architecture.md           # Technical architecture details
```

## Setup

### Prerequisites

- Python 3.11+
- OpenAI API key
- Tavily API key (for web search)

### Installation

```bash
# Clone or navigate to the project directory
cd project3_legal_advisor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Configuration

```bash
# Required
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
```

## Usage

### Running the Web Application

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### Using the Agent Programmatically

```python
from src.main_agent import LegalAdvisorAgent

agent = LegalAdvisorAgent()

# Simple query
result = agent.query(
    query="근로기준법상 최대 근로시간은 몇 시간인가요?",
    rag_type="adaptive"
)
print(result["answer"])

# With human-in-the-loop
result = agent.query(
    query="임대차 계약 해지 통보는 언제까지 해야 하나요?",
    rag_type="self",
    require_human_approval=True
)

# Resume after human approval
if result["status"] == "awaiting_approval":
    final_result = agent.approve_and_continue(
        thread_id=result["thread_id"],
        approved=True,
        feedback="답변이 정확합니다."
    )
```

### Building the Vector Store

```python
from src.legal_document_processor import LegalDocumentProcessor
from src.legal_vector_store import LegalVectorStore

processor = LegalDocumentProcessor()
docs = processor.batch_process(["data/sample_legal_docs.txt"])

store = LegalVectorStore()
store.build_from_documents(docs)
```

## Human-in-the-Loop Workflow

1. Agent processes query through selected RAG pipeline
2. Before delivering final answer, agent pauses at `human_review` checkpoint
3. UI displays: proposed answer, retrieved sources, confidence score, grader evaluations
4. Human reviewer can:
   - **Approve**: Answer is delivered to user
   - **Reject with feedback**: Agent rewrites answer incorporating feedback
   - **Request web search**: Triggers additional Tavily search
5. After approval, answer includes legal disclaimer

## Graders

| Grader | Purpose | Output |
|---|---|---|
| `RelevanceGrader` | Is retrieved doc relevant to query? | yes/no + score |
| `HallucinationGrader` | Does answer hallucinate facts? | yes/no + reasoning |
| `AnswerGrader` | Does answer address the query? | yes/no + score |
| `QueryGrader` | What type/complexity is the query? | type + complexity |

## Legal Tools

| Tool | Description |
|---|---|
| `LegalDocSearchTool` | Vector search over loaded legal documents |
| `LawArticleLookupTool` | Direct lookup of specific law articles |
| `LegalWebSearchTool` | Tavily-powered web search for legal info |
| `CourtDecisionSearchTool` | Search for court decisions and precedents |
| `LegalCalculatorTool` | Calculate legal deadlines, statute of limitations |

## Disclaimer

This system is for informational purposes only and does not constitute legal advice. For specific legal matters, please consult a licensed attorney (변호사). The information provided may not reflect the most recent legal changes.

이 시스템은 정보 제공 목적으로만 사용되며 법적 조언을 구성하지 않습니다. 구체적인 법률 문제에 대해서는 변호사와 상담하시기 바랍니다.
