# Project 4: Company Analysis Agent

Korean listed company report analysis system using Multimodal RAG and Graph RAG.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Company Analysis Agent                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐  │
│  │  PDF Upload  │───▶│ Document Parser  │───▶│   Element Classifier     │  │
│  │  (사업보고서) │    │  (Unstructured)  │    │  Text/Table/Image/Chart  │  │
│  └──────────────┘    └──────────────────┘    └──────────────────────────┘  │
│                                                          │                  │
│                       ┌──────────────────────────────────┤                  │
│                       │                                  │                  │
│                       ▼                                  ▼                  │
│  ┌──────────────────────────┐           ┌───────────────────────────────┐  │
│  │  Multimodal Processor    │           │   Knowledge Graph Builder     │  │
│  │  - CLIP Image Encoding   │           │   - Company Nodes             │  │
│  │  - GPT-4V Description    │           │   - Executive Nodes           │  │
│  │  - Chart Data Extraction │           │   - Subsidiary Nodes          │  │
│  │  - Table OCR             │           │   - Financial Nodes           │  │
│  └──────────────────────────┘           └───────────────────────────────┘  │
│                │                                         │                  │
│                ▼                                         ▼                  │
│  ┌──────────────────────────┐           ┌───────────────────────────────┐  │
│  │  Multimodal Vector Store │           │        Neo4j Graph DB         │  │
│  │  - Text Embeddings       │           │        (Graph RAG)            │  │
│  │  - Image Embeddings      │           │                               │  │
│  │  - FAISS Index           │           └───────────────────────────────┘  │
│  └──────────────────────────┘                           │                  │
│                │                                        │                  │
│                └────────────────┬───────────────────────┘                  │
│                                 ▼                                           │
│                  ┌──────────────────────────┐                              │
│                  │    Company Analyzer       │                              │
│                  │    (Orchestrator)         │                              │
│                  │  - query()               │                              │
│                  │  - compare_companies()   │                              │
│                  │  - generate_report()     │                              │
│                  └──────────────────────────┘                              │
│                                 │                                           │
│                                 ▼                                           │
│                  ┌──────────────────────────┐                              │
│                  │      Gradio Web UI       │                              │
│                  │  - PDF Upload            │                              │
│                  │  - Multimodal Q&A        │                              │
│                  │  - Graph Visualization   │                              │
│                  │  - Company Comparison    │                              │
│                  └──────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Document Processing
- **Unstructured-based PDF parsing**: Extracts text, tables, images, and charts from Korean business reports (사업보고서)
- **Element classification**: Automatically classifies document elements by type (narrative text, financial tables, charts, images)
- **Structure preservation**: Maintains section hierarchy (목차, 사업의 내용, 재무제표, etc.)

### Multimodal RAG
- **CLIP embeddings**: Unified embedding space for text and images
- **GPT-4V integration**: Generates detailed descriptions of charts and financial images
- **Chart data extraction**: Parses numerical data from bar charts, line graphs, pie charts
- **Cross-modal search**: Find images relevant to text queries and vice versa

### Graph RAG (Neo4j)
- **Company knowledge graph**: Models companies, executives, subsidiaries, products, and financial data as nodes
- **Relationship extraction**: LLM-powered entity and relationship extraction from document text
- **Local search**: Entity-focused queries using graph neighborhood traversal
- **Global search**: Community-level summaries using Leiden algorithm
- **Hybrid search**: Combines vector similarity with graph traversal

### Analysis Capabilities
- **재무분석 (Financial Analysis)**: Revenue trends, profitability ratios, debt structure
- **사업분석 (Business Analysis)**: Core business segments, competitive positioning, market share
- **리스크분석 (Risk Analysis)**: Business risks, regulatory exposure, operational risks
- **Company comparison**: Side-by-side comparison of multiple companies

## Setup

### Prerequisites
- Python 3.10+
- Neo4j 5.x (local or AuraDB)
- OpenAI API key
- 8GB+ RAM recommended

### Installation

```bash
# Clone and navigate
cd projects/project4_company_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Neo4j Setup

```bash
# Using Docker
docker run \
  --name neo4j-company \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
  neo4j:5.15

# Or use Neo4j AuraDB (cloud): https://neo4j.com/cloud/platform/aura-graph-database/
```

### Environment Variables

```bash
cp .env.example .env
# Fill in:
# OPENAI_API_KEY=sk-...
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your_password
```

## Usage

### Run the Gradio App

```bash
python app.py
# Access at http://localhost:7860
```

### Programmatic Usage

```python
from src.company_analyzer import CompanyAnalyzer

analyzer = CompanyAnalyzer()

# Analyze a business report
result = analyzer.analyze_business_report(
    pdf_path="data/samsung_2023_annual_report.pdf",
    company_name="삼성전자"
)

# Query the analyzed data
answer = analyzer.query(
    question="삼성전자의 2023년 영업이익은 얼마인가요?",
    analysis_type="재무분석"
)

# Compare companies
comparison = analyzer.compare_companies(
    company1="삼성전자",
    company2="SK하이닉스",
    aspect="반도체 사업 경쟁력"
)

# Generate full analysis report
report = analyzer.generate_analysis_report("삼성전자")
```

### Sample Data

```bash
# Load sample company data
python -c "
from src.knowledge_graph_builder import KnowledgeGraphBuilder
import json

with open('data/sample_company_info.json') as f:
    companies = json.load(f)

builder = KnowledgeGraphBuilder()
for company in companies:
    builder.create_company_node(company)
print('Sample data loaded.')
"
```

## Project Structure

```
project4_company_analysis/
├── README.md
├── requirements.txt
├── .env.example
├── app.py                          # Gradio web application
├── src/
│   ├── config.py                   # Configuration management
│   ├── document_parser.py          # Unstructured PDF parsing
│   ├── multimodal_processor.py     # CLIP + GPT-4V processing
│   ├── multimodal_vector_store.py  # FAISS multimodal store
│   ├── knowledge_graph_builder.py  # Neo4j graph construction
│   ├── graph_rag.py               # Graph RAG implementation
│   └── company_analyzer.py        # Main orchestrator
├── data/
│   └── sample_company_info.json   # Sample Korean company data
└── docs/
    └── architecture.md            # Technical architecture details
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | unstructured[pdf], pdfplumber |
| Text Embeddings | OpenAI text-embedding-3-large |
| Image Embeddings | CLIP (ViT-L/14) |
| Vision LLM | GPT-4o (vision) |
| Vector Store | FAISS |
| Graph Database | Neo4j 5.x |
| LLM | GPT-4o |
| Web UI | Gradio 4.x |
| Graph Visualization | pyvis, networkx |

## Performance

- Document parsing: ~30-60 seconds per 100-page PDF
- CLIP encoding: ~2ms per image (GPU), ~50ms (CPU)
- Graph construction: ~5 minutes per full business report
- Query response: ~3-8 seconds (hybrid search)

## License

MIT License
