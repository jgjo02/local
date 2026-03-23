"""
Knowledge Graph Builder using Neo4j.

Constructs a property graph of Korean listed companies with:
- Company nodes (기업)
- Executive nodes (임원)
- Subsidiary nodes (자회사)
- Product/Service nodes (제품/서비스)
- Financial metric nodes (재무지표)
- Market/Industry nodes (시장/산업)
- Risk factor nodes (리스크)

Relationships: EMPLOYS, OWNS, PRODUCES, COMPETES_IN, HAS_RISK, etc.
"""

import json
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import config
from .document_parser import ParsedDocument


# --- Node and Relationship data classes ---------------------------------

@dataclass
class CompanyNode:
    """Represents a Korean listed company."""
    company_id: str
    name: str
    english_name: Optional[str] = None
    ticker: Optional[str] = None
    market: Optional[str] = None  # KOSPI, KOSDAQ
    sector: Optional[str] = None
    industry: Optional[str] = None
    founded_year: Optional[int] = None
    ceo: Optional[str] = None
    employees: Optional[int] = None
    headquarters: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class ExecutiveNode:
    """Represents a company executive."""
    exec_id: str
    name: str
    position: str
    company_id: str
    birth_year: Optional[int] = None
    tenure_start: Optional[str] = None
    education: Optional[str] = None
    career_history: list[str] = field(default_factory=list)
    shares_owned: Optional[int] = None


@dataclass
class SubsidiaryNode:
    """Represents a subsidiary or affiliated company."""
    subsidiary_id: str
    name: str
    parent_company_id: str
    ownership_pct: Optional[float] = None
    relationship_type: str = "자회사"  # 자회사, 관계회사, 손회사
    country: Optional[str] = None
    business: Optional[str] = None


@dataclass
class FinancialNode:
    """Represents a financial metric for a given period."""
    metric_id: str
    company_id: str
    period: str  # e.g., "2023Q4", "2023FY"
    metric_type: str  # revenue, operating_profit, net_income, etc.
    value: float
    unit: str = "억원"
    yoy_change_pct: Optional[float] = None


@dataclass
class ProductNode:
    """Represents a product or service."""
    product_id: str
    name: str
    company_id: str
    category: Optional[str] = None
    market_share_pct: Optional[float] = None
    revenue_contribution_pct: Optional[float] = None


@dataclass
class RiskNode:
    """Represents a business risk factor."""
    risk_id: str
    description: str
    risk_type: str  # market, regulatory, operational, financial, ESG
    company_id: str
    severity: Optional[str] = None  # high, medium, low


# -----------------------------------------------------------------------

ENTITY_EXTRACTION_SYSTEM_PROMPT = """당신은 한국 기업의 사업보고서에서 구조화된 정보를 추출하는 전문가입니다.
주어진 텍스트에서 다음 유형의 엔티티와 관계를 JSON 형식으로 추출하세요.

엔티티 유형:
- Company: 기업 (name, ticker, sector, market)
- Executive: 임원 (name, position, company)
- Subsidiary: 자회사/관계회사 (name, parent, ownership_pct, relationship_type)
- Product: 제품/서비스 (name, company, category, market_share)
- FinancialMetric: 재무지표 (metric_type, value, unit, period)
- Market: 시장/산업 (name, segment, size)
- Risk: 리스크 요인 (description, risk_type, severity)

관계 유형:
- EMPLOYS: Company -> Executive
- OWNS: Company -> Subsidiary
- PRODUCES: Company -> Product
- COMPETES_IN: Company -> Market
- HAS_RISK: Company -> Risk
- REPORTS_METRIC: Company -> FinancialMetric
- AFFILIATED_WITH: Company -> Company

응답 형식:
{
  "entities": [
    {"type": "Company", "id": "unique_id", "properties": {...}},
    {"type": "Executive", "id": "unique_id", "properties": {...}}
  ],
  "relationships": [
    {"from_id": "id1", "to_id": "id2", "type": "EMPLOYS", "properties": {}}
  ]
}

텍스트에서 실제로 언급된 정보만 추출하세요. 추측하지 마세요."""


class KnowledgeGraphBuilder:
    """
    Builds a Neo4j knowledge graph from Korean company reports.

    Usage:
        builder = KnowledgeGraphBuilder()
        builder.connect()
        builder.build_from_document(parsed_doc, "삼성전자")
        builder.disconnect()
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.uri = uri or config.neo4j.uri
        self.user = user or config.neo4j.user
        self.password = password or config.neo4j.password
        self.database = database or config.neo4j.database
        self._driver = None
        self._openai_client = None

    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=config.openai.api_key)
        return self._openai_client

    # --- Connection Management -------------------------------------------

    def connect(self) -> None:
        """Establish Neo4j driver connection."""
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
            self._create_indexes()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def disconnect(self) -> None:
        """Close Neo4j driver."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    @contextmanager
    def _session(self) -> Generator:
        """Context manager for Neo4j sessions."""
        if not self._driver:
            self.connect()
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def _create_indexes(self) -> None:
        """Create Neo4j indexes for performance."""
        indexes = [
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)",
            "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
            "CREATE INDEX exec_name IF NOT EXISTS FOR (e:Executive) ON (e.name)",
            "CREATE INDEX product_name IF NOT EXISTS FOR (p:Product) ON (p.name)",
            "CREATE INDEX subsidiary_name IF NOT EXISTS FOR (s:Subsidiary) ON (s.name)",
            "CREATE INDEX financial_metric IF NOT EXISTS FOR (f:FinancialMetric) ON (f.company_id, f.period)",
        ]
        for idx_query in indexes:
            try:
                self.run_cypher(idx_query)
            except Exception as e:
                logger.debug(f"Index creation note: {e}")

    # --- Node Creation ---------------------------------------------------

    def create_company_node(self, company_data: dict) -> str:
        """
        Create or update a Company node in Neo4j.

        Args:
            company_data: Dict with company properties (name required)

        Returns:
            company_id (str)
        """
        name = company_data.get("name", "")
        if not name:
            raise ValueError("company_data must include 'name'")

        company_id = company_data.get(
            "company_id", f"company_{self._slugify(name)}"
        )

        cypher = """
        MERGE (c:Company {name: $name})
        SET c.company_id = $company_id,
            c.english_name = $english_name,
            c.ticker = $ticker,
            c.market = $market,
            c.sector = $sector,
            c.industry = $industry,
            c.founded_year = $founded_year,
            c.ceo = $ceo,
            c.employees = $employees,
            c.headquarters = $headquarters,
            c.website = $website,
            c.description = $description,
            c.updated_at = datetime()
        RETURN c.company_id AS id
        """

        params = {
            "name": name,
            "company_id": company_id,
            "english_name": company_data.get("english_name"),
            "ticker": company_data.get("ticker"),
            "market": company_data.get("market"),
            "sector": company_data.get("sector"),
            "industry": company_data.get("industry"),
            "founded_year": company_data.get("founded_year"),
            "ceo": company_data.get("ceo"),
            "employees": company_data.get("employees"),
            "headquarters": company_data.get("headquarters"),
            "website": company_data.get("website"),
            "description": company_data.get("description", ""),
        }

        result = self.run_cypher(cypher, params)
        logger.info(f"Created/updated company node: {name}")

        # Create financial metrics if provided
        if "financials" in company_data:
            for metric in company_data["financials"]:
                metric["company_id"] = company_id
                self._create_financial_metric_node(metric)
                self.create_relationship(
                    company_id, metric.get("metric_id", ""),
                    "REPORTS_METRIC"
                )

        # Create subsidiaries
        for sub in company_data.get("subsidiaries", []):
            sub["parent_company_id"] = company_id
            sub_id = self._create_subsidiary_node(sub)
            self.create_relationship(company_id, sub_id, "OWNS", {
                "ownership_pct": sub.get("ownership_pct")
            })

        # Create executives
        for exec_data in company_data.get("executives", []):
            exec_data["company_id"] = company_id
            exec_id = self.create_executive_node(exec_data)
            self.create_relationship(company_id, exec_id, "EMPLOYS")

        # Create products
        for prod in company_data.get("products", []):
            prod["company_id"] = company_id
            prod_id = self._create_product_node(prod)
            self.create_relationship(company_id, prod_id, "PRODUCES")

        return company_id

    def create_executive_node(self, executive_data: dict) -> str:
        """
        Create or update an Executive node.

        Args:
            executive_data: Dict with executive properties

        Returns:
            exec_id (str)
        """
        name = executive_data.get("name", "")
        position = executive_data.get("position", "")
        company_id = executive_data.get("company_id", "")

        exec_id = executive_data.get(
            "exec_id",
            f"exec_{self._slugify(name)}_{self._slugify(company_id)}"
        )

        cypher = """
        MERGE (e:Executive {name: $name, company_id: $company_id})
        SET e.exec_id = $exec_id,
            e.position = $position,
            e.birth_year = $birth_year,
            e.tenure_start = $tenure_start,
            e.education = $education,
            e.shares_owned = $shares_owned,
            e.updated_at = datetime()
        RETURN e.exec_id AS id
        """

        params = {
            "name": name,
            "company_id": company_id,
            "exec_id": exec_id,
            "position": position,
            "birth_year": executive_data.get("birth_year"),
            "tenure_start": executive_data.get("tenure_start"),
            "education": executive_data.get("education"),
            "shares_owned": executive_data.get("shares_owned"),
        }

        self.run_cypher(cypher, params)
        return exec_id

    def _create_subsidiary_node(self, sub_data: dict) -> str:
        """Create or update a Subsidiary node."""
        name = sub_data.get("name", "")
        parent_id = sub_data.get("parent_company_id", "")
        sub_id = sub_data.get(
            "subsidiary_id",
            f"sub_{self._slugify(name)}_{self._slugify(parent_id)}"
        )

        cypher = """
        MERGE (s:Subsidiary {name: $name, parent_company_id: $parent_company_id})
        SET s.subsidiary_id = $sub_id,
            s.ownership_pct = $ownership_pct,
            s.relationship_type = $relationship_type,
            s.country = $country,
            s.business = $business,
            s.updated_at = datetime()
        RETURN s.subsidiary_id AS id
        """

        params = {
            "name": name,
            "parent_company_id": parent_id,
            "sub_id": sub_id,
            "ownership_pct": sub_data.get("ownership_pct"),
            "relationship_type": sub_data.get("relationship_type", "자회사"),
            "country": sub_data.get("country", "대한민국"),
            "business": sub_data.get("business", ""),
        }

        self.run_cypher(cypher, params)
        return sub_id

    def _create_financial_metric_node(self, metric_data: dict) -> str:
        """Create or update a FinancialMetric node."""
        metric_id = metric_data.get(
            "metric_id",
            f"metric_{uuid.uuid4().hex[:8]}"
        )

        cypher = """
        MERGE (f:FinancialMetric {
            company_id: $company_id,
            period: $period,
            metric_type: $metric_type
        })
        SET f.metric_id = $metric_id,
            f.value = $value,
            f.unit = $unit,
            f.yoy_change_pct = $yoy_change_pct,
            f.updated_at = datetime()
        RETURN f.metric_id AS id
        """

        params = {
            "company_id": metric_data.get("company_id", ""),
            "period": metric_data.get("period", ""),
            "metric_type": metric_data.get("metric_type", ""),
            "metric_id": metric_id,
            "value": metric_data.get("value", 0),
            "unit": metric_data.get("unit", "억원"),
            "yoy_change_pct": metric_data.get("yoy_change_pct"),
        }

        self.run_cypher(cypher, params)
        return metric_id

    def _create_product_node(self, prod_data: dict) -> str:
        """Create or update a Product node."""
        name = prod_data.get("name", "")
        company_id = prod_data.get("company_id", "")
        prod_id = prod_data.get(
            "product_id",
            f"prod_{self._slugify(name)}_{self._slugify(company_id)}"
        )

        cypher = """
        MERGE (p:Product {name: $name, company_id: $company_id})
        SET p.product_id = $prod_id,
            p.category = $category,
            p.market_share_pct = $market_share_pct,
            p.revenue_contribution_pct = $revenue_contribution_pct,
            p.updated_at = datetime()
        RETURN p.product_id AS id
        """

        params = {
            "name": name,
            "company_id": company_id,
            "prod_id": prod_id,
            "category": prod_data.get("category", ""),
            "market_share_pct": prod_data.get("market_share_pct"),
            "revenue_contribution_pct": prod_data.get("revenue_contribution_pct"),
        }

        self.run_cypher(cypher, params)
        return prod_id

    def _create_risk_node(self, risk_data: dict) -> str:
        """Create a Risk node."""
        risk_id = risk_data.get("risk_id", f"risk_{uuid.uuid4().hex[:8]}")
        company_id = risk_data.get("company_id", "")

        cypher = """
        CREATE (r:Risk {
            risk_id: $risk_id,
            description: $description,
            risk_type: $risk_type,
            company_id: $company_id,
            severity: $severity,
            created_at: datetime()
        })
        RETURN r.risk_id AS id
        """

        params = {
            "risk_id": risk_id,
            "description": risk_data.get("description", ""),
            "risk_type": risk_data.get("risk_type", "operational"),
            "company_id": company_id,
            "severity": risk_data.get("severity", "medium"),
        }

        self.run_cypher(cypher, params)
        return risk_id

    # --- Relationships ---------------------------------------------------

    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[dict] = None,
    ) -> None:
        """
        Create a relationship between two nodes by their IDs.

        Searches all node types for matching IDs.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            rel_type: Relationship type (e.g., EMPLOYS, OWNS)
            properties: Optional relationship properties
        """
        if not from_id or not to_id:
            return

        props = properties or {}

        # Universal MERGE using any node with matching id field
        cypher = f"""
        MATCH (a) WHERE a.company_id = $from_id
            OR a.exec_id = $from_id
            OR a.subsidiary_id = $from_id
            OR a.product_id = $from_id
            OR a.metric_id = $from_id
            OR a.risk_id = $from_id
        MATCH (b) WHERE b.company_id = $to_id
            OR b.exec_id = $to_id
            OR b.subsidiary_id = $to_id
            OR b.product_id = $to_id
            OR b.metric_id = $to_id
            OR b.risk_id = $to_id
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        RETURN type(r) AS rel
        """

        self.run_cypher(cypher, {
            "from_id": from_id,
            "to_id": to_id,
            "props": props,
        })

    # --- LLM Entity Extraction -------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def extract_entities_with_llm(
        self, text: str, company_name: Optional[str] = None
    ) -> dict:
        """
        Use GPT-4o to extract entities and relationships from document text.

        Args:
            text: Text chunk to extract from
            company_name: Primary company name for context

        Returns:
            Dict with 'entities' and 'relationships' lists
        """
        context = (
            f"분석 대상 기업: {company_name}\n\n" if company_name else ""
        )
        user_message = f"{context}다음 텍스트에서 엔티티와 관계를 추출하세요:\n\n{text}"

        response = self.openai_client.chat.completions.create(
            model=config.openai.llm_model,
            messages=[
                {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=config.llm.entity_extraction_max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Entity extraction JSON parse error: {raw[:200]}")
            return {"entities": [], "relationships": []}

    def _save_extracted_entities(
        self, extracted: dict, company_name: str
    ) -> None:
        """Persist LLM-extracted entities to Neo4j."""
        entity_id_map: dict[str, str] = {}

        for entity in extracted.get("entities", []):
            etype = entity.get("type", "")
            props = entity.get("properties", {})
            temp_id = entity.get("id", "")

            try:
                if etype == "Company":
                    props.setdefault("name", company_name)
                    real_id = self.create_company_node(props)
                elif etype == "Executive":
                    props["company_id"] = f"company_{self._slugify(company_name)}"
                    real_id = self.create_executive_node(props)
                elif etype == "Subsidiary":
                    props["parent_company_id"] = f"company_{self._slugify(company_name)}"
                    real_id = self._create_subsidiary_node(props)
                elif etype == "Product":
                    props["company_id"] = f"company_{self._slugify(company_name)}"
                    real_id = self._create_product_node(props)
                elif etype == "FinancialMetric":
                    props["company_id"] = f"company_{self._slugify(company_name)}"
                    real_id = self._create_financial_metric_node(props)
                elif etype == "Risk":
                    props["company_id"] = f"company_{self._slugify(company_name)}"
                    real_id = self._create_risk_node(props)
                else:
                    continue

                entity_id_map[temp_id] = real_id

            except Exception as e:
                logger.warning(f"Failed to create {etype} entity: {e}")

        # Create relationships
        for rel in extracted.get("relationships", []):
            from_id = entity_id_map.get(rel.get("from_id", ""))
            to_id = entity_id_map.get(rel.get("to_id", ""))
            rel_type = rel.get("type", "RELATED_TO")
            props = rel.get("properties", {})

            if from_id and to_id:
                try:
                    self.create_relationship(from_id, to_id, rel_type, props)
                except Exception as e:
                    logger.warning(f"Failed to create relationship: {e}")

    # --- Full Pipeline ---------------------------------------------------

    def build_from_document(
        self,
        document: ParsedDocument,
        company_name: str,
        max_chunks: int = 50,
        chunk_size: int = 1500,
    ) -> None:
        """
        Build complete knowledge graph from a parsed document.

        Processes text sections in chunks and extracts entities via LLM.

        Args:
            document: Parsed document from DocumentParser
            company_name: Company being analyzed
            max_chunks: Max number of text chunks to process
            chunk_size: Characters per chunk
        """
        logger.info(
            f"Building knowledge graph for {company_name} "
            f"from {len(document.all_elements)} elements"
        )

        # Ensure company node exists
        self.create_company_node({"name": company_name})

        # Collect text for processing
        all_text = "\n\n".join(
            e.text for e in document.all_elements
            if e.text and len(e.text) > 50
        )

        # Split into chunks
        chunks = self._split_text(all_text, chunk_size)
        chunks = chunks[:max_chunks]

        logger.info(f"Processing {len(chunks)} text chunks for entity extraction")

        for i, chunk in enumerate(chunks):
            try:
                extracted = self.extract_entities_with_llm(chunk, company_name)
                self._save_extracted_entities(extracted, company_name)
                logger.debug(
                    f"Chunk {i + 1}/{len(chunks)}: "
                    f"extracted {len(extracted.get('entities', []))} entities"
                )
            except Exception as e:
                logger.warning(f"Entity extraction failed for chunk {i}: {e}")

        logger.info(f"Knowledge graph build complete for {company_name}")

    def _split_text(self, text: str, chunk_size: int) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        overlap = chunk_size // 5
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # --- Graph Queries ---------------------------------------------------

    def run_cypher(
        self, query: str, params: Optional[dict] = None
    ) -> list[dict]:
        """
        Execute a raw Cypher query.

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            List of result records as dicts
        """
        if not self._driver:
            self.connect()

        with self._session() as session:
            result = session.run(query, parameters=params or {})
            return [dict(record) for record in result]

    def clear_graph(self) -> None:
        """Remove all nodes and relationships from the database."""
        logger.warning("Clearing entire graph database!")
        self.run_cypher("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    def get_company_subgraph(self, company_name: str, depth: int = 2) -> dict:
        """Get the subgraph centered on a company."""
        cypher = """
        MATCH (c:Company {name: $name})
        CALL apoc.path.subgraphAll(c, {maxLevel: $depth})
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        try:
            return self.run_cypher(cypher, {"name": company_name, "depth": depth})
        except Exception:
            # Fallback without APOC
            cypher_fallback = """
            MATCH (c:Company {name: $name})-[r*1..2]-(n)
            RETURN DISTINCT n, r, c
            """
            return self.run_cypher(cypher_fallback, {"name": company_name})

    def get_company_stats(self) -> dict:
        """Return graph statistics."""
        results = self.run_cypher("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
        """)
        node_counts = {r["label"]: r["count"] for r in results}

        rel_results = self.run_cypher("""
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
        """)
        rel_counts = {r["rel_type"]: r["count"] for r in rel_results}

        return {"nodes": node_counts, "relationships": rel_counts}

    # --- Utilities -------------------------------------------------------

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a safe identifier."""
        if not text:
            return "unknown"
        # Remove special chars, replace spaces with underscore
        result = re.sub(r"[^\w가-힣]", "_", text.strip())
        result = re.sub(r"_+", "_", result)
        return result.strip("_")[:50].lower()
