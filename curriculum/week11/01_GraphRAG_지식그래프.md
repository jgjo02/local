# Week 11-1: Graph RAG 및 지식 그래프

## 학습 목표
- Knowledge Graph의 개념과 구조 이해
- Neo4j + Cypher로 지식 그래프 구축
- Graph RAG의 장점과 활용 방법

---

## 1. 왜 Graph RAG인가?

### 1.1 일반 RAG의 한계

```
질문: "삼성전자와 SK하이닉스의 공통 고객사는?"

Naive RAG:
- 삼성전자 문서 검색 → 고객사 목록
- SK하이닉스 문서 검색 → 고객사 목록
- 교집합을 자동으로 찾지 못함 ❌

Graph RAG:
- 노드: 삼성전자, SK하이닉스, Apple, TSMC...
- 엣지: SUPPLIES_TO, COMPETES_WITH
- Cypher로 교집합 자동 추출 ✅
```

### 1.2 지식 그래프 구조

```
(삼성전자) -[COMPETES_WITH]-> (SK하이닉스)
(삼성전자) -[SUPPLIES_TO]-> (Apple)
(SK하이닉스) -[SUPPLIES_TO]-> (Apple)
(삼성전자) -[LOCATED_IN]-> (수원)
(삼성전자) -[FOUNDED_BY]-> (이병철)
```

---

## 2. Neo4j 설치 및 연결

### 2.1 Docker로 Neo4j 실행

```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

### 2.2 Python 연결

```python
from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: dict = None) -> list:
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

# 연결
conn = Neo4jConnection(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
```

---

## 3. Cypher 기초

### 3.1 노드 생성

```cypher
-- 회사 노드 생성
CREATE (c:Company {
    name: "삼성전자",
    ticker: "005930",
    sector: "반도체",
    revenue_trillion: 258
})

-- 여러 노드 한번에
CREATE
    (a:Company {name: "삼성전자", sector: "반도체"}),
    (b:Company {name: "SK하이닉스", sector: "반도체"}),
    (c:Company {name: "Apple", sector: "테크"})
```

### 3.2 관계 생성

```cypher
-- 관계 생성
MATCH (a:Company {name: "삼성전자"})
MATCH (b:Company {name: "Apple"})
CREATE (a)-[:SUPPLIES_TO {product: "메모리", since: 2007}]->(b)

-- 경쟁 관계
MATCH (a:Company {name: "삼성전자"})
MATCH (b:Company {name: "SK하이닉스"})
CREATE (a)-[:COMPETES_WITH]->(b)
```

### 3.3 쿼리

```cypher
-- 삼성전자의 모든 관계
MATCH (c:Company {name: "삼성전자"})-[r]->(target)
RETURN c.name, type(r), target.name

-- 공통 고객 찾기
MATCH (a:Company {name: "삼성전자"})-[:SUPPLIES_TO]->(customer)
MATCH (b:Company {name: "SK하이닉스"})-[:SUPPLIES_TO]->(customer)
RETURN customer.name AS common_customer

-- 2홉 관계 탐색
MATCH path = (c:Company {name: "삼성전자"})-[*1..2]->(target)
RETURN path LIMIT 20
```

---

## 4. LangChain + Neo4j

```python
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Neo4j 그래프 연결
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# 스키마 자동 탐색
print(graph.schema)

# GraphCypherQAChain: 자연어 → Cypher → 결과
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
)

result = chain.invoke({"query": "삼성전자의 주요 고객사는 어디인가요?"})
print(result["result"])
```

---

## 5. 기업 분석 지식 그래프 구축

```python
def build_company_graph(conn: Neo4jConnection, companies: list) -> None:
    """기업 지식 그래프 구축"""

    # 회사 노드 생성
    for company in companies:
        conn.query("""
            MERGE (c:Company {name: $name})
            SET c.ticker = $ticker,
                c.sector = $sector,
                c.market_cap = $market_cap
        """, company)

    # 경쟁사 관계 설정
    competitor_pairs = [
        ("삼성전자", "SK하이닉스"),
        ("현대자동차", "기아"),
        ("카카오", "네이버"),
    ]

    for a, b in competitor_pairs:
        conn.query("""
            MATCH (a:Company {name: $a})
            MATCH (b:Company {name: $b})
            MERGE (a)-[:COMPETES_WITH]->(b)
            MERGE (b)-[:COMPETES_WITH]->(a)
        """, {"a": a, "b": b})

    # 공급망 관계
    supply_chains = [
        {"supplier": "삼성전자", "customer": "Apple", "product": "메모리/디스플레이"},
        {"supplier": "SK하이닉스", "customer": "Apple", "product": "메모리"},
    ]

    for sc in supply_chains:
        conn.query("""
            MATCH (s:Company {name: $supplier})
            MATCH (c:Company {name: $customer})
            MERGE (s)-[:SUPPLIES_TO {product: $product}]->(c)
        """, sc)


def query_company_relationships(conn: Neo4jConnection, company_name: str) -> dict:
    """기업 관계 쿼리"""
    return {
        "competitors": conn.query("""
            MATCH (c:Company {name: $name})-[:COMPETES_WITH]->(comp)
            RETURN comp.name AS name, comp.sector AS sector
        """, {"name": company_name}),

        "customers": conn.query("""
            MATCH (c:Company {name: $name})-[r:SUPPLIES_TO]->(cust)
            RETURN cust.name AS name, r.product AS product
        """, {"name": company_name}),
    }
```

---

## 핵심 정리

| 항목 | 일반 RAG | Graph RAG |
|------|----------|-----------|
| 관계 탐색 | 불가 | 가능 (1-N홉) |
| 집계 쿼리 | 제한적 | Cypher로 강력 |
| 데이터 구조 | 비구조화 텍스트 | 구조화 그래프 |
| 적합 도메인 | 일반 문서 Q&A | 기업 분석, 지식베이스 |
| 구축 비용 | 낮음 | 높음 |
