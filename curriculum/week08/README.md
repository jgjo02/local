# Week 8: LangGraph 에이전트 활용

## 주요 학습 내용

LangGraph로 상태 기반 에이전트를 구축하고, 메모리, Human-in-the-Loop를 적용합니다.

## 세션 구성

| 차시 | 주제 |
|------|------|
| 1차시 | LangGraph StateGraph & 핵심 개념 |
| 2차시 | MessagesState & Reducer 함수 |
| 3차시 | LangGraph ReAct 에이전트 구현 |
| 4차시 | 메모리 & 체크포인트 |
| 5차시 | Human-in-the-Loop & 서브그래프 |

## 학습 목표

- StateGraph, Node, Edge의 관계를 이해하고 구현한다
- MemorySaver로 대화 상태를 영구 저장한다
- interrupt_before로 사용자 승인 워크플로우를 구현한다

## 핵심 개념

```
LangGraph = 상태(State) + 노드(Node) + 엣지(Edge)

상태: TypedDict로 정의한 데이터 구조
노드: 상태를 입력받아 수정하는 함수
엣지: 노드 간 연결 (일반 엣지 or 조건부 엣지)
```
