"""
2주차 1차시: NLP 기초 실습
- 토크나이제이션 (tiktoken)
- TF-IDF 계산
- 코사인 유사도
- Word2Vec 개념 시각화
"""

import os
import re
import math
from collections import Counter
from typing import List, Dict, Tuple
import tiktoken
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# 1. 토크나이제이션
# ============================================================

def demo_tokenization():
    """다양한 토크나이제이션 방식 비교"""
    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    test_texts = [
        "Hello, how are you?",
        "안녕하세요, 어떻게 지내세요?",
        "LangChain은 LLM 앱 개발 프레임워크입니다.",
        "unbelievable transformation",
        "ChatGPT와 Claude의 차이점",
        "1234567890",
    ]

    print("=" * 70)
    print(f"{'텍스트':40s} {'토큰수':6s} {'토큰 ID (앞 5개)':20s}")
    print("-" * 70)

    for text in test_texts:
        tokens = enc.encode(text)
        token_preview = str(tokens[:5]) + ("..." if len(tokens) > 5 else "")
        print(f"{text[:38]:40s} {len(tokens):6d} {token_preview:20s}")

    # 한국어 vs 영어 토큰 비교
    print("\n--- 한국어 vs 영어 토큰 비교 ---")
    pairs = [
        ("Hello World", "안녕하세요 세상"),
        ("machine learning", "머신러닝"),
        ("artificial intelligence", "인공지능"),
    ]
    for en, ko in pairs:
        en_t = len(enc.encode(en))
        ko_t = len(enc.encode(ko))
        ratio = ko_t / en_t
        print(f"EN '{en}' ({en_t}t) | KO '{ko}' ({ko_t}t) | 비율: {ratio:.1f}x")


# ============================================================
# 2. TF-IDF 계산
# ============================================================

def tokenize_korean(text: str) -> List[str]:
    """간단한 한국어 토크나이저 (공백/특수문자 기준)"""
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    return [w for w in text.split() if len(w) > 1]


def compute_tf(doc: str) -> Dict[str, float]:
    """단일 문서의 TF 계산"""
    words = tokenize_korean(doc)
    if not words:
        return {}
    word_count = Counter(words)
    total = len(words)
    return {word: count / total for word, count in word_count.items()}


def compute_idf(docs: List[str]) -> Dict[str, float]:
    """전체 문서 컬렉션의 IDF 계산"""
    N = len(docs)
    doc_freq = Counter()
    for doc in docs:
        words = set(tokenize_korean(doc))
        doc_freq.update(words)

    return {
        word: math.log(N / freq)
        for word, freq in doc_freq.items()
    }


def compute_tfidf(docs: List[str]) -> List[Dict[str, float]]:
    """TF-IDF 계산"""
    idf = compute_idf(docs)
    tfidf_scores = []

    for doc in docs:
        tf = compute_tf(doc)
        scores = {word: tf_val * idf.get(word, 0)
                  for word, tf_val in tf.items()}
        tfidf_scores.append(scores)

    return tfidf_scores


def demo_tfidf():
    """TF-IDF 데모"""
    documents = [
        "파이썬은 간결하고 읽기 쉬운 프로그래밍 언어입니다",
        "파이썬은 머신러닝과 인공지능에 널리 사용됩니다",
        "자바는 객체지향 프로그래밍 언어입니다",
        "자바는 안드로이드 앱 개발에 사용됩니다",
        "머신러닝은 인공지능의 한 분야입니다",
    ]

    print("\n" + "=" * 50)
    print("TF-IDF 분석")
    print("=" * 50)

    tfidf_scores = compute_tfidf(documents)

    for i, (doc, scores) in enumerate(zip(documents, tfidf_scores)):
        top_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"\n문서 {i+1}: {doc[:30]}...")
        print(f"  핵심 단어: {top_words}")


# ============================================================
# 3. 코사인 유사도
# ============================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """두 벡터의 코사인 유사도 계산"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def text_to_vector(text: str, vocabulary: List[str]) -> List[float]:
    """텍스트를 BoW 벡터로 변환"""
    words = tokenize_korean(text)
    word_count = Counter(words)
    return [word_count.get(word, 0) for word in vocabulary]


def demo_cosine_similarity():
    """코사인 유사도 데모"""
    print("\n" + "=" * 50)
    print("코사인 유사도 계산")
    print("=" * 50)

    texts = [
        "파이썬 머신러닝 인공지능",
        "파이썬 딥러닝 신경망",
        "자동차 엔진 기계",
        "인공지능 딥러닝 모델",
    ]

    # 어휘 구축
    all_words = set()
    for text in texts:
        all_words.update(tokenize_korean(text))
    vocabulary = sorted(list(all_words))

    # 벡터 변환
    vectors = [text_to_vector(text, vocabulary) for text in texts]

    # 유사도 계산
    print("\n유사도 행렬:")
    print(f"{'':30s}", end="")
    for text in texts:
        print(f"{text[:10]:12s}", end="")
    print()

    for i, (text_i, vec_i) in enumerate(zip(texts, vectors)):
        print(f"{text_i[:28]:30s}", end="")
        for vec_j in vectors:
            sim = cosine_similarity(vec_i, vec_j)
            print(f"{sim:12.3f}", end="")
        print()


# ============================================================
# 4. OpenAI 임베딩으로 실제 유사도 측정
# ============================================================

def demo_openai_embeddings():
    """OpenAI 임베딩으로 실제 의미적 유사도 측정"""
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        texts = [
            "파이썬 프로그래밍",
            "Python coding",           # 같은 의미 (영어)
            "자바 개발",               # 비슷한 범주
            "요리 레시피",             # 다른 범주
            "기계학습 알고리즘",       # 관련
        ]

        print("\n" + "=" * 50)
        print("OpenAI 임베딩 유사도")
        print("=" * 50)

        vectors = embeddings.embed_documents(texts)

        # 기준 텍스트와 다른 텍스트들의 유사도
        base_idx = 0
        base_text = texts[base_idx]
        base_vec = vectors[base_idx]

        print(f"\n기준: '{base_text}'과의 유사도:")
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            if i != base_idx:
                sim = cosine_similarity(base_vec, vec)
                print(f"  '{text}': {sim:.4f}")

    except Exception as e:
        print(f"OpenAI 임베딩 오류 (API 키 필요): {e}")


# ============================================================
# 5. Word2Vec 유추 (OpenAI 임베딩 활용)
# ============================================================

def demo_word_analogy():
    """단어 유추: 왕 - 남자 + 여자 ≈ ?"""
    try:
        from langchain_openai import OpenAIEmbeddings
        import numpy as np

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        print("\n" + "=" * 50)
        print("단어 유추 실험 (Word Analogy)")
        print("=" * 50)

        words = ["왕", "남자", "여자", "여왕", "공주", "왕자", "사람"]
        vectors = {word: np.array(embeddings.embed_query(word)) for word in words}

        # 왕 - 남자 + 여자 = ?
        result_vec = vectors["왕"] - vectors["남자"] + vectors["여자"]
        result_vec = result_vec / np.linalg.norm(result_vec)

        print("\n'왕' - '남자' + '여자' = ?")
        similarities = {}
        for word, vec in vectors.items():
            if word not in ["왕", "남자", "여자"]:
                norm_vec = vec / np.linalg.norm(vec)
                sim = float(np.dot(result_vec, norm_vec))
                similarities[word] = sim

        for word, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {word}: {sim:.4f}")

    except Exception as e:
        print(f"단어 유추 오류: {e}")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    print("2주차 1차시: NLP 기초 실습\n")

    demo_tokenization()
    demo_tfidf()
    demo_cosine_similarity()
    demo_openai_embeddings()
    demo_word_analogy()
