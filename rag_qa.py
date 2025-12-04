import os
import re
from typing import List, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from load_kb import load_documents_from_drive

# 模式：
# runtime = 云端（只读向量库 + fallback，不访问 Drive）
# refresh = 每日更新（访问 Drive + 重建向量库）
KB_MODE = os.getenv("KB_MODE", "runtime")

PERSIST_DIR = "kb_chroma"


SYSTEM_PROMPT = """
你是公司内部的知识库助手。
你只能根据“提供给你的文档内容”来回答问题，禁止根据外部知识或臆测回答。

回答规则：
1. 如果有提供文档内容：
   - 必须基于文档内容做出合理推断，不得说“没有相关信息”。
2. 如果完全没有文档内容：
   - 才可以回答“当前知识库中未找到相关内容，请咨询负责人”。
3. 禁止引用外部网络知识。
"""


# ======================
#  向量库管理
# ======================

def rebuild_vectordb():
    """refresh 模式：从 Google Drive 加载文档并重建向量库"""
    print(">>> [refresh] 开始从 Google Drive 加载文档...")
    docs = load_documents_from_drive()

    print(f">>> [refresh] 加载到 {len(docs)} 条文档，开始构建向量库...")
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    print(">>> [refresh] 向量库构建完成！")
    return vectordb


def get_vectordb() -> Chroma:
    """根据模式加载或构建向量库"""

    if KB_MODE == "refresh":
        return rebuild_vectordb()

    # runtime 模式 —— 只加载本地向量库，不访问 Drive
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectordb


# ======================
#  文本格式化
# ======================

def format_docs(docs) -> str:
    chunks = []
    for i, d in enumerate(docs, start=1):
        source = (
            d.metadata.get("source")
            or d.metadata.get("name")
            or d.metadata.get("title")
            or "Unknown"
        )
        chunks.append(f"[文档片段 {i} - 来源：{source}]\n{d.page_content}\n")
    return "\n\n".join(chunks)


# ======================
#  关键字提取
# ======================

def extract_keywords_from_question(question: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9\-]+", question)
    out = []

    for tok in raw:
        tok = tok.strip("-").lower()
        if len(tok) < 2:
            continue
        if tok.isdigit() and len(tok) < 3:
            continue
        out.append(tok)

    uniq, seen = [], set()
    for k in out:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


# ======================
#  🔥关键字兜底（runtime 模式也可用）
# ======================

def keyword_fallback_search(question: str, vectordb, max_hits: int = 5) -> List[Document]:
    """
    keyword fallback（兜底搜索）：
    —— 不访问 Google Drive（runtime 模式安全）
    —— 直接在向量库中的全部 chunks 做全文搜索
    """
    keywords = extract_keywords_from_question(question)
    if not keywords:
        return []

    # 获取向量库中所有文本
    data = vectordb._collection.get(include=["documents", "metadatas"])
    docs = []

    for content, meta in zip(data["documents"], data["metadatas"]):
        text_lower = content.lower()

        if any(kw in text_lower for kw in keywords):
            docs.append(Document(page_content=content, metadata=meta))
            if len(docs) >= max_hits:
                break

    return docs


# ======================
#   主回答函数
# ======================

def answer_question(question: str, k: int = 8) -> Tuple[str, List[Document]]:
    vectordb = get_vectordb()

    # 1) 语义检索
    docs = vectordb.similarity_search(question, k=k)

    # 2) 关键字兜底 —— runtime 模式也启用！！
    keywords = extract_keywords_from_question(question)
    if keywords:
        joined = "\n".join(d.page_content.lower() for d in docs)
        missing = [kw for kw in keywords if kw not in joined]

        if missing:
            extra_docs = keyword_fallback_search(question, vectordb)
            # 避免重复
            existing = set(
                (hash(d.page_content), d.metadata.get("source")) for d in docs
            )
            for d in extra_docs:
                key = (hash(d.page_content), d.metadata.get("source"))
                if key not in existing:
                    docs.append(d)
                    existing.add(key)

    # 3) 构造 context
    context = format_docs(docs)

    # 4) 让模型回答
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"以下是和问题最相关的知识库内容：\n{context}\n\n"
                f"请严格根据以上内容回答用户的问题。"
            ),
        },
    ]

    resp = llm.invoke(messages)
    return resp.content, docs


# ======================
#   CLI 模式
# ======================

if __name__ == "__main__":
    print(f"当前模式：KB_MODE={KB_MODE}")
    while True:
        q = input("\n请输入问题（exit 退出）：")
        if q.lower() in ["exit", "quit"]:
            break
        ans, ds = answer_question(q)
        print("\n回答：", ans)
        print("引用片段数量：", len(ds))
