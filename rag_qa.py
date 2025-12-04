import os
from typing import List, Tuple
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 复用你的 Drive 加载函数
from load_kb import load_documents_from_drive

# 模式：runtime = 云端运行（不访问 Drive）
#       refresh = 每日更新 or 按钮强制更新（访问 Drive + 重建向量库）
KB_MODE = os.getenv("KB_MODE", "runtime")

PERSIST_DIR = "kb_chroma"   # 与原本保持一致


SYSTEM_PROMPT = """
你是公司内部的知识库助手。
你只能根据“提供给你的文档内容”来回答问题，禁止根据外部知识或臆测回答。

你会收到两部分信息：
1. 用户的问题
2. 若干条与问题最相关的知识库内容（可能为 0 条，也可能很多条）

回答规则：

1. 如果有提供知识库内容（不为空）：
   - 必须基于文档内容做合理推断，不允许说“没有相关信息”。
   - 用自己的话总结文档，不要复读。
   - 如含义接近但文档未逐字写出，可给出总结并提示“以上内容仅根据知识库整理”。

2. 如果没有任何知识库内容：
   - 可以回答“未找到相关信息，请咨询负责人或查看原始文档”。

3. 禁止使用外部知识。
"""


# ======================
#   VECTOR STORE 部分
# ======================

def rebuild_vectordb():
    """
    仅在 KB_MODE = refresh 的时候运行。
    从 Google Drive 加载文档 → 重建向量库存盘。
    """
    print(">>> [refresh] 正在从 Google Drive 加载文档...")
    docs = load_documents_from_drive()

    print(f">>> [refresh] 加载完毕，共 {len(docs)} 条文档，开始重建向量库...")

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()

    print(">>> [refresh] 向量库更新完成！")
    return vectordb


def get_vectordb() -> Chroma:
    """
    runtime 模式：只从本地 kb_chroma 加载，不访问 Google Drive
    refresh 模式：重建向量库
    """
    if KB_MODE == "refresh":
        return rebuild_vectordb()

    # runtime 模式
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectordb


# ======================
#   文档格式化
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
#   关键字兜底
# ======================

def extract_keywords_from_question(question: str) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9\-]+", question)
    keywords: List[str] = []

    for tok in raw_tokens:
        tok = tok.strip("-").lower()
        if len(tok) < 2:
            continue
        if tok.isdigit() and len(tok) < 4:
            continue
        keywords.append(tok)

    # 去重
    uniq, seen = [], set()
    for k in keywords:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def keyword_fallback_search(question: str, max_hits: int = 5):
    """
    只有在 refresh 模式才访问 Drive。
    在 runtime 模式下禁止访问 Drive，避免报错。
    """
    if KB_MODE != "refresh":
        return []  # 云端运行不访问 Drive

    keywords = extract_keywords_from_question(question)
    if not keywords:
        return []

    all_docs = load_documents_from_drive()
    hits = []

    for d in all_docs:
        text_lower = d.page_content.lower()
        if any(kw in text_lower for kw in keywords):
            hits.append(d)
            if len(hits) >= max_hits:
                break
    return hits


# ======================
#   主要回答函数
# ======================

def answer_question(question: str, k: int = 8) -> Tuple[str, List]:
    vectordb = get_vectordb()

    # 第一步：语义检索
    docs = vectordb.similarity_search(question, k=k)

    # 第二步：关键字兜底（仅 refresh 模式才允许使用）
    keywords = extract_keywords_from_question(question)
    if keywords and KB_MODE == "refresh":
        joined = "\n".join(d.page_content.lower() for d in docs)

        missing = [kw for kw in keywords if kw not in joined]

        if missing:
            extra = keyword_fallback_search(question)
            existing_keys = set(
                (d.metadata.get("source"), d.metadata.get("page"), hash(d.page_content))
                for d in docs
            )

            for d in extra:
                key = (
                    d.metadata.get("source"),
                    d.metadata.get("page"),
                    hash(d.page_content),
                )
                if key not in existing_keys:
                    docs.append(d)
                    existing_keys.add(key)

    # 第三步：组织 context 送给 LLM
    context = format_docs(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"以下是和问题最相关的知识库内容：\n{context}\n\n"
                f"请严格根据以上内容回答用户的问题。"
                f"若内容明显不足以回答，请直接说不知道。"
            ),
        },
    ]

    resp = llm.invoke(messages)
    return resp.content, docs


# ======================
#   手动测试模式
# ======================

if __name__ == "__main__":
    print(f"当前模式：KB_MODE={KB_MODE}")
    while True:
        q = input("\n请输入问题（exit退出）：")
        if q.lower() in ["exit", "quit"]:
            break
        ans, ds = answer_question(q)
        print("\n回答：", ans)
        print("引用片段数量：", len(ds))
