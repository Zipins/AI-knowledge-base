from typing import List, Tuple
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# ⚠️ Cloud 上不再使用 Google Drive loader！
# from load_kb import load_documents_from_drive   <-- 删除！！


PERSIST_DIR = "kb_chroma"


SYSTEM_PROMPT = """
你是公司内部的知识库助手。
你只能根据“提供给你的文档内容”来回答问题，禁止根据外部知识或臆测回答。

回答规则：
1. 如果有文档片段：整理为清晰答复。
2. 如果没有文档片段：回答“根据当前知识库，我未找到相关信息。”
3. 不允许凭空编造、不要引用外部知识。
"""


# ======================================================
# 加载已存在的向量库（云端唯一数据来源）
# ======================================================

def get_vectordb() -> Chroma:
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectordb


# ======================================================
# 文档格式化（给模型用）
# ======================================================

def format_docs(docs) -> str:
    chunks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "Unknown")
        chunks.append(f"[文档片段 {i} - 来源：{source}]\n{d.page_content}")
    return "\n\n".join(chunks)


# ======================================================
# 提取关键字（HO5 / BOP / EPLI 等）
# ======================================================

def extract_keywords_from_question(question: str) -> List[str]:
    raw = re.findall(r"[A-Za-z0-9\-]+", question)
    out = []
    for tok in raw:
        tok = tok.strip("-").lower()
        if len(tok) >= 2:
            out.append(tok)
    return list(dict.fromkeys(out))


# ======================================================
# Cloud-safe 关键字 fallback（从向量库内部搜索）
# ======================================================

def keyword_fallback_from_vectordb(vectordb, keywords: List[str], max_hits: int = 5):
    """
    替代原来的 load_documents_from_drive()！
    纯从向量库中检索出全部 chunks，自行做关键词匹配。

    这样 cloud 不需要访问 Google Drive，不会报错。
    """

    # 读取整个向量库的所有 documents
    all_docs = vectordb.get(include=['documents', 'metadatas'])  # Chroma 0.5+ API

    docs = []
    for content, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        text_lower = content.lower()
        if any(kw in text_lower for kw in keywords):
            from langchain_core.documents import Document
            docs.append(Document(page_content=content, metadata=meta))
            if len(docs) >= max_hits:
                break
    return docs


# ======================================================
# 主入口
# ======================================================

def answer_question(question: str, k: int = 8) -> Tuple[str, List]:
    vectordb = get_vectordb()

    # 1) 语义检索
    docs = vectordb.similarity_search(question, k=k)

    # 2) 检查是否关键字缺失
    keywords = extract_keywords_from_question(question)
    if keywords:
        joined = "\n".join(d.page_content.lower() for d in docs)
        missing = [kw for kw in keywords if kw not in joined]

        if missing:
            # 做 cloud-safe fallback
            extra_docs = keyword_fallback_from_vectordb(vectordb, keywords, max_hits=8)

            # 去除重复
            existing_hash = {hash(d.page_content) for d in docs}
            for d in extra_docs:
                if hash(d.page_content) not in existing_hash:
                    docs.append(d)

    # 3) 模型回答
    context = format_docs(docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"用户问题：{question}\n\n"
            f"相关文档片段：\n{context}\n\n"
            f"请严格根据文档内容作答。"
        }
    ]

    resp = llm.invoke(messages)
    return resp.content, docs


if __name__ == "__main__":
    while True:
        q = input("请输入问题：")
        if q.lower() in ["exit", "quit"]:
            break
        ans, ds = answer_question(q)
        print("\n回答：", ans)
        print("命中文档数：", len(ds))
