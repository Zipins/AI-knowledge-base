from typing import List, Tuple
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from load_kb import load_documents_from_drive

PERSIST_DIR = "kb_chroma"


SYSTEM_PROMPT = """
你是公司内部知识库助手。
你只能根据“提供给你的文档内容”回答问题，不允许使用外部知识。

回答规则：
1. 若提供了文档内容，必须根据文档回答，不得说“没有相关信息”。
2. 若文档内容不足，则可补充一句“以上结论仅根据当前知识库内容整理。”
3. 若没有任何文档内容，则回答“根据当前知识库，没有找到相关信息。”。
"""


# ================= 工具函数 ================= #

def extract_keywords_from_question(question: str) -> List[str]:
    raw_tokens = re.findall(r"[A-Za-z0-9\-]+", question)
    keywords = []
    for tok in raw_tokens:
        tok = tok.strip("-").lower()
        if len(tok) < 2:
            continue
        if tok.isdigit() and len(tok) < 4:
            continue
        keywords.append(tok)

    uniq = []
    seen = set()
    for k in keywords:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def keyword_fallback_search(question: str, max_hits: int = 5) -> List[Document]:
    keywords = extract_keywords_from_question(question)
    if not keywords:
        return []

    all_docs = load_documents_from_drive()  # 加载全文（Drive）

    hits = []
    for d in all_docs:
        text_lower = d.page_content.lower()
        if any(kw in text_lower for kw in keywords):
            hits.append(d)
            if len(hits) >= max_hits:
                break
    return hits


def format_docs(docs: List[Document]) -> str:
    chunks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source") or d.metadata.get("name") or "Unknown"
        page = d.metadata.get("page", "")
        page_info = f"(page {page})" if page else ""

        chunks.append(f"[文档片段 {i} - 来源：{source} {page_info}]\n{d.page_content}\n")

    return "\n\n".join(chunks)


def get_vectordb() -> Chroma:
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectordb


# ================= 主函数 ================= #

def answer_question(question: str, k: int = 8) -> Tuple[str, List[Document], dict]:

    vectordb = get_vectordb()

    # -------- Step 1: similarity search -------- #
    docs = vectordb.similarity_search(question, k=k)

    # -------- Step 2: keyword fallback -------- #
    keywords = extract_keywords_from_question(question)

    extra_docs = []
    if keywords:
        sim_text = "\n".join(d.page_content.lower() for d in docs)
        missing = [kw for kw in keywords if kw not in sim_text]

        if missing:
            extra_docs = keyword_fallback_search(question, max_hits=5)

    # -------- Step 3: merge docs (de-duplicate) -------- #
    merged_docs = []
    seen = set()

    def key_of(d: Document):
        return (
            d.metadata.get("source"),
            d.metadata.get("page"),
            hash(d.page_content),
        )

    for d in docs + extra_docs:
        k_ = key_of(d)
        if k_ not in seen:
            merged_docs.append(d)
            seen.add(k_)

    # -------- Step 4: pass to LLM -------- #
    context = format_docs(merged_docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"用户问题：{question}\n\n"
                f"以下是和问题最相关的知识库内容：\n{context}\n\n"
                f"请严格依照这些内容回答。"
            )
        }
    ]

    resp = llm.invoke(messages)

    debug_info = {
        "keywords": keywords,
        "similarity_hits": len(docs),
        "fallback_hits": len(extra_docs),
        "total_docs_returned": len(merged_docs),
    }

    return resp.content, merged_docs, debug_info
