import os
IS_CLOUD = os.getenv("IS_CLOUD", "false").lower() == "true"

from typing import List, Tuple
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 复用你已有的 Drive 加载函数
from load_kb import load_documents_from_drive


# 这里要和 build_index.py 里的一致
PERSIST_DIR = "kb_chroma"


SYSTEM_PROMPT = """
你是公司内部的知识库助手。
你只能根据“提供给你的文档内容”来回答问题，禁止根据外部知识或臆测回答。

你会收到两部分信息：
1. 用户的问题
2. 若干条与问题最相关的知识库内容（可能为 0 条，也可能很多条）

回答规则：

1. 如果有提供知识库内容（不为空）：
   - 一定要认真阅读这些内容，从中找出和问题最相关的句子或段落。
   - 尽量用自己的话，把文档里的信息整理成一段清晰、直接的回答。
   - 即使文档没有逐字写出问题的答案，只要有接近的含义，也要根据文档内容做出“最合理的推断”，并可以加一句：
     “以上结论仅根据当前知识库内容整理。”

2. 如果完全没有提供任何知识库内容（为空）：
   - 才可以回答类似：
     “根据目前提供的知识库内容，我没有找到相关信息，请咨询负责人或查看原始文档。”
   - 不能自己编造外部知识。

3. 禁止的行为：
   - 当已经给出了多条文档内容时，不允许说“没有相关信息”。
   - 不要引用外部常识或网上资料，只能基于给出的内容作答。
   - 不要重复问题本身，直接给出结论和必要的简短解释即可。
"""


def get_vectordb() -> Chroma:
    """加载已经持久化好的 Chroma 向量库。"""
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vectordb


def format_docs(docs) -> str:
    """把检索出来的文档片段拼成一个大字符串，给模型当 context 用。"""
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


# ========= 关键字兜底相关函数（新加） =========

def extract_keywords_from_question(question: str) -> List[str]:
    """
    从问题里抽取可能有用的“字母数字”关键字，用于全文兜底搜索。
    例如：
      - “Ho5保单哪个公司能做” -> ["ho5"]
      - “EPLI 保障什么” -> ["epli"]
      - “什么是COI证书” -> ["coi"]
    """
    # 把类似 HO5 / HO-5 / DP3 / BOP / EPLI / COI 这些 token 抽出来
    raw_tokens = re.findall(r"[A-Za-z0-9\-]+", question)
    keywords: List[str] = []

    for tok in raw_tokens:
        tok = tok.strip("-").lower()
        if len(tok) < 2:
            continue
        # 非常短的纯数字意义不大（如“10”、“20”之类），先略过
        if tok.isdigit() and len(tok) < 4:
            continue
        keywords.append(tok)

    # 去重并保持顺序
    seen = set()
    uniq = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def keyword_fallback_search(question: str, max_hits: int = 5):
    """
    关键字兜底搜索：
    - 当向量检索里没有包含这些关键字的片段时，
      从原始 Drive 文档里做一轮简单的“包含关键字”搜索。
    """
    keywords = extract_keywords_from_question(question)
    if not keywords:
        return []

    # 从 Drive 加载所有文档（数量不多，这样做没问题）
    all_docs = load_documents_from_drive()

    hits = []
    for d in all_docs:
        text_lower = d.page_content.lower()
        # 只要这个片段里至少包含一个关键字就可以
        if any(kw in text_lower for kw in keywords):
            hits.append(d)
            if len(hits) >= max_hits:
                break

    return hits


def answer_question(question: str, k: int = 8) -> Tuple[str, List]:
    """
    主函数：输入问题，返回（答案, 被引用的文档片段列表）。

    升级点：
    1. 向量检索 top_k 从 5 提高到 8；
    2. 如果问题里有像 HO5 / EPLI / COI 这种字母数字关键字，
       且向量检索的结果里没有包含这些关键字的内容，
       则再做一次全文关键字兜底搜索，并把命中的片段追加进来。
    """
    vectordb = get_vectordb()
    # 第一步：正常的语义检索
    docs = vectordb.similarity_search(question, k=k)

    # 第二步：检查是否需要做关键字兜底
    keywords = extract_keywords_from_question(question)
    if keywords:
        joined = "\n".join(d.page_content.lower() for d in docs)
        # 看看当前检索结果里是否已经覆盖了这些关键字
        missing_keywords = [kw for kw in keywords if kw not in joined]

        if missing_keywords:
            # 当前语义检索结果里完全没出现这些关键字 -> 做全文兜底搜索
            extra_docs = keyword_fallback_search(question, max_hits=5)

            # 把兜底命中的片段追加到 docs 里，避免重复
            existing_keys = set()
            for d in docs:
                key = (
                    d.metadata.get("source"),
                    d.metadata.get("page"),
                    hash(d.page_content),
                )
                existing_keys.add(key)

            for d in extra_docs:
                key = (
                    d.metadata.get("source"),
                    d.metadata.get("page"),
                    hash(d.page_content),
                )
                if key not in existing_keys:
                    docs.append(d)
                    existing_keys.add(key)

    # 第三步：把所有命中的片段给到模型
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


if __name__ == "__main__":
    while True:
        q = input("\n请输入问题（输入 exit 退出）：")
        if q.lower() in ["exit", "quit"]:
            break
        ans, ds = answer_question(q)
        print("\n回答：", ans)
        print("引用片段数量：", len(ds))
