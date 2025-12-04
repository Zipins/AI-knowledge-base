import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

PERSIST_DIR = "kb_chroma"

# -------------------------
#  加载向量库
# -------------------------
def load_vectordb():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )


# -------------------------
#  fallback：从向量库 metadata 中获取全文
# -------------------------
def keyword_fallback_search(query, vectordb, max_hits=5):

    # vectordb._collection.get() 取回所有 metadata
    data = vectordb._collection.get(include=["metadatas", "documents"])

    results = []
    for meta, txt in zip(data["metadatas"], data["documents"]):
        source = meta.get("source_text", "")
        if query.lower() in source.lower():
            results.append(
                Document(
                    page_content=source,
                    metadata={"source": meta.get("source", "fallback")}
                )
            )
        if len(results) >= max_hits:
            break

    return results


# -------------------------
#  主回答逻辑
# -------------------------
def answer_question(query):
    vectordb = load_vectordb()

    # 1) 用向量检索
    docs = vectordb.similarity_search(query, k=5)

    # 2) 如果向量检索为空 → fallback
    if not docs:
        fallback_docs = keyword_fallback_search(query, vectordb)
        docs = fallback_docs

    # 3) 如果还是没找到 → 返回空
    if not docs:
        return "目前知识库中没有找到相关信息。", []

    # 4) LLM 生成答案
    llm = ChatOpenAI(temperature=0)

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
你是公司的内部知识库助手。请根据以下知识库内容回答用户问题。

问题：
{query}

知识库内容：
{context}

请给出准确回答：
"""
    response = llm.invoke(prompt)
    return response.content, docs
