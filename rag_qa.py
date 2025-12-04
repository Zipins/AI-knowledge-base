from typing import List, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# 这里要和 build_index.py 里的一致
PERSIST_DIR = "kb_chroma"


SYSTEM_PROMPT = """
你是公司内部的知识库助手。
你只能根据“提供给你的文档内容”来回答问题，禁止根据外部知识或臆测回答。

你会收到两部分信息：
1. 用户的问题
2. 若干条与问题最相关的知识库内容（可能为 0 条，也可能很多条）

回答规则：

1. **如果有提供知识库内容（不为空）：**
   - 一定要认真阅读这些内容，从中找出和问题最相关的句子或段落。
   - 尽量用自己的话，把文档里的信息整理成一段清晰、直接的回答。
   - 即使文档没有逐字写出问题的答案，只要有接近的含义，也要根据文档内容做出“最合理的推断”，并可以加一句：
     “以上结论仅根据当前知识库内容整理。”

2. **如果完全没有提供任何知识库内容（为空）：**
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
        source = d.metadata.get("source") or d.metadata.get("name") or d.metadata.get("title") or "Unknown"
        chunks.append(f"[文档片段 {i} - 来源：{source}]\n{d.page_content}\n")
    return "\n\n".join(chunks)


def answer_question(question: str, k: int = 5) -> Tuple[str, List]:
    """主函数：输入问题，返回（答案, 被引用的文档片段列表）。"""
    vectordb = get_vectordb()
    docs = vectordb.similarity_search(question, k=k)
    context = format_docs(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"用户问题：{question}\n\n以下是和问题最相关的知识库内容：\n{context}\n\n请严格根据以上内容回答用户的问题。若内容明显不足以回答，请直接说不知道。",
        },
    ]

    resp = llm.invoke(messages)
    return resp.content, docs


if __name__ == "__main__":
    # 你可以先随便写一个测试问题
    test_q = "我们内部知识库主要包含哪些内容？"
    answer, docs = answer_question(test_q)
    print("=== 问题 ===")
    print(test_q)
    print("\n=== 回答 ===")
    print(answer)
    print("\n=== 引用的文档片段数量 ===", len(docs))

