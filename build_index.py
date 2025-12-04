from __future__ import annotations
import os
from pathlib import Path
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from load_kb import load_documents_from_drive

PERSIST_DIR = "kb_chroma"  # 向量库持久化目录


def split_docs(docs):
    """把长文档切成适合检索的 chunk。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
    )
    return splitter.split_documents(docs)


def build_index():
    # 0. 先把旧向量库删掉，避免脏数据
    if Path(PERSIST_DIR).exists():
        print("🧹 删除旧向量库目录 kb_chroma ...")
        shutil.rmtree(PERSIST_DIR)

    # 1. 从 Google Drive 加载文档
    docs = load_documents_from_drive()
    print(f"原始文档数：{len(docs)}")

    # 2. 切块
    split = split_docs(docs)
    print(f"切块后的文档数：{len(split)}")

    # 3. Embedding + 存入 Chroma
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=split,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()

    # 4. 打印一下云端库里有多少条，后面可以对比
    print("✅ 向量索引构建完成，已保存到 kb_chroma 目录。")
    print("✅ 当前向量库文档数量 =", vectordb._collection.count())


if __name__ == "__main__":
    build_index()
