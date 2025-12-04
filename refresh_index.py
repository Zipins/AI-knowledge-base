# refresh_index.py
import os

os.environ["KB_MODE"] = "refresh"

from rag_qa import get_vectordb

print(">>> 开始每日更新知识库向量库...")
get_vectordb()
print(">>> 完成。")
