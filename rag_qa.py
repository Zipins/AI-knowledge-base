# rag_qa.py
from typing import List, Tuple, Dict, Any
import os
import json
import shutil
import hashlib
import re

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import GoogleDriveLoader
from langchain_core.documents import Document  # 用于构造文档对象


# =============== 基本配置 ===============

# 向量库持久化目录（要和以前 build_index.py 里保持一致）
PERSIST_DIR = "kb_chroma"

# 用来存 index 元数据（文档清单、hash 等）
META_PATH = os.path.join(PERSIST_DIR, "kb_index_metadata.json")

# 你的 Google Drive 知识库所在的 folder_id（建议放到环境变量里）
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "在这里填你的_folder_id")

SYSTEM_PROMPT = """
你是公司内部的知识库助手。
❗️重要规则：
1. 你只能根据“提供给你的文档内容”来回答问题，禁止用外部知识或瞎编。
2. 如果文档里确实没有答案，就老实说“目前知识库中没有相关信息，请咨询负责人或查看原始文档”，不要猜。
3. 回答要尽量简洁、直接，优先用中文。
"""

# 全局缓存：最近一次加载的文档列表（给管理页面用）
_last_loaded_docs: List[Document] = []


# =============== Embedding & LLM ===============

def _get_embeddings():
    # 你可以根据自己账户情况换模型
    return OpenAIEmbeddings(model="text-embedding-3-small")


def _get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============== 1. 从 Google Drive 加载文档 ===============

def load_documents_from_drive() -> List[Document]:
    """
    从 Google Drive 加载所有知识库文档。
    同时更新全局 _last_loaded_docs，供管理页面展示。
    """
    global _last_loaded_docs

    loader = GoogleDriveLoader(
        folder_id=GOOGLE_DRIVE_FOLDER_ID,
        recursive=True,
        file_types=["document"],  # 只要 Google Docs
    )
    docs = loader.load()
    _last_loaded_docs = docs
    return docs


def _compute_manifest(docs: List[Document]) -> Dict[str, Any]:
    """
    根据文档的 metadata 生成一个“指纹”(manifest)，用来判断知识库是否有变动。
    """
    manifest: Dict[str, Any] = {}
    for d in docs:
        src = d.metadata.get("source", "unknown")
        modified_time = d.metadata.get("modifiedTime", "")
        title = d.metadata.get("title", "")
        manifest[src] = {
            "modifiedTime": modified_time,
            "title": title,
        }
    return manifest


def _hash_manifest(manifest: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()


# =============== 2. 自动检测变更 + 重建向量库 ===============

def _load_index_metadata() -> Dict[str, Any]:
    if not os.path.exists(META_PATH):
        return {}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index_metadata(meta: Dict[str, Any]):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    """
    根据 Google Drive 文档情况自动决定：
    - 没有 index → 新建
    - 有 index 且文档没变 → 直接加载
    - 文档有变 / 手动要求重建 → 删除旧 index + 重建
    """
    embeddings = _get_embeddings()

    # 先从 Drive 拉一遍文档（顺便作为“真相来源”）
    docs = load_documents_from_drive()
    manifest = _compute_manifest(docs)
    manifest_hash = _hash_manifest(manifest)

    old_meta = _load_index_metadata()
    old_hash = old_meta.get("manifest_hash")

    need_rebuild = force_rebuild or (manifest_hash != old_hash)

    if (not need_rebuild) and os.path.isdir(PERSIST_DIR):
        # 不需要重建，直接加载现有向量库
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
        return vectordb

    # 需要重建：删旧库
    shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()

    # 保存新的 metadata
    _save_index_metadata(
        {
            "manifest_hash": manifest_hash,
            "doc_count": len(docs),
            "docs": manifest,
        }
    )

    return vectordb


# =============== 3. 关键字提取 & 兜底检索 ===============

def extract_keywords_from_question(question: str) -> List[str]:
    """
    非严格版关键字提取：
    - 支持中英混合，提取汉字词 + 字母数字串
    - 过滤掉太短的词
    """
    # 提取“连续的汉字或字母数字”
    tokens = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_.]+", question)
    # 简单去掉一些非常常见的虚词（可以按需要扩展）
    stop_words = {"的", "了", "吗", "呢", "啊", "请问", "可以", "怎么", "怎样"}
    keywords: List[str] = []
    for t in tokens:
        t2 = t.strip().lower()
        if not t2:
            continue
        if t2 in stop_words:
            continue
        if len(t2) <= 1:
            continue
        keywords.append(t2)
    return keywords


def keyword_fallback_search(
    question: str,
    vectordb: Chroma,
    max_hits: int = 5,
) -> List[Document]:
    """
    关键字兜底检索：
    - 不再去重新从 Drive 拉文档，而是仍然基于同一个 vectordb 做“按关键字”的相似度搜索，
      避免出现“兜底命中了一堆 ACA 医保文章”的情况。
    """
    keywords = extract_keywords_from_question(question)
    if not keywords:
        return []

    results: List[Document] = []
    seen = set()

    # 每个关键词单独搜几条，然后合并去重
    for kw in keywords:
        hits = vectordb.similarity_search(kw, k=3)
        for d in hits:
            key = (d.metadata.get("source"), d.metadata.get("page", 0))
            if key in seen:
                continue
            seen.add(key)
            results.append(d)
            if len(results) >= max_hits:
                break
        if len(results) >= max_hits:
            break

    return results


# =============== 4. 对外主接口：回答问题 ===============

def answer_question(
    question: str,
    k: int = 5,
) -> Tuple[str, List[Document], Dict[str, Any]]:
    """
    返回：
    - answer：模型回答
    - docs：最终用于回答的文档片段列表
    - debug_info：调试信息（用于前端显示命中的文档列表）
    """
    llm = _get_llm()
    vectordb = build_or_load_vectorstore(force_rebuild=False)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # 第一步：正常向量检索
    docs = retriever.get_relevant_documents(question)

    # 第二步：关键字兜底 —— 检查现有 docs 是否覆盖了问题中的关键字
    keywords = extract_keywords_from_question(question)
    if keywords:
        joined = "\n".join(d.page_content.lower() for d in docs)
        missing_keywords = [kw for kw in keywords if kw not in joined]

        if missing_keywords:
            extra_docs = keyword_fallback_search(question, vectordb, max_hits=5)
            # 追加兜底命中的片段，注意去重
            existing_keys = set(
                (d.metadata.get("source"), d.metadata.get("page", 0)) for d in docs
            )
            for d in extra_docs:
                key = (d.metadata.get("source"), d.metadata.get("page", 0))
                if key not in existing_keys:
                    docs.append(d)
                    existing_keys.add(key)

    # 构造上下文
    context_parts = []
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "")
        title = d.metadata.get("title", "")
        context_parts.append(
            f"[文档{i+1}] 来源: {src}  标题: {title}\n内容片段:\n{d.page_content[:1200]}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

用户问题：
{question}

下面是从知识库检索到的相关内容（可能来自多个文档的片段）：
{context}

请严格根据上述内容回答问题：
- 如果能找到明确回答，就直接用中文回答，条理清晰，尽量简短。
- 如果文档中没有提到用户想知道的点，就回答：“目前知识库中没有相关信息，请咨询负责人或查看原始文档。”
- 禁止凭空补充知识库中没有的信息。
"""

    answer = llm.invoke(prompt).content

    debug_info: Dict[str, Any] = {
        "retrieved_sources": [
            {
                "rank": i + 1,
                "source": d.metadata.get("source", ""),
                "title": d.metadata.get("title", ""),
                "modifiedTime": d.metadata.get("modifiedTime", ""),
            }
            for i, d in enumerate(docs)
        ]
    }

    return answer, docs, debug_info


# =============== 5. 管理页面用：文档列表 & 手动重建 ===============

def get_loaded_docs_summary() -> List[Dict[str, Any]]:
    """
    给 Streamlit 管理页面用：返回当前内存中最近一次 load 的文档列表概要。
    注意：这是最近一次 build_or_load_vectorstore() / force_rebuild 调用时从 Drive 拉到的列表。
    """
    summaries: List[Dict[str, Any]] = []
    for d in _last_loaded_docs:
        summaries.append(
            {
                "source": d.metadata.get("source", ""),
                "title": d.metadata.get("title", ""),
                "modifiedTime": d.metadata.get("modifiedTime", ""),
                "chars": len(d.page_content),
            }
        )
    return summaries


def force_rebuild_vectorstore() -> Dict[str, Any]:
    """
    给管理页面用：手动点按钮触发重建。
    返回当前 index 的 meta 信息，方便展示。
    """
    _ = build_or_load_vectorstore(force_rebuild=True)
    meta = _load_index_metadata()
    return meta


if __name__ == "__main__":
    # 本地简单测试用
    tests = [
        "Ho5保单哪个公司能做？",
        "EPLI 主要保什么？",
    ]
    for q in tests:
        print("\n==============================")
        print("问题：", q)
        ans, ds, dbg = answer_question(q)
        print("回答：", ans)
        print("引用的文档片段数量：", len(ds))
        print("命中文档：", dbg.get("retrieved_sources"))
