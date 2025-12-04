# app.py
import streamlit as st
from rag_qa import (
    answer_question,
    get_loaded_docs_summary,
    force_rebuild_vectorstore,
)

st.set_page_config(page_title="内部知识库助手", page_icon="🤖", layout="wide")

st.title("🤖 公司内部 AI 知识库助手")
st.write("AI 将 **严格根据你上传到 Google Drive 的知识库文档** 来回答问题。")

# 保存对话历史
if "history" not in st.session_state:
    st.session_state["history"] = []

# 问题输入
question = st.text_input("请输入你的问题：", "")

col1, col2 = st.columns([3, 1])
with col1:
    submit = st.button("提交问题", use_container_width=True)
with col2:
    rebuild = st.button("🔄 强制重建知识库向量库", use_container_width=True)

# 手动重建按钮
if rebuild:
    with st.spinner("正在从 Google Drive 重新同步并重建向量库..."):
        meta = force_rebuild_vectorstore()
    st.success("已重建知识库向量库 ✅")
    st.json(meta)

# 提交问题
if submit and question.strip():
    with st.spinner("AI 正在检索知识库并生成回答..."):
        answer, docs, debug_info = answer_question(question)
        st.session_state["history"].append(
            {
                "question": question,
                "answer": answer,
                "docs": docs,
                "debug_info": debug_info,
            }
        )

# ========== 对话记录 ==========
st.subheader("💬 对话记录")

for qa in reversed(st.session_state["history"]):
    st.markdown(f"**🧑‍💼 你问：** {qa['question']}")
    st.markdown(f"**🤖 AI 回答：**\n\n{qa['answer']}")

    # 答案和引用片段之间空一行
    st.markdown("")

    # 展开查看命中文档信息 + 原文片段
    with st.expander("📄 查看本次检索命中的知识库原文片段"):
        debug_info = qa.get("debug_info", {})
        retrieved_sources = debug_info.get("retrieved_sources", [])

        if retrieved_sources:
            st.markdown("**本次命中文档列表：**")
            for src in retrieved_sources:
                st.markdown(
                    f"- **Top {src['rank']}**  "
                    f"来源: `{src['source']}`  "
                    f"标题: {src.get('title','') or '(无标题)' }  "
                    f"修改时间: {src.get('modifiedTime','')}"
                )
        else:
            st.write("（本次没有调试信息）")

        st.markdown("---")
        st.markdown("**具体片段内容：**")

        if qa["docs"]:
            for i, d in enumerate(qa["docs"], start=1):
                source = d.metadata.get("source") or "未命名文件"
                st.markdown(f"**片段 {i} - 来源文件：{source}**")
                st.write(d.page_content)
                st.markdown("---")
        else:
            st.write("本次回答未引用具体文档片段。")

    st.markdown("---")

# ========== 知识库管理 / 调试区域 ==========
st.subheader("🛠 知识库管理 / 调试")

with st.expander("📚 当前已加载的知识库文档列表"):
    docs_summary = get_loaded_docs_summary()
    st.write(f"当前内存中已加载文档数量：**{len(docs_summary)}**")

    if not docs_summary:
        st.caption("暂时还没有加载文档（等第一次检索或点击重建后会出现）。")
    else:
        for i, d in enumerate(docs_summary, start=1):
            st.markdown(
                f"{i}. **{d['title'] or '(无标题)'}**  \n"
                f"`source`: `{d['source']}`  \n"
                f"`修改时间`: {d['modifiedTime']}  \n"
                f"`字符数`: {d['chars']}"
            )
            st.markdown("---")
