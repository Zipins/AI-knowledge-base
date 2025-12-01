import streamlit as st
from rag_qa import answer_question

st.set_page_config(page_title="内部知识库助手", page_icon="🤖", layout="wide")

st.title("🤖 公司内部 AI 知识库助手")
st.write("AI 将 **严格根据你上传到 Google Drive 的知识库文档** 来回答问题。")

if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.text_input("请输入你的问题：", "")

if st.button("提交问题"):
    if question.strip():
        with st.spinner("AI 正在检索知识库并生成回答..."):
            answer, docs = answer_question(question)
            st.session_state["history"].append(
                {"question": question, "answer": answer, "docs": docs}
            )

st.markdown("---")

for qa in reversed(st.session_state["history"]):
    st.markdown(f"### 🧑‍💼 你问：{qa['question']}")
    st.markdown(f"**🤖 AI 回答：**\n\n{qa['answer']}")

    with st.expander("📄 查看引用的知识库原文片段"):
        for i, d in enumerate(qa["docs"], start=1):
            source = d.metadata.get("source") or "未命名文件"
            st.markdown(f"**片段 {i} - 来源文件：{source}**")
            st.write(d.page_content)
    st.markdown("---")
