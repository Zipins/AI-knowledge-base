import streamlit as st
from rag_qa import answer_question
import os
import streamlit as st

if st.button("🔄 强制更新知识库"):
    os.environ["KB_MODE"] = "refresh"
    from rag_qa import get_vectordb
    get_vectordb()
    os.environ["KB_MODE"] = "runtime"
    st.success("知识库已成功更新！")
st.set_page_config(page_title="内部知识库助手", page_icon="🤖", layout="wide")

st.title("🤖 公司内部 AI 知识库助手")
st.write("AI 将 **严格根据你上传到 Google Drive 的知识库文档** 来回答问题。")

# 保存对话历史
if "history" not in st.session_state:
    st.session_state["history"] = []

# 问题输入
question = st.text_input("请输入你的问题：", "")

if st.button("提交问题"):
    if question.strip():
        with st.spinner("AI 正在检索知识库并生成回答..."):
            answer, docs = answer_question(question)
            st.session_state["history"].append(
                {"question": question, "answer": answer, "docs": docs}
            )

st.markdown("---")

# 反向展示历史（最近的在最上面）
for qa in reversed(st.session_state["history"]):
    st.markdown(f"### 🧑‍💼 你问：{qa['question']}")

    # 先打印“AI 回答”这个标题
    st.markdown("**🤖 AI 回答：**")

    # 如果没有命中任何知识库文档，用红色提示
    if not qa["docs"]:
        st.error("目前知识库中没有相关信息，请在Google Docs里加上此问题，并截图发给同事，获得答案后记得更新哦。")
    else:
        # 正常情况就显示回答内容
        st.markdown(qa["answer"])

    # 答案和引用片段之间空一行
    st.markdown("")

    # 展开查看引用片段
    with st.expander("📄 查看引用的知识库原文片段"):
        if qa["docs"]:
            for i, d in enumerate(qa["docs"], start=1):
                source = d.metadata.get("source") or "未命名文件"
                st.markdown(f"**片段 {i} - 来源文件：{source}**")
                st.write(d.page_content)
        else:
            st.write("本次回答未引用具体文档片段。")

    st.markdown("---")
