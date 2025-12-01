from langchain_community.document_loaders import GoogleDriveLoader

# 这里改成你知识库文件夹的 ID
# 比如："1APR5Cl2nSabOv43CH5qfVrNKWULcKFm8"
FOLDER_ID = "1APR5Cl2nSabOv43CH5qfVrNKWULcKFm8"


def load_documents_from_drive():
    """从 Google Drive 知识库文件夹读取所有文档，返回 LangChain Document 列表。"""
    loader = GoogleDriveLoader(
        folder_id=FOLDER_ID,
        recursive=True,              # 递归子文件夹（如果有）
        include_comments=False,
        credentials_path="credentials.json",
        token_path="token.json",
    )
    docs = loader.load()
    print(f"从 Drive 加载到文档数量：{len(docs)}")
    return docs


def main():
    # 单独运行 load_kb.py 时测试用
    load_documents_from_drive()


if __name__ == "__main__":
    main()

