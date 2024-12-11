# 作业1 使用其他的线上文档或离线文件，重新构建向量数据库，尝试提出3个相关问题，测试 LCEL 构建的 RAG Chain 是否能成功召回。
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def store_documents(url):
    """
    加载url并将文档存储到chroma向量数据库中
    :param url: 文档链接
    :return: 文档保存到向量数据库的对象
    """
    # 使用 WebBaseLoader 从网页加载内容，并仅保留标题、标题头和文章内容
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()
    # 检查加载的文档内容长度
    print(len(docs[0].page_content))  # 打印第一个文档内容的长度
    # 查看第一个文档（前100字符）
    print(docs[0].page_content[:100])
    # 使用 RecursiveCharacterTextSplitter 将文档分割成块，每块1000字符，重叠200字符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    # 检查分割后的块数量和内容
    print(len(all_splits))  # 打印分割后的文档块数量
    print(len(all_splits[0].page_content))  # 打印第一个块的字符数
    print(all_splits[0].page_content)  # 打印第一个块的内容
    print(all_splits[0].metadata)  # 打印第一个块的元数据

    return Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings()
    )


def query_documents(retriever, query):
    retrieved_docs = retriever.invoke(query)
    print(len(retrieved_docs))  # 打印检索到的文档数量
    print(retrieved_docs[0].page_content)  # 打印第一个文档的内容
    print(retrieved_docs[0].metadata)  # 打印第一个文档的元数据
    return retrieved_docs


if __name__ == '__main__':
    vector_store = store_documents("https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/")
    # 使用 VectorStoreRetriever 从向量存储中检索与查询最相关的文档
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retrieved_docs = query_documents(retriever, "What is Multi-Head Self-Attention?")
    retrieved_docs2 = query_documents(retriever, "How does the transformer achieve context memory?")
    retrieved_docs3 = query_documents(retriever, "How does Transformer handle distance enhancement?")
    print("********Q1")
    print(len(retrieved_docs))  # 打印检索到的文档数量
    print(retrieved_docs[0].page_content)  # 打印第+一个检索到的文档内容
    print("********Q2")
    print(len(retrieved_docs2))  # 打印检索到的文档数量
    print(retrieved_docs2[0].page_content)  # 打印第二个检索到的文档内容
    print("********Q3")
    print(len(retrieved_docs3))  # 打印检索到的文档数量
    print(retrieved_docs3[0].page_content)  # 打印第三个检索到的文档内容
