# 重新设计或在 LangChain Hub 上找一个可用的 RAG 提示词模板，测试对比两者的召回率和生成质量。
from langchain.schema import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from home_work_8_1 import store_documents


# 定义格式化文档的函数
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == '__main__':
    # 加载文档
    vector_store = store_documents("https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # system_prompt = hub.pull("ohkgi/superb_system_instruction_prompt")
    # 定义agent
    system_prompt = """
     You are proficient in reading article summaries. The content of the article you are now reading is {context}. At the same time, you will receive some content and respond based on your understanding of the article content.
    """
    human_prompt = """"
    I have a question for you:
     {question}
    """
    system_message = SystemMessage(system_prompt)
    human_message = HumanMessage(human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message.content),
        ("human", human_message.content)
    ])
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt_question = ""
    custom_rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | chat_prompt
            | llm
            | StrOutputParser()
    )
    # 使用自定义 prompt 生成回答
    res=custom_rag_chain.invoke("What is Multi-Head Self-Attention?")
    print(res)
