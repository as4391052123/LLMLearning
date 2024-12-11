from operator import itemgetter
from tabnanny import verbose
import os
import langchain.globals
from langchain.chains import LLMChain

from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAI


#  基于langchain v0.3.10

def get_llm_response(user_prompt: str, model_name: str = "gpt-4o-mini") -> str:
    """
    使用指定的 LLM 模型生成响应
    """
    # 创建一个 ChatOpenAI 对象
    chat_model = ChatOpenAI(model_name=model_name, temperature=1.0, max_tokens=10, verbose=True)
    chat_model.verbose = True
    messages = [
        SystemMessage("我会问你问题，你给出我想要的答案"),
        HumanMessage(user_prompt)
    ]
    chat_result = chat_model.invoke(input=messages)
    print(chat_result.content)


# 使用 ChatOpenAI 模型生成响应

def get_llm_chains_response(name: str = "小明", hometown: str = "北京", model_name: str = "gpt-4o-mini"):
    chat_model = ChatOpenAI(model_name=model_name, temperature=1.0, max_tokens=100, verbose=True)
    user_prompt = PromptTemplate(
        input_variables=["name", "hometown"],
        template="给{name}介绍一下我的家乡{hometown},50字"
    )
    res = (user_prompt | chat_model).invoke({"name": name, "hometown": hometown})
    print(res.content)


def get_llm_chains_interview_response(resume: str, job_detail: str, model_name: str = "gpt-4o-mini"):
    llm = ChatOpenAI(model_name=model_name, temperature=1.0, max_tokens=10)
    langchain.globals.set_verbose(True)
    template_hr = """
    你是一个专业的HR，你将收到一封简历{resume}和一份岗位信息{job_detail}，请针对需要招聘的岗位信息进行简历筛选，并指出简历的哪些地方不合适，并提供简历修改意见。
    """
    hr_prompt = PromptTemplate(
        input_variables=["resume", "job_detail"],
        template=template_hr
    )
    hr_chain = hr_prompt | llm
    hr_response = hr_chain.invoke({"resume": resume, "job_detail": job_detail})
    hr_comment = hr_response.content

    template_interviewer = """
    你是一个专业的技术总监，你将收到一封简历{resume}和一份岗位信息{job_detail}，请你对简历中的求职者以岗位信息和自己的专业技术能力进行面试，提出10个面试问题，并给出相应的答案。
    """
    interviewer_prompt = PromptTemplate(
        input_variables=["resume", "job_detail"],
        template=template_interviewer
    )
    interviewer_chain = interviewer_prompt | llm
    interviewer_response = interviewer_chain.invoke({"resume": resume, "job_detail": job_detail})
    interviewer_comment = interviewer_response.content

    template_boss = """
    你是人工智能公司的CEO,你将收到一份简历{resume}，同时也收到hr对这份简历的评价和修改意见{hr_comment}和技术总监的面试记录问题{interviewer_comment}，针对这些你对候选人、hr和技术总监分别做出点评，并对候选人进行打分，满分100分，并给出匹配的相关度和评价。
    """
    boss_prompt = PromptTemplate(
        input_variables=["resume", "hr_comment", "interviewer_comment"],
        template=template_boss
    )
    boss_chain = boss_prompt | llm
    boss_response = boss_chain.invoke(
        {"resume": resume, "hr_comment": hr_comment, "interviewer_comment": interviewer_comment})
    boss_comment = boss_response.content

    print("HR 评论:\n", hr_comment)
    print("技术总监面试问题:\n", interviewer_comment)
    print("CEO 评论:\n", boss_comment)


def get_llm_chains_theatre_response(title: str, model_name: str = "gpt-4o-mini"):
    # 设置模型
    # 开启模型debug，输出chains执行过程
    # langchain.debug = True
    # 局部开启
    langchain.globals.set_debug(True)
    llm = ChatOpenAI(model_name=model_name, temperature=1.0, max_tokens=100, verbose=True)
    langchain.globals.set_verbose(True)
    # 第一个链：剧作家生成简介
    template1 = """
      你是一位剧作家。根据戏剧的标题，你的任务是为该标题写一段10字简介。
      标题：{title}
      剧作家：以下是对上述戏剧的简介：
      """
    prompt_template_synopsis = PromptTemplate(input_variables=["title"], template=template1, verbose=True)
    synopsis_chain = prompt_template_synopsis | llm  # 使用 prompt | llm 代替 LLMChain
    # synopsis_chain =LLMChain(llm=llm, prompt=prompt_template_synopsis)
    # 第二个链：戏剧评论家生成评论
    template2 = """
      你是一位《纽约时报》的戏剧评论家。根据剧情简介，你的工作是为该剧撰写一篇10字评论
      剧情简介：{synopsis}
      以下是来自《纽约时报》戏剧评论家对上述剧目的评论：
      """
    prompt_template_review = PromptTemplate(input_variables=["synopsis"], template=template2)
    review_chain = prompt_template_review | llm  # 使用 prompt | llm 代替 LLMChain

    # 创建顺序链
    sequential_chain = synopsis_chain | review_chain  # 使用 | 连接链
    print(f"langchain.globals.get_verbose():{langchain.globals.get_verbose()}")
    # 执行链并显示执行过程
    print("执行顺序链：")

    response = sequential_chain.invoke({"title": title, "verbose": True})
    print(f"最终生成的评论:\n{response.content}")


def get_llm_discuss_response(topic: str, model_name: str = "gpt-4o-mini"):

    langchain.globals.set_debug(True)
    langchain.globals.set_verbose(True)
    planer = (
            ChatPromptTemplate.from_template("生成关于以下内容的论点：{topic}")
            | ChatOpenAI(model_name=model_name, temperature=1.0)
            | StrOutputParser()
            | {"base_response": RunnablePassthrough()}
    )
    # 创建正面观点
    argument_for = (
            ChatPromptTemplate.from_template("列出关于{base_response}的正面或者有利的观点")
            | ChatOpenAI(model_name=model_name, temperature=1.0)
            | StrOutputParser()
    )
    # 创建反面观点
    argument_against = (
            ChatPromptTemplate.from_template("列出关于{base_response}的反面或者不利的观点")
            | ChatOpenAI(model_name=model_name, temperature=1.0)
            | StrOutputParser()
    )
    # 创建最终响应者
    final_responder = (
            ChatPromptTemplate.from_messages(
                [
                    ("ai", "{original_response}"),
                    ("human", "正面观点：\n{results_1}\n\n反面观点:\n{results_2}"),
                    ("system", "给出评价后生成最终回应")
                ]
            )
            | ChatOpenAI(model_name=model_name, temperature=1.0)
            | StrOutputParser()
    )
    chain = (
            planer
            | {
                "results_1": argument_for,
                "results_2": argument_against,
                "original_response": itemgetter("base_response"),
            }
            | final_responder
    )
    chain.invoke({"topic": topic})

if __name__ == '__main__':
    # llm test
    # response = get_llm_response("成都大运会是哪一届？")
    # get_llm_chains_response()
    # resume = """
    #     朱文灿
    #     联系方式: 18618154519 | 854140394@qq.com | 英语: CET-4 | 学历: 安徽工业大学 | 专业: 软件工程
    #     全栈开发工程师，精通Java、Python、Vue.js、小程序，具备微服务架构设计与实施经验。熟悉LangChain、TensorFlow、自然语言处理与深度学习技术，擅长大模型应用开发。具有敏捷开发经验，能带领团队高效交付项目，提升系统稳定性与扩展性。
    #     在华易众欣科技任技术经理，负责“易融气象融媒体平台”的架构设计与微服务实施，成功推动多个产品线产品化，并引入Kubernetes提升运维效率。在爱信诺征信主导数据中台重构，提升数据同步效率30%。在橙天嘉禾设计并实现音乐人平台的后端架构，支持平台的稳定运营。
    #     精通MySQL、Redis、Kubernetes等技术，熟悉云计算与容器化部署。能够快速响应市场需求，推动技术创新与项目优化。
    # """
    # job_detail = """
    # python高级开发工程师
    # 岗位职责：
    #     1. 主导团队大模型应用框架及配套工具链的设计和研发；
    #     2. 主导团队大模型在各垂直领域的创新场景应用并制定系统化解决方案；
    #     3. 主导重要项目的技术攻关及架构调整；
    #     4. 进行系统优化，保证系统的高可用性、性能优良、并发稳定；
    #     5. 与产品团队合作带领后端团队完成业务目标
    # 任职要求：
    #     1. 全日制本科及以上学历，211院校，计算机相关专业，6年以上后端开发经验，精通Python、且了解java、go等语言；
    #     2.熟悉使用Pyhton常用web框架，熟悉Flask、Django、Fastapi等常见框架至少一种；
    #     3.熟悉分布式、缓存、消息队列等机制，如：redis、kafka。了解分布式环境下，稳定性建设方案：trace体系，灰度发布等；
    #     4.熟悉主流大模型（如GPT、Gemini、LLaMA、Claude等）工作原理及应用，熟悉Prompt Engineering，RAG，Fine Tune，熟悉常见的嵌入模型；
    #     5.有Prompt 工程、LangChain、LlamaIndex、Semantic Kernel、AutoGPT、Auto-gen等开源大模型应用开发者优先
    #     6.熟悉常用开发工具及版本管理工具；
    #     7.具有良好的编程习惯和代码规范，思路清晰，注重细节，有良好的文档编写能力；
    #     8.具备较强的问题分析、归纳总结与解决能力，具有优秀的团队沟通能力。
    #     """
    # get_llm_chains_interview_response(resume, job_detail)
    # get_llm_chains_theatre_response("时空行者")
    get_llm_discuss_response("2025年北京房价")
