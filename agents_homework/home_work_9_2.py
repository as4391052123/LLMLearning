import os
from typing import Annotated, List, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from langchain_core.messages import ToolMessage, AIMessage
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

# 定义状态
class State(TypedDict):
    """定义状态类型，包含消息列表"""
    messages: Annotated[List[dict], add_messages]  # 使用 add_messages 函数将消息追加到现有列表

def setup_environment():
    """设置环境变量，用于 LangSmith 跟踪和调试"""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "ChatBot"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

def create_chat_model():
    """创建并返回一个 GPT-4o-mini 模型实例"""
    return ChatOpenAI(model="gpt-4o-mini")

def create_tool():
    """创建并返回一个 Tavily 搜索工具实例，最大搜索结果数设置为 2"""
    return TavilySearchResults(max_results=2)

def create_graph_builder(state_type):
    """创建并返回一个状态图对象，传入状态定义"""
    return StateGraph(state_type)

class BasicToolNode:
    """一个在最后一条 AIMessage 中执行工具请求的节点"""

    def __init__(self, tools_by_name):
        """初始化工具节点，传入工具名称和工具实例的映射"""
        self.tools_by_name = tools_by_name

    def __call__(self, inputs: dict):
        """执行工具调用

        参数:
        inputs: 包含 "messages" 键的字典，"messages" 是对话消息的列表，
                其中最后一条消息可能包含工具调用的请求。

        返回:
        包含工具调用结果的消息列表
        """
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("输入中未找到消息")

        last_message = messages[-1]
        outputs = []
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                try:
                    # 根据工具名称找到相应的工具，并调用工具的 invoke 方法执行工具
                    tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                    outputs.append(
                        ToolMessage(
                            content=json.dumps(tool_result),  # 工具调用的结果以 JSON 格式保存
                            name=tool_call["name"],  # 工具的名称
                            tool_call_id=tool_call["id"],  # 工具调用的唯一标识符
                        )
                    )
                except Exception as e:
                    logging.error(f"工具调用失败: {e}")
                    outputs.append(
                        ToolMessage(
                            content=json.dumps({"error": str(e)}),  # 记录错误信息
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
        return {"messages": outputs}


def create_basic_tool_node(tools):
    """创建并返回一个 BasicToolNode 实例，传入工具列表"""
    tools_by_name = {tool.name: tool for tool in tools}
    return BasicToolNode(tools_by_name)

def route_tools(state: State) -> Literal["tools", "__end__"]:
    """
    使用条件边来检查最后一条消息中是否有工具调用。

    参数:
    state: 状态字典或消息列表，用于存储当前对话的状态和消息。

    返回:
    如果最后一条消息包含工具调用，返回 "tools" 节点，表示需要执行工具调用；
    否则返回 "__end__"，表示直接结束流程。
    """
    # 检查状态是否是列表类型（即消息列表），取最后一条 AI 消息
    if isinstance(state, list):
        ai_message = state[-1]
    # 否则从状态字典中获取 "messages" 键，取最后一条消息
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    # 如果没有找到消息，则抛出异常
    else:
        raise ValueError(f"输入状态中未找到消息: {state}")

    # 检查最后一条消息是否有工具调用请求
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"  # 如果有工具调用请求，返回 "tools" 节点
    return "__end__"  # 否则返回 "__end__"，流程结束

def chat_with_tool():
    """主函数，用于初始化和运行聊天机器人"""
    setup_environment()  # 设置环境变量

    chat_model = create_chat_model()  # 创建聊天模型
    tool = create_tool()  # 创建工具
    tools = [tool]
    llm_with_tools = chat_model.bind_tools(tools)  # 绑定工具到模型

    graph_builder = create_graph_builder(State)  # 创建状态图

    def chatbot(state: State):
        """聊天机器人节点函数，支持工具调用"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)  # 添加聊天机器人节点

    tool_node = create_basic_tool_node(tools)  # 创建工具节点
    graph_builder.add_node("tools", tool_node)  # 添加工具节点

    # 添加条件边，判断是否需要调用工具
    graph_builder.add_conditional_edges(
        "chatbot",  # 从聊天机器人节点开始
        route_tools,  # 路由函数，决定下一个节点
        {
            "tools": "tools",
            "__end__": "__end__"
        },  # 定义条件的输出，工具调用走 "tools"，否则走 "__end__"
    )

    # 当工具调用完成后，返回到聊天机器人节点以继续对话
    graph_builder.add_edge("tools", "chatbot")

    # 指定从 START 节点开始，进入聊天机器人节点
    graph_builder.add_edge(START, "chatbot")
    # 编译状态图，生成可执行的流程图
    graph = graph_builder.compile()

    # 生成状态图的可视化图像
    try:
        from IPython.display import Image, display
        from PIL import Image as PILImage
        import io

        # 生成图像
        image_data = graph.get_graph().draw_mermaid_png()

        # 将图像数据转换为 PIL Image 对象
        image = PILImage.open(io.BytesIO(image_data))

        # 保存图像到文件
        output_path = 'output_image.png'
        image.save(output_path)

        # 在 Jupyter Notebook 中显示保存的图像
        display(Image(filename=output_path))
    except Exception as e:
        logging.error(f"生成图像失败: {e}")

    # 开始一个简单的聊天循环
    while True:
        # 获取用户输入
        user_input = input("User: ")

        # 可以随时通过输入 "quit"、"exit" 或 "q" 退出聊天循环
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")  # 打印告别信息
            break  # 结束循环，退出聊天

        # 将每次用户输入的内容传递给 graph.stream，用于聊天机器人状态处理
        for event in graph.stream({"messages": [("user", user_input)]}):
            # 遍历每个事件的值
            for value in event.values():
                # 打印输出 chatbot 生成的最新消息
                print("Assistant:", value["messages"][-1].content)

if __name__ == '__main__':
    chat_with_tool()  # 运行主函数
