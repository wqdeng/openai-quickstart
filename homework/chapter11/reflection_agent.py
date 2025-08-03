import asyncio
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a writing assistant tasked with creating well-crafted, coherent, and engaging articles based on the user's request."
            " Focus on clarity, structure, and quality to produce the best possible piece of writing."
            " If the user provides feedback or suggestions, revise and improve the writing to better align with their expectations.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

writer = writer_prompt | ChatOllama(
    model="llama3.1:8b",
    max_tokens=8192,
    temperature=1.2,
)

# article = ""
#
# topic = HumanMessage(
#     content="参考水浒传的风格，改写吴承恩的西游记中任意篇章"
# )
#
# for chunk in writer.stream({"messages": [topic]}):
#     print(chunk.content, end="")
#     article += chunk.content
#
# # 使用Markdown显示优化后的格式
# print(article)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a reviewer responsible for providing constructive feedback and improvement suggestions for user-submitted content. "
            "Please provide a detailed evaluation, including suggestions regarding clarity, structure, content depth, and style, as well as specific areas that need improvement."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect = reflection_prompt | ChatOllama(
    model="llama3.1:8b",
    max_tokens=8192,
    temperature=0.2,
)

# reflection = ""
#
# # 将主题（topic）和生成的文章（article）作为输入发送给反思智能体
# for chunk in reflect.stream({"messages": [topic, HumanMessage(content=article)]}):
#     print(chunk.content, end="")
#     reflection += chunk.content
#
# # 使用Markdown显示优化后的格式
# print(reflection)


# 定义状态类，使用TypedDict以保存消息
class State(TypedDict):
    messages: Annotated[list, add_messages]  # 使用注解确保消息列表使用add_messages方法处理

# 异步生成节点函数：生成内容（如作文）
# 输入状态，输出包含新生成消息的状态
async def generation_node(state: State) -> State:
    # 调用生成器(writer)，并将消息存储到新的状态中返回
    return {"messages": [await writer.ainvoke(state['messages'])]}

# 异步反思节点函数：对生成的内容进行反思和反馈
# 输入状态，输出带有反思反馈的状态
async def reflection_node(state: State) -> State:
    # 创建一个消息类型映射，ai消息映射为HumanMessage，human消息映射为AIMessage
    cls_map = {"ai": HumanMessage, "human": AIMessage}

    # 处理消息，保持用户的原始请求（第一个消息），转换其余消息的类型
    translated = [state['messages'][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state['messages'][1:]
    ]

    # 调用反思器(reflect)，将转换后的消息传入，获取反思结果
    res = await reflect.ainvoke(translated)

    # 返回新的状态，其中包含反思后的消息
    return {"messages": [HumanMessage(content=res.content)]}

MAX_ROUND = 6

# 定义条件函数，决定是否继续反思过程
# 如果消息数量超过6条，则终止流程
def should_continue(state: State):
    if len(state["messages"]) > MAX_ROUND:
        return END  # 达到条件时，流程结束
    return "reflect"  # 否则继续进入反思节点

# 创建状态图，传入初始状态结构
builder = StateGraph(State)

# 在状态图中添加"writer"节点，节点负责生成内容
builder.add_node("writer", generation_node)

# 在状态图中添加"reflect"节点，节点负责生成反思反馈
builder.add_node("reflect", reflection_node)

# 定义起始状态到"writer"节点的边，从起点开始调用生成器
builder.add_edge(START, "writer")


# 在"writer"节点和"reflect"节点之间添加条件边
# 判断是否需要继续反思，或者结束
builder.add_conditional_edges("writer", should_continue)

# 添加从"reflect"节点回到"writer"节点的边，进行反复的生成-反思循环
builder.add_edge("reflect", "writer")

# 创建内存保存机制，允许在流程中保存中间状态和检查点
memory = MemorySaver()

# 编译状态图，使用检查点机制
graph = builder.compile(checkpointer=memory)

# 定义装饰器，记录函数调用次数
def track_steps(func):
    step_counter = {'count': 0}  # 用于记录调用次数

    def wrapper(event, *args, **kwargs):
        # 增加调用次数
        step_counter['count'] += 1
        # 在函数调用之前打印 step
        print(f"## Round {step_counter['count']}")
        # 调用原始函数
        return func(event, *args, **kwargs)

    return wrapper

# 使用装饰器装饰 pretty_print_event_markdown 函数
@track_steps
def pretty_print_event_markdown(event):
    # 如果是生成写作部分
    if 'writer' in event:
        generate_md = "#### 生成:\n"
        for message in event['writer']['messages']:
            generate_md += f"- {message.content}\n"
        print(generate_md)

    # 如果是反思评论部分
    if 'reflect' in event:
        reflect_md = "#### 评论反思:\n"
        for message in event['reflect']['messages']:
            reflect_md += f"- {message.content}\n"
        print(reflect_md)

inputs = {
    "messages": [
        HumanMessage(content="使用 Java 实现一个函数，计算两个矩阵的乘积")
    ],
}

config = {"configurable": {"thread_id": "1"}}

# 假设 graph.astream(inputs, config) 返回一个异步生成器
async def main():
    async for event in graph.astream(inputs, config):
        pretty_print_event_markdown(event)

# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(main())
