import asyncio
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

# 1. 创建需求分析链 - 先明确需求细节
requirement_analyzer = (
    ChatPromptTemplate.from_template("""
        请澄清并详细说明以下编程需求，确保理解所有细节：
        原始需求: {input}

        请输出:
        1. 功能的核心目标
        2. 必要的输入输出
        3. 边界条件和异常情况
        4. 性能考虑(如有)
    """)
    | ChatOllama(model="deepseek-coder-v2:16b")
    | StrOutputParser()
    | {"analyzed_requirement": RunnablePassthrough()}
)

# 2. 创建Python代码生成链
python_generator = (
    ChatPromptTemplate.from_template("""
        根据以下详细需求生成Python代码实现:
        {analyzed_requirement}
        
        要求:
        - 包含完整可运行的代码
        - 添加适当注释
        - 包含示例用法
        - 遵循PEP8规范
    
        只返回代码，不要解释。
    """)
    | ChatOllama(model="deepseek-coder-v2:16b")
    | StrOutputParser()
)

# 3. 创建Java代码生成链
java_generator = (
    ChatPromptTemplate.from_template("""
        根据以下详细需求生成Java代码实现:
        {analyzed_requirement}
        
        要求:
        - 包含完整可运行的类和方法
        - 添加适当注释
        - 包含main方法示例
        - 遵循Java编码规范
        
        只返回代码，不要解释。
    """)
    | ChatOllama(model="deepseek-coder-v2:16b")
    | StrOutputParser()
)

# 4. 创建代码审查链
code_reviewer = (
    ChatPromptTemplate.from_messages([
        ("system", "你是一个资深代码审查员"),
        ("user", """
            Python实现:
            {python_code}
            
            Java实现:
            {java_code}
            
            请检查两种实现是否:
            1. 功能等价
            2. 满足原始需求
            3. 没有明显错误
            指出任何不一致之处。
        """)
    ])
    | ChatOllama(model="deepseek-coder-v2:16b")
    | StrOutputParser()
)

# 5. 构建完整链
chain = (
    requirement_analyzer
    | {
        "python_code": python_generator,
        "java_code": java_generator,
        "original_requirement": itemgetter("analyzed_requirement"),
    }
    | {
        "final_python": itemgetter("python_code"),
        "final_java": itemgetter("java_code"),
        "review_comments": code_reviewer,
    }
)

print("多链代码生成器 (Python & Java)")
result = chain.invoke({"input": "实现一个函数，计算两个矩阵的乘积"})
print("\n=== Python 实现 ===")
print(result["final_python"])

print("\n=== Java 实现 ===")
print(result["final_java"])

print("\n=== 代码审查意见 ===")
print(result["review_comments"])