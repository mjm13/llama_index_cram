from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SummaryIndex,VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import FunctionTool
import time

def add(x: int, y: int) -> int:
    """两个数相加"""
    return x + y
def mystery(x: int, y: int) -> int:
    """两个数相乘"""
    return y * x
add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

start_time = time.time()
llm =  Ollama(model="llama3.1:latest",temperature=0.6,request_timeout=10000.0)
response = llm.predict_and_call(
    [add_tool, mystery_tool],
    "告诉我2,5这两个参数通过mystery函数执行的结果",
    verbose=True
)
print(str(response))
end_time = time.time()
print(f"耗时: {end_time - start_time} 秒")


#  部分模型不支持 tools,需查看模型介绍页面是否有Tools 标签
# 中文英文看起来都能正常调用