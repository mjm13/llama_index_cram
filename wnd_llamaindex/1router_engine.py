from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SummaryIndex,VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
import time

start_time = time.time()
Settings.llm = Ollama(model="qwen2:1.5b",temperature=0.3,request_timeout=10000.0)
# Settings.llm = Ollama(model="gemma2:2b",temperature=0.6,request_timeout=10000.0)
Settings.embed_model = OllamaEmbedding(model_name="sunzhiyuan/suntray-embedding")

documents = SimpleDirectoryReader(input_files=["D:\Java.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
# TODO 创建索引之后如何存储索引
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "用于对问题结合文档进行总结"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "用于从文档中检索特定上下文"
    ),
)

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

response = query_engine.query("这个文档是干嘛的?")
print(str(response))
print(len(response.source_nodes))
response = query_engine.query(
    "为什么建议开发者谨慎使用继承?这一章讲的是什么"
)
print(str(response))
print(len(response.source_nodes))
end_time = time.time()
print(f"耗时: {end_time - start_time} 秒")

# 使用低级模型无法区分
# Selecting query engine 1: 从文档中检索特定上下文.
# 这个文档是关于Java开发手册，主要介绍了如何避免重复代码、日志规约以及维护和归类查找日志文件的方法。
# 2
# Selecting query engine 1: 从文档中检索特定上下文.
# 这个回答是正确的，因为根据给定的上下文信息，它提到了谨慎使用继承的原因。这可能意味着在某些情况下，使用继承可能会导致代码难以维护和扩展。因此，建议开发者谨慎使用继承以确保系统设计的可维护性和可扩展性。
# 2
# 耗时: 120.88532066345215 秒