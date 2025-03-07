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
