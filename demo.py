# 扫描本地代码
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama

ollama_embedding = OllamaEmbedding(
model_name="mxbai-embed-large:latest",
base_url="http://10.6.6.240:11435",
)
llm = Ollama(model="deepseek-coder-v2:16b-lite-instruct-fp16", request_timeout=240.0, base_url='http://10.6.6.18:9090',
context_window=32000,
system_prompt="你是一个专业的架构师，接下来的问题请以架构师的角度回答，如果涉及到代码输出，则代码要充分利用异步非阻塞特性，注释丰富，多打印日志，容错性好，对外部输入进行充分校验。")

Settings.llm = llm
Settings.embed_model = ollama_embedding
documents = SimpleDirectoryReader(input_dir="代码目录", recursive=True, required_exts=['.扩展名']).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("这个项目都有什么功能？")

print(response)