from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.1:latest", request_timeout=60.0)
ollama_embedding = OllamaEmbedding(model_name="sunzhiyuan/suntray-embedding")

response = llm.complete("中国的首都是哪里?")
print(response)


