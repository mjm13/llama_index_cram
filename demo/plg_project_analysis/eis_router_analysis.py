import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorIndexManager:
    def __init__(
            self,
            source_dir: str,
            index_dir: str = "./vector_index",
            file_extensions: list[str] = ['.java', '.xml', '.yml', '.properties'],
            llm_model: str = "llama3.1:latest",
            embedding_model: str = "sunzhiyuan/suntray-embedding"
    ):
        """
        初始化向量索引管理器

        Args:
            source_dir: 源代码目录
            index_dir: 索引存储目录
            file_extensions: 要索引的文件扩展名
            llm_model: LLM模型名称
            embedding_model: 嵌入模型名称
        """
        self.source_dir = Path(source_dir)
        self.index_dir = Path(index_dir)
        self.file_extensions = file_extensions

        # 验证目录
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        # 配置Settings
        try:
            Settings.llm = Ollama(
                model=llm_model,
                request_timeout=60.0,
                context_window=32000,
                system_prompt="""
                你是一个专业的架构师，接下来的问题请以架构师的角度回答，
                如果涉及到代码输出，则代码要充分利用异步非阻塞特性，
                注释丰富，多打印日志，容错性好，对外部输入进行充分校验。
                """
            )
            Settings.embed_model = OllamaEmbedding(model_name=embedding_model)
            logger.info("Successfully initialized LLM and embedding models")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def _create_index(self) -> VectorStoreIndex:
        """创建新的向量索引"""
        logger.info(f"Creating new index from {self.source_dir}")
        try:
            documents = SimpleDirectoryReader(
                input_dir=str(self.source_dir),
                recursive=True,
                required_exts=self.file_extensions
            ).load_data()
            logger.info(f"Loaded {len(documents)} documents")

            index = VectorStoreIndex.from_documents(documents)

            # 确保索引目录存在
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # 保存索引
            index.storage_context.persist(persist_dir=str(self.index_dir))
            logger.info(f"Successfully saved index to {self.index_dir}")

            return index

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def _load_index(self) -> Optional[VectorStoreIndex]:
        """从本地加载索引"""
        try:
            if self.index_dir.exists():
                logger.info(f"Loading existing index from {self.index_dir}")
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.index_dir)
                )
                return load_index_from_storage(storage_context)
            return None
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return None

    def get_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        获取向量索引，优先从本地加载

        Args:
            force_rebuild: 是否强制重建索引

        Returns:
            VectorStoreIndex: 向量索引对象
        """
        if not force_rebuild:
            existing_index = self._load_index()
            if existing_index is not None:
                return existing_index

        return self._create_index()

    def query(self, query_text: str) -> str:
        """
        查询索引

        Args:
            query_text: 查询文本

        Returns:
            str: 查询结果
        """
        try:
            logger.info(f"Querying index with: {query_text}")
            index = self.get_index()
            query_engine = index.as_query_engine()
            response = query_engine.query(query_text)
            logger.info("Query completed successfully")
            return str(response)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise


def main():
    # 使用示例
    try:
        manager = VectorIndexManager(
            source_dir="D:/Project/PLG/202410/eis-router-center-parent",
            index_dir="D:/Project/PLG/202410/eis-router-center-parent_vector-index"
        )

        # 查询项目功能
        response = manager.query("这个项目都有什么功能？")
        print(response)

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()