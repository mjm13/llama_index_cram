import os
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.utils.workflow import draw_all_possible_flows
import configparser
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
    Event
)

config = configparser.ConfigParser()
# 读取配置文件
config.read('../config.ini')

os.environ["GOOGLE_API_KEY"] = config.get('api_key', 'google');
Settings.llm = Gemini(model_name="models/gemini-2.0-flash-exp")
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004")


# 自定义事件
class DocumentLoadedEvent(Event):
    index: VectorStoreIndex


class QueryProcessedEvent(Event):
    response: str


class TextStreamEvent(Event):
    chunk: str


# 创建一个高级问答工作流
class AdvancedGeminiQAWorkflow(Workflow):
    @step
    async def load_data(self, ctx: Context, ev: StartEvent) -> DocumentLoadedEvent:
        ctx.write_event_to_stream(TextStreamEvent(chunk="正在加载文档..."))

        # 加载文档
        documents = SimpleDirectoryReader("./data").load_data()



        index = VectorStoreIndex.from_documents(documents)

        ctx.write_event_to_stream(TextStreamEvent(chunk="文档加载完成，准备处理查询。\n"))
        return DocumentLoadedEvent(index=index)

    @step
    async def process_query(self, ctx: Context, ev: DocumentLoadedEvent) -> QueryProcessedEvent:
        ctx.write_event_to_stream(TextStreamEvent(chunk="正在处理查询...\n"))

        # 获取查询问题
        question = await ctx.get("question")

        # 创建查询引擎
        query_engine = ev.index.as_query_engine(streaming=False)

        # 执行流式查询
        response = query_engine.query(question).response

        return QueryProcessedEvent(response=response)

    @step
    async def finalize(self, ctx: Context, ev: QueryProcessedEvent) -> StopEvent:
        ctx.write_event_to_stream(TextStreamEvent(chunk="\n\n处理完成!"))
        return StopEvent(result=ev.response)


# 使用示例
async def main():
    workflow = AdvancedGeminiQAWorkflow(timeout=60, verbose=True)
    handler = workflow.run()

    # 设置问题
    await handler.ctx.set("question", "解释self.txt文档姓名是什么，做过什么项目?使用中文回答")

    # 处理流式输出
    async for ev in handler.stream_events():
        if isinstance(ev, TextStreamEvent):
            print(ev.chunk, end="", flush=True)

    # 获取最终结果
    final_result = await handler
    print(f"\n\n完整回答: {final_result}")

    # draw_all_possible_flows(
    #     workflow,
    #     filename="workflows/advancedGeminiQAWorkflow.html"
    # )

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())