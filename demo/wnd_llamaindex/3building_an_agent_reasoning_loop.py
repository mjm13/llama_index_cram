from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
# --------------------------
agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool], 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)
print(response.source_nodes[0].get_content(metadata_mode="all"))
response = agent.chat(
    "Tell me about the evaluation datasets used."
)
response = agent.chat("Tell me the results over one of the above datasets.")