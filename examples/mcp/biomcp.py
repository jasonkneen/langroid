"""
Simple example of using the BioMCP server.

https://github.com/genomoncology/biomcp

The server offers several tools, but here we only use the ArticleSearchTool,
to illustrate how dead-simple the Langroid-MCP integration is.

Run like this:

    uv run examples/mcp/biomcp.py --model gpt-4.1-mini

"""

import langroid as lr
import langroid.language_models as lm
from langroid.mytypes import NonToolAction
from langroid.agent.tools.mcp.fastmcp_client import get_langroid_tool_async
from fastmcp.client.transports import UvxStdioTransport, StdioTransport
from fire import Fire


async def main(model: str = ""):
    transport = StdioTransport(
        command="uv",
        args=["run", "--with", "biomcp-python", "biomcp", "run"]
    )
    ArticleSearchTool = await get_langroid_tool_async(transport, "article_searcher")
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use the tool
    agent.enable_message(ArticleSearchTool)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    task = lr.Task(agent, interactive=False)
    await task.run_async()


if __name__ == "__main__":
    Fire(main)
