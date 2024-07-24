import os

from langchain.chains.llm_math.base import LLMMathChain
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain_community.chat_models.azureml_endpoint import (LlamaChatContentFormatter, AzureMLChatOnlineEndpoint)
from langchain.tools import BaseTool

import chainlit as cl
from chainlit.sync import run_sync
from langchain_community.llms.azureml_endpoint import AzureMLEndpointApiType

import langchain

langchain.debug = True


class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "human"
    description = (
        "You can ask a human for guidance when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(
            self,
            query: str,
            run_manager=None,
    ) -> str:
        """Use the Human input tool."""

        res = run_sync(cl.AskUserMessage(content=query).send())
        return res["content"]

    async def _arun(
            self,
            query: str,
            run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        res = await cl.AskUserMessage(content=query).send()
        return res["output"]


@cl.on_chat_start
def start():
    llm = AzureMLChatOnlineEndpoint(
        endpoint_url='https://Meta-Llama-3-1-8B-Instruct-uswvd.eastus2.models.ai.azure.com/v1/chat/completions',
        endpoint_type='serverless',
        endpoint_api_type=AzureMLEndpointApiType.serverless,
        endpoint_api_key=os.environ['AZURE_AI_KEY'],
        content_formatter=LlamaChatContentFormatter(),
        model_kwargs={"temperature": 0.8, "max_new_tokens": 4000},
    )
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        HumanInputChainlit(),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
            coroutine=llm_math_chain.arun,
        ),
    ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
    )

    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    res = await agent.arun(
        message.content, callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    await cl.Message(content=res).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)