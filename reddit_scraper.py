# from typing import List
# import os
# from utils import *
# from typing import List
# import asyncio
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.prebuilt import create_react_agent
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os
# from tenacity import (
#     retry, 
#     stop_after_attempt,
#     wait_exponential,
#     retry_if_exception_type
# )
# from aiolimiter import AsyncLimiter
# from tenacity import retry, stop_after_attempt, wait_exponential
# from datetime import datetime, timedelta


# load_dotenv()


# two_weeks_ago = datetime.today() - timedelta(days=14) 
# two_weeks_ago_str = two_weeks_ago.strftime('%Y-%m-%d')


# class MCPOverloadedError(Exception):
#     pass


# mcp_limiter = AsyncLimiter(1, 15)

# model = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0,
#     max_tokens=1000,
#     api_key=os.getenv("GROQ_API_KEY"),
#     # tool_choice="auto"   # allow only registered tools
# )

# server_params = StdioServerParameters(
#     command="npx",
#     env={
#         "API_TOKEN": os.getenv("API_TOKEN"),
#         "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
#     },
#     args=["@brightdata/mcp"],
# )


# @retry(
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=15, max=60),
#     retry=retry_if_exception_type(MCPOverloadedError),
#     reraise=True
# )
# async def process_topic(agent, topic: str):
#     async with mcp_limiter:
#         messages = [
#             {
#                 "role": "system",
#                 "content": f"""You are a Reddit analysis expert.

#                 IMPORTANT RULES:
#                 - Do NOT explain your reasoning
#                 - Do NOT describe steps you are taking
#                 - Do NOT think out loud
#                 - You may ONLY use the tools explicitly provided to you.
#                 - Your final response must ONLY be the final summary text

#                 Task:
#                 Analyze Reddit posts about the topic AFTER {two_weeks_ago_str}.
#                 Return a clean summary only."""
#             },
#             {
#                 "role": "user",
#                 "content": f"""Analyze Reddit posts about '{topic}'. 
#                 Provide a comprehensive summary including:
#                 - Main discussion points
#                 - Key opinions expressed
#                 - Any notable trends or patterns
#                 - Summarize the overall narrative, discussion points and also quote interesting comments without mentioning names
#                 - Overall sentiment (positive/neutral/negative)"""
#             }                   
#         ]
        
#         try:
#             response = await agent.ainvoke({"messages": messages})
#             return response["messages"][-1].content
#         except Exception as e:
#             if "Overloaded" in str(e):
#                 raise MCPOverloadedError("Service overloaded")
#             else:
#                 raise



# async def scrape_reddit_topics(topics: List[str]) -> dict[str, dict]:
#     """Process list of topics and return analysis results"""
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()
#             tools = await load_mcp_tools(session)
#             agent = create_react_agent(model, tools)
            
#             reddit_results = {}
#             for topic in topics:
#                 summary = await process_topic(agent, topic)
#                 reddit_results[topic] = summary
#                 await asyncio.sleep(5)  # Maintain rate limiting
                
#             return {"reddit_analysis": reddit_results}

# The below code is designed with the langchain mcp client way which is easier and efficent 
# then the pythonic sdk way of calling mcp

# # try commenting the above code and uncommenting the below code and may be some libraries 
# # to be installed

#Older implementations often allowed implicit fallback to the LLM’s knowledge which means
# if the tools failed, the LLM would just try to answer based on its training data, which could lead to hallucinations.
# In contrast, the LangChain approach enforces explicit tool execution. If a tool fails,
# when tools failed, whereas LangChain enforces explicit tool execution, 
# failing fast instead of hallucinating answers.    
# so in the below code it sometimes can happen the if tool call is failed due to the reason like 
# proxy error, network error the code will fails
# because the retry logic is only implemented for the MCPOverloadedError         
from typing import List, Dict
import os
import asyncio
from datetime import datetime, timedelta

from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# -------------------------------------------------------------------
# ENV SETUP
# -------------------------------------------------------------------
load_dotenv()

two_weeks_ago = datetime.today() - timedelta(days=14)
two_weeks_ago_str = two_weeks_ago.strftime("%Y-%m-%d")

# -------------------------------------------------------------------
# CUSTOM ERROR
# -------------------------------------------------------------------
class MCPOverloadedError(Exception):
    pass

# -------------------------------------------------------------------
# RATE LIMITING
# -------------------------------------------------------------------
mcp_limiter = AsyncLimiter(1, 15)

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1000,
    api_key=os.getenv("GROQ_API_KEY"),
)

# -------------------------------------------------------------------
# MCP CLIENT (LANGCHAIN WAY)
# -------------------------------------------------------------------
client = MultiServerMCPClient(
    {
        "reddit": {
            "command": "npx",
            "args": ["@brightdata/mcp"],
            "transport": "stdio",
            "env": {
                "API_TOKEN": os.getenv("API_TOKEN"),
                "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
            },
        }
    }
)

# -------------------------------------------------------------------
# PROCESS SINGLE TOPIC
# -------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=15, max=60),
    retry=retry_if_exception_type(MCPOverloadedError),
    reraise=True,
)
async def process_topic(agent, topic: str) -> str:
    async with mcp_limiter:
        messages = [
            {
                "role": "system",
                "content": f"""
You are a Reddit analysis expert.

IMPORTANT RULES:
- Do NOT explain your reasoning
- Do NOT describe steps you are taking
- Do NOT think out loud
- You may ONLY use the reddit specific tools.
- Your final response must ONLY be the final summary text

Task:
Analyze Reddit posts about the topic AFTER {two_weeks_ago_str}.
Return a clean summary only.
"""
            },
            {
                "role": "user",
                "content": f"""
Analyze Reddit posts about '{topic}'.
Provide:
- Main discussion points
- Key opinions
- Notable trends
- Quote interesting comments (no usernames)
- Overall sentiment (positive/neutral/negative)
"""
            },
        ]

        try:
            response = await agent.ainvoke({"messages": messages})
            return response["messages"][-1].content
        except Exception as e:
            if "Overloaded" in str(e):
                raise MCPOverloadedError("MCP service overloaded")
            raise

# -------------------------------------------------------------------
# SCRAPE MULTIPLE TOPICS
# -------------------------------------------------------------------
async def scrape_reddit_topics(topics: List[str]) -> Dict[str, str]:
    """
    LangChain-style MCP Reddit scraper
    """
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)

    results = {}

    for topic in topics:
        summary = await process_topic(agent, topic)
        results[topic] = summary
        await asyncio.sleep(5)

    return {"reddit_analysis": results}
