
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results = 2,tavily_api_key="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")

tools = [tool]

result = tool.invoke("What's a 'node' in LangGraph?")

print(result)