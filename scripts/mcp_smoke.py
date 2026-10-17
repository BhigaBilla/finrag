"""Start the MCP server over stdio as a real client would, and call each tool."""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("tools:", [t.name for t in (await session.list_tools()).tools])
            for name, args in [("list_reports", {}),
                               ("search_filings", {"query": "voluntary attrition", "company": "TCS",
                                                   "fy": "FY2026", "k": 3}),
                               ("get_page", {"company": "TCS", "fy": "FY2026", "page": 56})]:
                res = await session.call_tool(name, args)
                text = " ".join(getattr(c, "text", "") for c in res.content)
                print(f"\n{name} -> error={res.is_error}\n{text[:600]}")


asyncio.run(main())
