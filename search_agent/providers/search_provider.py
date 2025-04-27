from tavily import AsyncTavilyClient

class SearchProvider:
    def __init__(
            self,
            api_key: str
    ):
        self.client = AsyncTavilyClient(api_key=api_key)


    async def search(
            self,
            query: str
    ) -> dict:
        """
        Searches for information using Tavily API.
        
        For SaaS growth advisor, this can be used to search for:
        - Market research on specific industries
        - Startup funding news
        - Competitor analysis
        - Growth strategy best practices
        - Investor information
        """
        
        # We could enhance this with specific search parameters
        # For example, focusing on recent information for market trends
        results = await self.client.search(
            query,
            search_depth="advanced",  # More comprehensive search
            include_domains=["ycombinator.com", "a16z.com", "firstround.com", "techcrunch.com", "crunchbase.com"],  # Focus on high-quality startup sources
            max_results=5  # Limit to most relevant results
        )
        return results