import logging
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from search_agent.providers.model_provider import ModelProvider
from search_agent.providers.search_provider import SearchProvider
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler)
from typing import AsyncIterator, Dict, List, Any


load_dotenv()
# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GrowthAgent(AbstractAgent):
    def __init__(
            self,
            name: str
    ):
        super().__init__(name)

        # Load API keys from environment variables
        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            logger.error("MODEL_API_KEY is not set in environment variables")
            raise ValueError("MODEL_API_KEY is not set")
        self._model_provider = ModelProvider(api_key=model_api_key)

        search_api_key = os.getenv("TAVILY_API_KEY")
        if not search_api_key:
            logger.error("TAVILY_API_KEY is not set in environment variables")
            raise ValueError("TAVILY_API_KEY is not set") 
            
        # Initialize the search provider with explicit logging
        logger.info("Initializing Tavily search provider")
        self._search_provider = SearchProvider(api_key=search_api_key)
        logger.info("Search provider initialized successfully")
        
        # Simple memory to store user profile data between sessions
        # In a production environment, this would be a database
        self._user_memory = {}


    # Implement the assist method as required by the AbstractAgent class
    async def assist(
            self,
            session: Session,
            query: Query,
            response_handler: ResponseHandler
    ):
        """Provide growth advisory for SaaS entrepreneurs."""
        
        logger.info(f"Received query: {query.prompt}")
        logger.info(f"Session processor_id: {session.processor_id}")
        
        # Check if this is a new user or returning user
        user_id = session.processor_id
        is_new_user = user_id not in self._user_memory
        
        # Set up the final response stream
        final_response_stream = response_handler.create_text_stream(
            "FINAL_RESPONSE"
        )
        
        # For new users, create profile and start with single onboarding question
        if is_new_user:
            logger.info(f"New user detected: {user_id}")
            
            # Initialize user profile with default values
            self._user_memory[user_id] = {
                "profile_complete": False,
                "startup_idea": "",
                "creation_time": datetime.now().isoformat()
            }
            
            # Single onboarding question - just get their startup idea
            intro_message = "Hey! What's your SaaS startup idea in 1-2 sentences?"
            
            # Stream the message
            await self._stream_response(intro_message, final_response_stream)
            
        else:
            # Handle returning users
            user_data = self._user_memory[user_id]
            logger.info(f"Returning user: {user_id}, Profile complete: {user_data.get('profile_complete', False)}")
            
            # Check if query explicitly asks for search or contains search triggers
            needs_search = self._needs_search(query.prompt)
            logger.info(f"Query needs search: {needs_search}")
            
            if not user_data.get('profile_complete', False):
                # Complete the onboarding with a single response
                await self._handle_onboarding(user_id, query.prompt, final_response_stream)
            elif needs_search:
                # Provide ultra-concise search-enhanced response
                logger.info("Processing search-enhanced request")
                await self._provide_search_enhanced_advice(user_id, query.prompt, response_handler, final_response_stream)
            else:
                # Provide ultra-concise standard advice
                logger.info("Processing standard request")
                await self._provide_growth_advice(user_id, query.prompt, final_response_stream)
            
        await final_response_stream.complete()
        await response_handler.complete()
    
    
    async def _stream_response(self, message, stream):
        """Helper method to stream response in small chunks."""
        # Split by spaces and punctuation to get natural breaks
        import re
        chunks = re.findall(r'\S+\s*', message)
        
        # Send chunks with 3-4 words each for faster streaming
        buffer = ""
        for chunk in chunks:
            buffer += chunk
            if len(buffer.split()) >= 3 or chunk.endswith(('.', '!', '?', ':', ';', '\n')):
                await stream.emit_chunk(buffer)
                buffer = ""
        
        # Send any remaining text
        if buffer:
            await stream.emit_chunk(buffer)
    
    
    async def _handle_onboarding(
            self,
            user_id: str,
            prompt: str,
            response_stream
    ):
        """Process the simple one-question onboarding and proceed to advice."""
        user_data = self._user_memory[user_id]
        
        # If response is too short or completely off-topic, still accept it but with a prompt
        if len(prompt.strip()) < 5:
            user_data["startup_idea"] = "Unspecified SaaS startup"
            response = "Got it. What specific aspect of your SaaS business do you need help with today?"
        else:
            # Store the startup idea
            user_data["startup_idea"] = prompt
            
            # Generate a very simple hypothesis
            hypothesis = await self._generate_simple_hypothesis(user_data["startup_idea"])
            
            # Mark onboarding as complete
            user_data["profile_complete"] = True
            
            # Create short, personalized response
            response = f"Thanks! I see you're building {hypothesis.get('description', 'a SaaS product')}. What growth challenge can I help with today?"
        
        # Stream the response
        await self._stream_response(response, response_stream)
    
    
    async def _provide_search_enhanced_advice(
            self,
            user_id: str,
            prompt: str,
            response_handler: ResponseHandler,
            final_response_stream
    ):
        """Provide concise advice enhanced with real-time search data."""
        user_data = self._user_memory[user_id]
        
        # Let the user know we're searching (short message)
        await response_handler.emit_text_block(
            "SEARCH_NOTIFICATION", "Searching for market data..."
        )
        
        # Construct a search query based on the user question and their startup
        startup_context = f"SaaS {user_data.get('startup_idea', '')}"
        
        # Clean up the query to focus on the search request
        clean_query = self._clean_search_query(prompt)
        search_query = f"{clean_query} {startup_context}"
        
        # Log the search query for debugging
        logger.info(f"Search query: {search_query}")
        
        try:
            # Perform the search with error handling
            logger.info(f"Calling Tavily search API")
            search_results = await self._search_provider.search(search_query)
            logger.info(f"Search complete, results received")
            
            # Add debugging to check search results
            if search_results:
                logger.info(f"Search results structure: {list(search_results.keys())}")
                if "results" in search_results:
                    logger.info(f"Number of results: {len(search_results['results'])}")
            else:
                logger.info("Search returned empty or null result")
            
            # Process search results
            if search_results and "results" in search_results and len(search_results["results"]) > 0:
                # Emit search results to the client
                logger.info("Emitting search results to client")
                await response_handler.emit_json(
                    "SEARCH_RESULTS", {"results": search_results["results"][:3]}  # Show top 3 results
                )
                
                # Create prompt for ultra-concise advice with search data
                enhanced_prompt = f"""
                As a growth advisor for a SaaS founder building: {user_data.get('startup_idea', 'a SaaS product')}
                
                Their question: {prompt}
                
                Relevant search results:
                {json.dumps(search_results["results"][:3], indent=2)}
                
                Provide ultra-concise advice (2-3 sentences maximum) that:
                1. Directly answers their question using the search data
                2. Gives 1 specific action step
                
                No introductions or explanations. Be extremely direct and practical.
                Clearly reference the source of information (e.g., "According to [source]").
                
                Your response should cite specific data from the search results.
                """
                
                # Generate response
                logger.info("Generating concise search-enhanced advice")
                search_enhanced_response = await self._model_provider.query(enhanced_prompt)
                
                # Stream the enhanced response
                logger.info("Streaming search-enhanced response")
                await self._stream_response(search_enhanced_response, final_response_stream)
            else:
                # If search failed or returned no results, fall back to standard advice
                logger.warning("Search returned no results, falling back to standard advice")
                await response_handler.emit_text_block(
                    "SEARCH_ERROR", "No search results found. Here's my best advice:"
                )
                await self._provide_growth_advice(user_id, prompt, final_response_stream)
        except Exception as e:
            # Log error and fall back to standard advice if search fails
            logger.error(f"Search failed with error: {str(e)}")
            await response_handler.emit_text_block(
                "SEARCH_ERROR", "Search couldn't be completed. Here's my best advice:"
            )
            await self._provide_growth_advice(user_id, prompt, final_response_stream)
    
    
    async def _provide_growth_advice(
            self,
            user_id: str,
            prompt: str,
            response_stream
    ):
        """Provide concise growth advice to an onboarded user."""
        user_data = self._user_memory[user_id]
        
        # Generate concise advice
        advice_prompt = f"""
        As a growth advisor for a SaaS founder building: {user_data.get('startup_idea', 'a SaaS product')}
        
        Their question: {prompt}
        
        Provide extremely concise growth advice (2-3 sentences maximum) that:
        1. Directly answers their question
        2. Gives 1 clear, actionable next step
        
        Use a casual, direct tone. No introductions or pleasantries needed.
        """
        
        logger.info("Generating concise growth advice")
        
        # Get compact response
        response = await self._model_provider.query(advice_prompt)
        
        # Stream it to the client
        await self._stream_response(response, response_stream)
    
    
    async def _generate_simple_hypothesis(self, startup_idea: str) -> Dict[str, Any]:
        """Generate a minimal hypothesis based just on startup idea."""
        
        hypothesis_prompt = f"""
        Based solely on this SaaS startup idea: "{startup_idea}"
        
        Generate a very simple JSON with just:
        1. A short description (under 7 words)
        2. Primary customer segment (1 segment)
        3. Main growth lever (1 channel)
        
        Format as a JSON with keys: "description", "target_segment", "growth_lever"
        Keep all values extremely concise.
        """
        
        hypothesis_text = await self._model_provider.query(hypothesis_prompt)
        
        # Simple parsing with fallback
        try:
            # Try to find and extract the JSON part
            json_match = re.search(r'(\{.*\})', hypothesis_text, re.DOTALL)
            if json_match:
                hypothesis_json = json_match.group(1)
                hypothesis = json.loads(hypothesis_json)
            else:
                hypothesis = json.loads(hypothesis_text)
            return hypothesis
        except Exception as e:
            logger.error(f"Failed to parse hypothesis JSON: {hypothesis_text}")
            logger.error(f"Error details: {str(e)}")
            # Simple fallback
            return {
                "description": "a SaaS solution",
                "target_segment": "businesses",
                "growth_lever": "product-led growth"
            }
    
    def _needs_search(self, query: str) -> bool:
        """
        Enhanced detection system for determining if a query requires external search.
        Handles both explicit and implicit search requests in a domain-agnostic way.
        """
        query_lower = query.lower().strip()
        
        # 1. EXPLICIT SEARCH REQUESTS
        explicit_search_patterns = [
            r'search for',
            r'look up',
            r'find( information| details)?( about| on)?',
            r'research',
            r'google',
            r'can you (search|look|find)',
            r'(show|tell) me (about|the)',
            r'get( me)? information'
        ]
        
        for pattern in explicit_search_patterns:
            if re.search(pattern, query_lower):
                logger.info(f"Explicit search request detected: {pattern}")
                return True
        
        # 2. QUESTION PATTERNS THAT TYPICALLY NEED FACTUAL DATA
        factual_question_patterns = [
            # WH-Questions about facts
            r'^(what|who|where|when|which) (is|are|was|were|do|does|did|has|have|had|can|could|should|would|will)',
            r'^(what\'s|who\'s|where\'s|when\'s)',
            
            # How questions about factual information
            r'^how (do|does|did|can|could|would|should|to|many|much|long)',
            
            # Get me / Tell me / Show me + information
            r'^(get|tell|show)( me)? (the|a|some|all)',
            
            # Requests for lists or examples
            r'(list|give( me)?|name|what are)( the| some| a few| all)? (examples|types|kinds|categories|options|alternatives|companies|tools|solutions)',
            
            # Information requests
            r'(i need|i want|i\'m looking for)( some| more)? information( about| on)?',
            
            # Current or recent information
            r'current|recent|latest|today|this (week|month|year)|trending|new',
            
            # Comparisons
            r'(compare|comparison|versus|vs\.?|difference between|better)'
        ]
        
        # Short query detection - likely a follow-up question
        words = query_lower.split()
        if len(words) <= 3:
            # Check if it's a potential factual follow-up
            continuation_words = ["their", "they", "them", "these", "those", "this", "that", "it", "its", "any", "the"]
            information_words = ["contact", "email", "website", "address", "phone", "details", "info", "information"]
            
            # Short queries that are likely asking for factual information
            if words[0] in continuation_words or any(word in information_words for word in words):
                logger.info(f"Short follow-up query detected that may need search: {query_lower}")
                return True
        
        # Check factual question patterns for longer queries
        if len(words) > 3:
            for pattern in factual_question_patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Factual question pattern detected: {pattern}")
                    return True
        
        # 3. INDICATORS THAT USUALLY REQUIRE EXTERNAL DATA
        # These are specific terms that almost always need external data
        universal_search_indicators = [
            # Factual information markers
            "market", "industry", "statistics", "data", "report", "survey", "research",
            "analysis", "trend", "growth", "forecast", "projection", "estimate",
            
            # Entities that need lookup
            "contact", "email", "phone", "address", "website", "form", "application", 
            "directory", "list", "database", "source", "reference",
            
            # Events and timing
            "conference", "event", "summit", "webinar", "schedule", "agenda", "calendar",
            "date", "time", "deadline", "upcoming", "this week", "this month", "this year",
            
            # Comparison and ranking
            "competitor", "alternative", "similar", "leader", "trending", "popular", "top",
            "review", "rating", "ranking", "best", "worst", "versus", "vs"
        ]
        
        # Check for universal search indicators
        for indicator in universal_search_indicators:
            if indicator in query_lower:
                logger.info(f"Universal search indicator detected: {indicator}")
                return True
        
        # No search triggers detected
        logger.info("No search triggers detected in query")
        return False
    
    def _clean_search_query(self, query: str) -> str:
        """Clean up search queries to focus on the important parts."""
        # Remove search command prefixes
        cleaned = re.sub(r'^(search for|look up|find|research)\s+', '', query.lower())
        
        # Remove filler words
        cleaned = re.sub(r'\b(please|can you|could you|i want to know|tell me|i need)\b', '', cleaned)
        
        # Strip extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned


if __name__ == "__main__":
    # Create an instance of a GrowthAgent
    agent = GrowthAgent(name="SaaS Growth Advisor")
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()