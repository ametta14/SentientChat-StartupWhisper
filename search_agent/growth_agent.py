import logging
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from providers.model_provider import ModelProvider
from providers.search_provider import SearchProvider
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler)
from typing import AsyncIterator, Dict, List, Any


load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GrowthAgent(AbstractAgent):
    def __init__(
            self,
            name: str
    ):
        super().__init__(name)

        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            raise ValueError("MODEL_API_KEY is not set")
        self._model_provider = ModelProvider(api_key=model_api_key)

        search_api_key = os.getenv("TAVILY_API_KEY")
        if not search_api_key:
            raise ValueError("TAVILY_API_KEY is not set") 
        self._search_provider = SearchProvider(api_key=search_api_key)
        
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
        
        # For new users, create profile and start onboarding
        if is_new_user:
            logger.info(f"New user detected: {user_id}")
            
            # Initialize user profile
            self._user_memory[user_id] = {
                "profile_complete": False,
                "startup_idea": "",
                "experience": "",
                "philosophy": "",
                "learning_style": "",
                "onboarding_stage": "startup_idea",
                "creation_time": datetime.now().isoformat()
            }
            
            # Send greeting and first onboarding question
            intro_message = "Hey Founder! I'm excited to build with you. To provide personalized growth advice, I'd like to learn about your startup. What's your SaaS startup idea? Tell me about the problem you're solving and how."
            
            # Stream the message by individual words for better streaming effect
            await self._stream_response(intro_message, final_response_stream)
            
        else:
            # Handle returning users
            user_data = self._user_memory[user_id]
            logger.info(f"Returning user: {user_id}, Profile complete: {user_data.get('profile_complete', False)}")
            
            # Check if query indicates a need for search (specific market research query)
            needs_search = self._needs_search(query.prompt)
            
            # Handle onboarding flow
            if not user_data.get('profile_complete', False):
                await self._handle_onboarding(user_id, query.prompt, final_response_stream)
            elif needs_search:
                # Perform search-enhanced response
                await self._provide_search_enhanced_advice(user_id, query.prompt, response_handler, final_response_stream)
            else:
                # Standard growth advice
                await self._provide_growth_advice(user_id, query.prompt, final_response_stream)
            
        await final_response_stream.complete()
        await response_handler.complete()
    
    
    async def _stream_response(self, message, stream):
        """Helper method to stream response word by word or in small chunks."""
        # Split by spaces and punctuation to get natural breaks
        import re
        chunks = re.findall(r'\S+\s*', message)
        
        # For UI responsiveness, send chunks with 2-3 words each
        # This provides a balance between smoothness and performance
        buffer = ""
        for chunk in chunks:
            buffer += chunk
            if len(buffer.split()) >= 2 or chunk.endswith(('.', '!', '?', ':', ';', '\n')):
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
        """Process user onboarding to collect profile information."""
        user_data = self._user_memory[user_id]
        current_stage = user_data.get("onboarding_stage", "startup_idea")
        
        logger.info(f"Handling onboarding for stage: {current_stage}")
        
        # Handle different onboarding stages
        if current_stage == "startup_idea":
            user_data["startup_idea"] = prompt
            user_data["onboarding_stage"] = "experience"
            
            response = "Great! Thanks for sharing your startup idea. Now, can you tell me about your previous entrepreneurship experience? What businesses have you built or worked on before?"
            
        elif current_stage == "experience":
            user_data["experience"] = prompt
            user_data["onboarding_stage"] = "philosophy"
            
            response = "Thanks for sharing your background. What's your philosophy of change? How do you believe innovation happens and businesses succeed in creating value?"
            
        elif current_stage == "philosophy":
            user_data["philosophy"] = prompt
            user_data["onboarding_stage"] = "learning_style"
            
            response = "Interesting perspective! Last question - what's your learning style? How do you prefer to receive feedback and advice? For example, do you prefer direct feedback, examples, analogies, etc.?"
            
        elif current_stage == "learning_style":
            user_data["learning_style"] = prompt
            user_data["profile_complete"] = True
            
            # Generate initial growth hypothesis
            hypothesis = await self._generate_growth_hypothesis(user_id)
            
            # Create personalized response with actionable advice
            response = f"""Thanks for sharing all that information! Based on what you've told me, I've created an initial growth hypothesis for your startup.

Target Customer Segments:
1. {hypothesis.get('target_segments', [])[0] if hypothesis.get('target_segments') else 'Primary target market'}
2. {hypothesis.get('target_segments', [])[1] if len(hypothesis.get('target_segments', [])) > 1 else 'Secondary target market'}

Key Growth Levers:
1. {hypothesis.get('growth_levers', [])[0] if hypothesis.get('growth_levers') else 'Customer acquisition channel'}
2. {hypothesis.get('growth_levers', [])[1] if len(hypothesis.get('growth_levers', [])) > 1 else 'Retention strategy'}

Now, let's start working on your growth strategy. What specific aspect of growing your SaaS business would you like advice on first? (e.g., pricing, marketing, product-market fit, customer retention)"""
        
        # Stream the response
        await self._stream_response(response, response_stream)
    
    
    async def _provide_search_enhanced_advice(
            self,
            user_id: str,
            prompt: str,
            response_handler: ResponseHandler,
            final_response_stream
    ):
        """Provide advice enhanced with real-time search data."""
        user_data = self._user_memory[user_id]
        
        # Let the user know we're searching
        await response_handler.emit_text_block(
            "SEARCH_NOTIFICATION", "Researching the latest market data to provide you with up-to-date information..."
        )
        
        # Construct a search query based on the user question and their startup
        startup_context = f"SaaS startup {user_data.get('startup_idea', '')}"
        search_query = f"{prompt} {startup_context}"
        
        # Perform the search
        logger.info(f"Performing search for: {search_query}")
        search_results = await self._search_provider.search(search_query)
        
        # Process search results
        if search_results and "results" in search_results and len(search_results["results"]) > 0:
            # Emit the search results as JSON for the client to display
            await response_handler.emit_json(
                "SEARCH_RESULTS", {"results": search_results["results"][:3]}  # Limit to top 3 results
            )
            
            # Create prompt combining user context with search results
            enhanced_prompt = f"""
            As a growth advisor for a SaaS founder:
            
            Founder profile:
            - Startup idea: {user_data.get('startup_idea', 'Not specified')}
            - Experience: {user_data.get('experience', 'Not specified')}
            - Philosophy: {user_data.get('philosophy', 'Not specified')}
            - Learning style: {user_data.get('learning_style', 'Not specified')}
            
            Their question/input: {prompt}
            
            Recent search results:
            {json.dumps(search_results["results"][:3], indent=2)}
            
            Provide personalized, actionable growth advice that:
            1. Addresses their specific query directly
            2. Incorporates the most relevant insights from the search results (with brief citations)
            3. Suggests 2-3 specific next steps they can take today
            
            Use a casual, conversational tone that's professional but not formal. Be concise and practical. 
            Organize your response with clear sections.
            
            When citing search results, include only short quotes (under 25 words) if needed, and provide a brief citation to the source.
            """
            
            # Generate response
            logger.info("Generating search-enhanced advice")
            search_enhanced_response = await self._model_provider.query(enhanced_prompt)
            
            # Stream the enhanced response
            await self._stream_response(search_enhanced_response, final_response_stream)
        else:
            # If search failed, fall back to standard advice
            logger.info("Search returned no results, falling back to standard advice")
            await self._provide_growth_advice(user_id, prompt, final_response_stream)
    
    
    async def _provide_growth_advice(
            self,
            user_id: str,
            prompt: str,
            response_stream
    ):
        """Provide personalized growth advice to an onboarded user."""
        user_data = self._user_memory[user_id]
        
        # Generate personalized advice
        advice_prompt = f"""
        As a growth advisor for a SaaS founder:
        
        Founder profile:
        - Startup idea: {user_data.get('startup_idea', 'Not specified')}
        - Experience: {user_data.get('experience', 'Not specified')}
        - Philosophy: {user_data.get('philosophy', 'Not specified')}
        - Learning style: {user_data.get('learning_style', 'Not specified')}
        
        Their question/input: {prompt}
        
        Provide personalized, actionable growth advice that:
        1. Addresses their specific query directly
        2. Matches their learning style 
        3. Incorporates startup growth best practices from YC, A16Z, and First Round
        4. Suggests 2-3 specific next steps they can take today
        
        Use a casual but professional tone. Be direct and conversational but avoid profanity.
        Organize your response with clear sections.
        """
        
        logger.info("Generating standard growth advice")
        
        # Get full response
        full_response = await self._model_provider.query(advice_prompt)
        
        # Stream it to the client
        await self._stream_response(full_response, response_stream)
    
    
    async def _generate_growth_hypothesis(self, user_id: str) -> Dict[str, Any]:
        """Generate an initial growth hypothesis based on user profile."""
        user_data = self._user_memory[user_id]
        
        hypothesis_prompt = f"""
        Based on the founder's input:
        - Startup idea: {user_data['startup_idea']}
        - Experience: {user_data['experience']}
        - Philosophy: {user_data['philosophy']}
        - Learning style: {user_data['learning_style']}
        
        Generate a structured growth hypothesis with these elements:
        1. Target customer segments (2-3 specific segments)
        2. Key value propositions for each segment
        3. Growth levers (2-3 most promising channels)
        4. Initial success metrics
        
        Format as a structured JSON object with these keys.
        """
        
        hypothesis_text = await self._model_provider.query(hypothesis_prompt)
        
        # Simple parsing - in production, add better error handling
        try:
            hypothesis = json.loads(hypothesis_text)
            return hypothesis
        except:
            logger.error(f"Failed to parse hypothesis JSON: {hypothesis_text}")
            # Fallback if JSON parsing fails
            return {
                "target_segments": ["Early-stage SaaS founders", "Technical founders transitioning to CEO role"],
                "value_propositions": {
                    "Early-stage SaaS founders": "Accelerated time-to-market and founder-market fit",
                    "Technical founders": "Business skill acquisition without expensive MBA"
                },
                "growth_levers": ["Product-led growth", "Community building", "Content marketing"],
                "success_metrics": ["User acquisition cost", "Activation rate", "30-day retention"]
            }
    
    
    def _needs_search(self, query: str) -> bool:
        """Determine if the query requires external search."""
        # Keywords that indicate a need for search
        search_triggers = [
            # Market research terms
            "market", "competitor", "industry", "trend", "data", 
            "research", "analysis", "report", "investors", "funding",
            "benchmark", "statistics", "comparison", "insights",
            
            # Current events/updates
            "recent", "latest", "new", "current", "update", "today", "this week", "this month",
            "this year", "announcement", "launched", "released", "news",
            
            # Questions that need external data
            "what is the market size", "who are the leading", "what are the trends",
            "how big is", "what's happening in", "compare", "versus", "vs", 
            
            # Specific external knowledge
            "valuation", "funding round", "acquisition", "merger", "ipo", "regulations",
            "compliance", "best practices", "top companies", "market leaders"
        ]
        
        # Check if any trigger word is in the query
        return any(trigger in query.lower() for trigger in search_triggers)
    

if __name__ == "__main__":
    # Create an instance of a GrowthAgent
    agent = GrowthAgent(name="SaaS Growth Advisor")
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()