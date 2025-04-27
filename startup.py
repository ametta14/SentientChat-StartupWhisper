#!/usr/bin/env python3
"""
Startup script for the SaaS Growth Advisor agent.
This script verifies that all required services are available before starting the agent.
"""

import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("startup")

async def verify_api_keys():
    """Verify that all required API keys are set in the environment."""
    load_dotenv()
    
    # Check for model API key
    model_api_key = os.getenv("MODEL_API_KEY")
    if not model_api_key:
        logger.error("❌ MODEL_API_KEY is not set in .env file")
        return False
    logger.info("✅ MODEL_API_KEY is set")
    
    # Check for Tavily API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("❌ TAVILY_API_KEY is not set in .env file")
        return False
    logger.info("✅ TAVILY_API_KEY is set")
    
    return True

async def verify_tavily_connection():
    """Verify that we can connect to the Tavily API."""
    from search_agent.providers.search_provider import SearchProvider
    
    try:
        logger.info("Testing Tavily API connection...")
        search_provider = SearchProvider(api_key=os.getenv("TAVILY_API_KEY"))
        # Since your existing SearchProvider doesn't have test_connection, we'll do a simple search
        test_results = await search_provider.search("test connection")
        
        if test_results and "results" in test_results and len(test_results["results"]) > 0:
            logger.info("✅ Tavily API connection successful")
            return True
        else:
            logger.warning("⚠️ Tavily API returned empty results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing Tavily connection: {str(e)}")
        return False

async def verify_model_connection():
    """Verify that we can connect to the model API."""
    from search_agent.providers.model_provider import ModelProvider
    
    try:
        logger.info("Testing model API connection...")
        model_provider = ModelProvider(api_key=os.getenv("MODEL_API_KEY"))
        
        # Simple test query
        response = await model_provider.query("Hello, are you working?")
        
        if response and len(response) > 0:
            logger.info("✅ Model API connection successful")
            return True
        else:
            logger.error("❌ Model API returned empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing model connection: {str(e)}")
        return False

async def verification_checks():
    """Run all verification checks."""
    # Check API keys
    if not await verify_api_keys():
        logger.error("❌ API key verification failed. Please check your .env file")
        return False
    
    # Check Tavily connection
    if not await verify_tavily_connection():
        logger.error("❌ Tavily API connection failed. Check your API key and internet connection")
        logger.warning("⚠️ Continuing without search capability. The agent will fall back to standard advice")
    
    # Check model connection
    if not await verify_model_connection():
        logger.error("❌ Model API connection failed. Check your API key and internet connection")
        return False
        
    return True

def main():
    """Main startup sequence."""
    logger.info("🚀 Starting SaaS Growth Advisor")
    
    # Run verification checks
    if not asyncio.run(verification_checks()):
        sys.exit(1)
    
    # All verifications passed, start the agent
    logger.info("✅ All systems verified. Starting the growth agent server...")
    
    # Import and run the agent
    from search_agent.growth_agent import GrowthAgent
    from sentient_agent_framework import DefaultServer
    
    agent = GrowthAgent(name="SaaS Growth Advisor")
    server = DefaultServer(agent)
    
    # Run the server directly (not with asyncio.run)
    logger.info("Starting server on http://0.0.0.0:8000")
    server.run()

if __name__ == "__main__":
    main()