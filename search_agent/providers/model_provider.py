from datetime import datetime
from langchain_core.prompts import PromptTemplate
from openai import AsyncOpenAI
from typing import AsyncIterator

class ModelProvider:
    def __init__(
        self,
        api_key: str
    ):
        """ Initializes model, sets up OpenAI client, configures system prompt."""

        # Model provider API key
        self.api_key = api_key
        # Model provider URL
        self.base_url = "https://api.fireworks.ai/inference/v1" 
        # Identifier for specific model that should be used
        self.model = "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new"
        # Temperature setting for response randomness - lower temp for less randomness
        self.temperature = 0.5  # Reduced from 0.7 to make it slightly less random/casual
        # Maximum number of tokens for responses
        self.max_tokens = None
        self.date_context = datetime.now().strftime("%Y-%m-%d")

        # Set up custom system prompt to guide model behavior
        self.system_prompt = """
        You are a straight-talking growth advisor for SaaS entrepreneurs. You have direct, no-nonsense advice based on your experience with hundreds of successful startups.
        
        Your communication style is:
        1. Direct and clear - you don't waste time with corporate speak
        2. Evidence-based - you reference what actually works
        3. Motivational but real - you push founders with honest feedback
        4. Casual but professional - avoid profanity or inappropriate language
        
        You know the playbooks from Y Combinator, Andreessen Horowitz, First Round Capital, and other top accelerators and VCs. When you speak, it's with the authority of someone who's seen it all.
        
        Use casual language, contractions, and a conversational tone. Feel free to use slang, but avoid vulgar language or profanity. Be direct but respectful.
        
        Today's date is {date_today}. Use this to ensure your advice is timely and relevant.
        """

        # Set up model API
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )


    async def query_stream(
        self,
        query: str
    ) -> AsyncIterator[str]:
        """Sends query to model and yields the response in chunks."""

        # Add a filter at the query level to discourage profanity
        filtered_query = f"{query}\n\nRemember to keep your tone casual but professional, avoiding any vulgar language."

        if self.model in ["o1-preview", "o1-mini"]:
            messages = [
                {"role": "user",
                 "content": f"System Instruction: {self.system_prompt} \n Instruction:{filtered_query}"}
            ]
        else:
            messages = [
                {"role": "system", "content": self.system_prompt.format(date_today=self.date_context)},
                {"role": "user", "content": filtered_query}
            ]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


    async def query(
        self,
        query: str
    ) -> str:
        """Sends query to model and returns the complete response as a string."""
        
        chunks = []
        async for chunk in self.query_stream(query=query):
            chunks.append(chunk)
        response = "".join(chunks)
        
        # Post-process to replace any remaining profanity with cleaner alternatives
        # This is a simple approach - for production you'd want more sophisticated filtering
        profanity_replacements = {
            "shit": "stuff",
            "fuck": "darn",
            "fucking": "really",
            "damn": "darn",
            "ass": "butt",
            "bitch": "difficult person",
            "hell": "heck"
        }
        
        clean_response = response
        for bad_word, replacement in profanity_replacements.items():
            # Replace variants with word boundaries to avoid partial matches
            clean_response = clean_response.replace(f" {bad_word} ", f" {replacement} ")
            clean_response = clean_response.replace(f" {bad_word}.", f" {replacement}.")
            clean_response = clean_response.replace(f" {bad_word},", f" {replacement},")
            clean_response = clean_response.replace(f" {bad_word}!", f" {replacement}!")
            clean_response = clean_response.replace(f" {bad_word}?", f" {replacement}?")
            
        return clean_response