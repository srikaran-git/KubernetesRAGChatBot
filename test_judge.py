import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()


async def main():
    api_key = os.getenv("GROQ_API_KEY")

    print("Key loaded:", bool(api_key))
    print("Looks like Groq key:", api_key.startswith("gsk_") if api_key else False)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    response = await client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": "Return only the word OK",
            }
        ],
    )

    print(response.choices[0].message.content)


asyncio.run(main())