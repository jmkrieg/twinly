import os
from openai import AsyncAzureOpenAI


endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2023-05-15")

embedding_client = AsyncAzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=embedding_api_version,
)

embedding_deployment = (
                os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or
                "text-embedding-3-small"  # Safe default - don't fall back to chat model
            )

print(f"Using embedding deployment: {embedding_deployment}")

import asyncio

text = "This is a test text for embedding."

async def main():
    response = await embedding_client.embeddings.create(
        model=embedding_deployment,
        input=text
    )

    for item in response.data:
        length = len(item.embedding)
        print(
            f"data[{item.index}]: length={length}, "
            f"[{item.embedding[0]}, {item.embedding[1]}, "
            f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
        )

if __name__ == "__main__":
    asyncio.run(main())