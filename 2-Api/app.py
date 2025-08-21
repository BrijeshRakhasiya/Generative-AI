from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A Simple API Server"
)

# Base models
openai_model = ChatOpenAI()
ollama_model = Ollama(model="llama3.2:latest")

# Add base OpenAI route
add_routes(
    app=app,
    runnable=openai_model,
    path="/openai"
)

# Prompts
prompt1 = ChatPromptTemplate.from_messages([
    ("user", "Write me an essay about {topic} with 100 words.")
])

prompt2 = ChatPromptTemplate.from_messages([
    ("user", "Write me a poem about {topic} with 100 words.")
])

# Chains
essay_chain = prompt1 | openai_model
poem_chain = prompt2 | ollama_model

# Add custom routes
add_routes(
    app=app,
    runnable=essay_chain,
    path="/essay"
)

add_routes(
    app=app,
    runnable=poem_chain,
    path="/poem"
)

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)