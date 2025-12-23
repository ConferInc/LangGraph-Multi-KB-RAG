import os
import json
from typing import Literal, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import StateGraph, END, START

# Configuration
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = "https://litellm.confer.today"
QDRANT_URL = "https://qdrant.confersolutions.ai"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
LLM_MODEL = "gpt-4.1-nano"
EMBEDDING_MODEL = "text-embedding-3-small"

# Load prompts
with open("prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

# Vector Database Setup
qdrant_client = QdrantClient(url=QDRANT_URL, port=443, api_key=QDRANT_API_KEY)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

moxi_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="moxi-website",
    embedding=embeddings,
    content_payload_key="content",
)

confer_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="confer-website",
    embedding=embeddings,
    content_payload_key="content",
)

# State & Data Models
class GraphState(TypedDict):
    question: str
    classification: Optional[str]
    documents: List[str]
    generation: str

class RouteQuery(BaseModel):
    datasource: Literal["confer", "moxi", "general"] = Field(..., description="Target domain")

# Node Definitions
def classify_query(state: GraphState):
    question = state["question"]
    q_lower = question.lower()

    # Deterministic routing for known brands to avoid LLM misclassification.
    if "confer" in q_lower:
        return {"classification": "confer"}
    if "moxi" in q_lower:
        return {"classification": "moxi"}

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, base_url=OPENAI_API_BASE)
    prompt = ChatPromptTemplate.from_messages([("system", PROMPTS["classification"]), ("human", "{question}")])
    router = prompt | llm.with_structured_output(RouteQuery)
    result = router.invoke({"question": question})
    return {"classification": result.datasource}

def retrieve_moxi(state: GraphState):
    return {"documents": [d.page_content for d in moxi_store.similarity_search(state["question"], k=4)]}

def retrieve_confer(state: GraphState):
    return {"documents": [d.page_content for d in confer_store.similarity_search(state["question"], k=4)]}

def generate_response(state: GraphState):
    classification = state.get("classification", "general")
    documents = state.get("documents", [])
    question = state["question"]
    
    # Safety check
    if any(keyword in question.lower() for keyword in PROMPTS["harmful_keywords"]):
        return {"generation": PROMPTS["safety_message"]}
    
    if classification in ["moxi", "confer"] and documents:
        docs_text = "\n\n".join(documents)
        system_prompt = f"""You are a knowledgeable assistant specializing in {classification.upper()}.

{PROMPTS["personality"]}

{PROMPTS["rag_principles"]}

Context:
{docs_text}"""
        temperature = 0
    else:
        system_prompt = f"""You are a helpful AI assistant.

{PROMPTS["personality"]}

{PROMPTS["general_guidance"]}"""
        temperature = 0.5
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=temperature, api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    response = (ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")]) | llm).invoke({"question": question})
    return {"generation": response.content}

# Graph Orchestration
workflow = StateGraph(GraphState)
workflow.add_node("classify", classify_query)
workflow.add_node("moxi_retriever", retrieve_moxi)
workflow.add_node("confer_retriever", retrieve_confer)
workflow.add_node("generate_response", generate_response)

workflow.add_edge(START, "classify")
workflow.add_conditional_edges("classify", lambda x: x["classification"], {"moxi": "moxi_retriever", "confer": "confer_retriever", "general": "generate_response"})
workflow.add_edge("moxi_retriever", "generate_response")
workflow.add_edge("confer_retriever", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()

# Execution
if __name__ == "__main__":
    print("Starting interactive chat...\nType 'quit', 'exit', or 'q' to stop.\n")
    
    while True:
        query = input("User: ")
        if query.lower() in ["quit", "exit", "q"]:
            print("Exiting. Goodbye!")
            break
        if query.strip():
            print(f"Assistant: {app.invoke({'question': query})['generation']}")
