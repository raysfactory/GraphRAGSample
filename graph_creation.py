from langchain_core.runnables import (RunnableBranch, RunnableLambda, RunnableParallel,RunnablePassthrough)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
#from langchain.document_loaders import WikipediaLoader

from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

# load env file
load_dotenv()

def load_documents():
    raw_documents = TextLoader('sazae-san.wiki.txt').load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents[:3])
    return documents

def create_graph(documents : List[Document]):
    documents = load_documents()

    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo") 
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    return graph_documents

def save_graph(graph_documents, graphdb):
    print(graph_documents)
    graphdb.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)


def embed_graph(graphdb):
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    vector_index.create_index(graphdb)


def main():
    documents = load_documents()
    graph_documents = create_graph(documents)

    graphdb = Neo4jGraph(uri=os.getenv('NEO4J_URI'), user=os.getenv('NEO4J_USER'), password=os.getenv('NEO4J_PASSWORD'))

    save_graph(graph_documents, graphdb)

    embed_graph(graphdb)


def sampleLLM():
    llm=ChatOpenAI(temperature=1.0, model_name="gpt-3.5-turbo") 
    res = llm.invoke("What is the capital of France?")
    print(res)

    

if __name__ == "__main__":
    sampleLLM()
    main()
