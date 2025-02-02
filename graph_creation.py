from langchain_core.runnables import (RunnableBranch, RunnableLambda, RunnableParallel,RunnablePassthrough)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.document_loaders import WikipediaLoader

from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough

from neo4j import GraphDatabase

from dotenv import load_dotenv
from typing import Dict, List

# load env file
load_dotenv()

def load_documents():
    raw_documents = TextLoader('sazae-san.wiki.txt').load()
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    documents = text_splitter.split_documents(raw_documents[:3])
    return documents

def create_graph(documents : List[Document]):
    documents = load_documents()

    llm=ChatOpenAI(temperature=0, model_name="gpt-4o") 
    llm_transformer = LLMGraphTransformer(llm=llm,
        allowed_nodes=["Charactor", "Actor"],  
        allowed_relationships=["Family", "Act"],
        node_properties=["text"],
        relationship_properties=["text"],
        strict_mode=True,
        additional_instructions="ノードとリレーションシップそれぞれのtextプロパティには、それぞれのノードやリレーションシップの概要を入れてください。"
        )
    
    graph_documents = llm_transformer.convert_to_graph_documents(documents[0:3], )

    return graph_documents

def save_graph(graph_documents, graphdb):
    print(graph_documents)
    graphdb.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)


def embed_graph() -> Dict[str, Neo4jVector]:
    Document =  Neo4jVector.from_existing_graph( 
                    OpenAIEmbeddings(model="text-embedding-ada-002"),
                    index_name="Document_vector",
                    keyword_index_name="Document_keyword",
                    search_type="hybrid",
                    node_label="Document",
                    text_node_properties=["id", "text"],
                    embedding_node_property="embedding")
    Charactor= Neo4jVector.from_existing_graph( 
                    OpenAIEmbeddings(model="text-embedding-ada-002"),
                    index_name="Charactor_vector",
                    keyword_index_name="Charactor_keyword",
                    search_type="hybrid",
                    node_label="Charactor",
                    text_node_properties=["id", "text"],
                    embedding_node_property="embedding")
    Actor=     Neo4jVector.from_existing_graph( 
                    OpenAIEmbeddings(model="text-embedding-ada-002"),
                    index_name="Actor_vector",
                    keyword_index_name="Actor_keyword",
                    search_type="hybrid",
                    node_label=" Actor",
                    text_node_properties=["id", "text"],
                    embedding_node_property="embedding")

    vectorindexies = {
        "Document" :  Document,
        "Charactor" : Charactor,
        "Actor" :     Actor
    }

    return vectorindexies


def create_fulltextindex(graphdb: Neo4jGraph):
    graphdb.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


def main():
    graphdb = Neo4jGraph()
    documents = load_documents()
    graph_documents = create_graph(documents)

    save_graph(graph_documents, graphdb)

    embed_graph()


def sampleLLM():
    llm=ChatOpenAI(temperature=1.0, model_name="gpt-3.5-turbo") 
    res = llm.invoke("What is the capital of France?")
    print(res)

    

if __name__ == "__main__":
    sampleLLM()
    main()
