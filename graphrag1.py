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
from graph_creation import embed_graph
from pprint import pprint


# load env file
load_dotenv()


graphdb = Neo4jGraph()
vector_indexies = embed_graph()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")


from langchain.chains import GraphCypherQAChain

graphdb.refresh_schema()

cypher_chain = GraphCypherQAChain.from_llm(
    graph=graphdb,
    cypher_llm=llm,
    qa_llm=llm,
    validate_cypher=True,
    verbose=True,
    allow_dangerous_requests=True
)

result = cypher_chain.invoke({"query": "サザエさんと結婚しているのは？"})
pprint(result)


