

#clear all node and relationship in the graph neo4j
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough

from langchain_community.graphs import Neo4jGraph

from dotenv import load_dotenv
import os


# load env file
load_dotenv()

def print_neo4j_schema():
    graphdb = Neo4jGraph()
    graphdb.refresh_schema()  # DBからスキーマを取得
    print(graphdb.structured_schema)


if __name__ == "__main__":
    print_neo4j_schema()

