

#clear all node and relationship in the graph neo4j
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
import os

# load env file
load_dotenv()

def clear_graph():
    uri = os.getenv("NEO4J_URI")
    driver = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

if __name__ == "__main__":
    clear_graph()

