

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

def clear_all_index():
    uri = os.getenv("NEO4J_URI")
    driver = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    with driver.session() as session:
        session.run("DROP INDEX Document_vector IF EXISTS")
        session.run("DROP INDEX Plactice_vector IF EXISTS")
        session.run("DROP INDEX Consideration_vector IF EXISTS")
        session.run("DROP INDEX Azureresource_vector IF EXISTS")
        session.run("DROP INDEX Azureimprementation_vector IF EXISTS")
        session.run("DROP INDEX AzureResource_vector IF EXISTS")
        session.run("DROP INDEX AzureImprementation_vector IF EXISTS")
        session.run("DROP INDEX Charactor_vector IF EXISTS")
        session.run("DROP INDEX Actor_vector IF EXISTS")
        session.run("DROP INDEX Document_keyword IF EXISTS")
        session.run("DROP INDEX Plactice_keyword IF EXISTS")
        session.run("DROP INDEX Consideration_keyword IF EXISTS")
        session.run("DROP INDEX Azureresource_keyword IF EXISTS")
        session.run("DROP INDEX Azureimprementation_keyword IF EXISTS")
        session.run("DROP INDEX AzureResource_keyword IF EXISTS")
        session.run("DROP INDEX AzureImprementation_keyword IF EXISTS")
        session.run("DROP INDEX Charactor_keyword IF EXISTS")
        session.run("DROP INDEX Actor_keyword IF EXISTS")


if __name__ == "__main__":
    clear_all_index()
