from langchain_core.runnables import (RunnableBranch, RunnableLambda, RunnableParallel,RunnablePassthrough)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import WikipediaLoader

#from langchain_community.graphs import Neo4jGraph
#from langchain_community.vectorstores import Neo4jVector
#from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_neo4j import Neo4jGraph, Neo4jVector
#from langchain.document_loaders import WikipediaLoader

from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough

from neo4j import GraphDatabase

from dotenv import load_dotenv
from typing import Dict, List

# load env file
load_dotenv()

# OpenAI models
llm=ChatOpenAI(temperature=0, model_name="gpt-4o") 
embedding=OpenAIEmbeddings(model="text-embedding-3-large")


node_labels = ["Plactice", "Consideration", "Azureresource", "Azureimprementation"]
node_relationships = ["PlacticeToConsideration", "PlacticeToAzureResource", "AzureResourceToAzureImprementation", "PlacticeToAzureImprementation"]

def load_documents():
    urls = [
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/api-design",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/api-implementation",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/auto-scaling",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/background-jobs",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/caching",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/cdn",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/data-partitioning",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/data-partitioning-strategies",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/host-name-preservation",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/message-encode",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/monitoring",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/retry-service-specific",
        "https://learn.microsoft.com/ja-jp/azure/architecture/best-practices/transient-faults"
    ]
    documents = []
    for url in urls[3:4]:
        raw_documents = WebBaseLoader(url).load()
        text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=100)
        documents.extend(text_splitter.split_documents(raw_documents))
    return documents



def create_graph(documents : List[Document]):
    documents = load_documents()

    llm_transformer = LLMGraphTransformer(llm=llm,
        allowed_nodes=node_labels,  
        allowed_relationships=node_relationships,
        node_properties=["text"],
        relationship_properties=["text"],
        strict_mode=True,
        additional_instructions="""
各ノードのtextプロパティには、"できる限り元の情報そのままの内容" を入れてください。

リレーションシップのtextプロパティには、リレーションシップの概要を入れてください。
"できる限り"ノードとそれに紐づくリレーションシップの情報をもれなく入れてください。

Placticeノードには、Document内にて言及されている各種実現方式などのプラクティスに関する情報を入れてください。
Considerationノードには、各Plactice内にて言及されている考慮事項に関する情報を入れてください。
AzureResourceノードには、各Placticeを実現するために利用できる、Azureのリソース種別名を入れてください。
AzureImprementationノードには、各Plactice内にて言及されているAzureでの実装に関する情報を入れてください。
PlacticeToConsiderationリレーションシップには、PlacticeとConsiderationの関係性に関する情報を入れてください。
PlacticeToAzureResourceリレーションシップには、PlacticeとAzureResourceの関係性に関する情報を入れてください。
AzureResourceToAzureImprementationリレーションシップには、AzureResourceとAzureImprementationの関係性に関する情報を入れてください。
PlacticeToAzureImprementationリレーションシップには、PlacticeとAzureImprementationの関係性に関する情報を入れてください。
        """
        )
    
    graph_documents = llm_transformer.convert_to_graph_documents(documents[1:3], )

    # 汎用的に実装する方法がなく、今回は割愛するが、対象ドメイン毎にVector距離やWork距離やLLMによる解釈などを利用して、各Nodeの重複を削除する処理が必要（エンティティ解決）
    # refer to https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/

    return graph_documents


def save_graph(graph_documents, graphdb):
    print(graph_documents)
    for graph_document in graph_documents:
        graphdb.add_graph_documents([graph_document], baseEntityLabel=True, include_source=True)


def embed_graph() -> Dict[str, Neo4jVector]:
    index_dict = {}
    index_dict["Document"] = _embed_index("Document")
    for node_label in node_labels:
        index_dict[node_label] = _embed_index(node_label)
    return index_dict


def _embed_index(node_label: str) -> Neo4jVector:
    return Neo4jVector.from_existing_graph(
        embedding=embedding,
        index_name=f"{node_label}_vector",
        keyword_index_name=f"{node_label}_keyword",
        search_type="hybrid",
        node_label=node_label,
        text_node_properties=["id", "text"],
        embedding_node_property="embedding"
    )


def create_fulltextindex(graphdb: Neo4jGraph):
    graphdb.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


def main():
    graphdb = Neo4jGraph()
    documents = load_documents()
    graph_documents = create_graph(documents)

    save_graph(graph_documents, graphdb)
    embed_graph()


if __name__ == "__main__":
    main()



