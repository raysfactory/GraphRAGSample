from langchain_core.runnables import (RunnableBranch, RunnableLambda, RunnableParallel,RunnablePassthrough)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from openai import RateLimitError
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
from joblib import Parallel, delayed
import time

from util.knowlege_util import node_labels, relationship_labels, embed_graph

# load env file
load_dotenv()

# OpenAI models
#llm=ChatOpenAI(temperature=0, model_name="gpt-4o") 
llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini") 
embedding=OpenAIEmbeddings(model="text-embedding-3-large")

# Neo4j Graph
graphdb = Neo4jGraph()


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
    # URL毎に並列に処理を実施する
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_url, urls))

        # 結果を取得
        future = concurrent.futures.as_completed(results)


def process_url(url):
    start_time = time.time()

    print(f"START : processing : {url}")
    raw_documents = WebBaseLoader(url).load()
    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    graphs = create_graph(url, documents)
    print(f"PRG   : graph created : {url}")
    sage_graph(url, graphs)

    elapsed_time = time.time() - start_time
    print(f"END   : processing : {url} (Elapsed time: {elapsed_time:.2f} seconds)")



def create_graph(url, documents : List[Document]):

    llm_transformer = LLMGraphTransformer(llm=llm,
        allowed_nodes=node_labels,  
        allowed_relationships=relationship_labels,
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
    graph_documents = []
    for doc in documents:
        # 状況に応じてエラーが発生するため、処理の完遂を優先し、エラー時はログ出力のみして続行します。
        rate_limit_retry(url, lambda:
                         graph_documents.extend(llm_transformer.convert_to_graph_documents([doc])))

    return graph_documents


def rate_limit_retry(url, func):
    retrycount = 0
    while True:
        try:
            return func()
        except RateLimitError as e:
            retrycount += 1
            print(f"ERROR!:{url}")
            print(e)
            if retrycount >= 5:
                print("!!!! Retry count exceeded. Aborting.")
                return
            retry_after = float(e.response.headers.get("retry-after", 5))
            print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
            time.sleep(retry_after)


def sage_graph(url, graph_documents):
    for graph_document in graph_documents:
        try:
            graphdb.add_graph_documents([graph_document], baseEntityLabel=True, include_source=True)
        except Exception as e:
            print(f"ERROR!:{url}")
            print(e)



def create_fulltextindex(graphdb: Neo4jGraph):
    graphdb.query(
    "CREATE FULLTEXT INDEX ftx_entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


def main():
    load_documents()
    #create_graph(documents)

    # 汎用的に実装する方法がなく、今回は割愛するが、対象ドメイン毎にVector距離やWork距離やLLMによる解釈などを利用して、各Nodeの重複を削除する処理が必要（エンティティ解決）
    # refer to https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/


    embed_graph(embedding)
    create_fulltextindex(graphdb)


if __name__ == "__main__":
    main()



