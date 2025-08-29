from langchain_core.runnables import (RunnableBranch, RunnableLambda, RunnableParallel,RunnablePassthrough)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field

from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
import os
#from langchain_community.graphs import Neo4jGraph
#from langchain_community.vectorstores import Neo4jVector
#from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_neo4j import Neo4jGraph, Neo4jVector
#from langchain.document_loaders import WikipediaLoader

from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, AzureOpenAI, AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from graph_creation import embed_graph
from pprint import pprint
from time import sleep
import pprint

from util.knowlege_util import node_labels, relationship_labels, embed_graph

# load env file
load_dotenv()

# OpenAI models
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
embedding=OpenAIEmbeddings(model="text-embedding-3-large")

# Neo4j Graph
graphdb = Neo4jGraph()
graphdb.refresh_schema()  # DBからスキーマを取得

# neo4j Vector Indexis
vector_indexies = embed_graph(embedding)



def vector_node_search(vector_index: Neo4jVector, query: str, k: int = 5):    
    """
    Vector similarity search with configurable result count.
    Increased default from 3 to 5 for better recall.
    """
    # クエリを設定して検索を実行
    # (このサンプルではVector Index作成する際にHybrid Indexを作成しているため、Hybrid Indexを使用して検索を行う)
    documents = vector_index.similarity_search(query, k=k)
    return documents


def get_target_node_types(question: str):
    """
    Improved prompt for better node type selection accuracy.
    """
    message = [
        SystemMessage(content= f"""
次の質問に回答するために、Graph内のどのノードをVector検索するかを正確に判断してください。

**重要な指示:**
1. 複数のノードの種類を指定する場合は、カンマ区切りで指定してください
2. 出力には "- output" や説明文は含めず、ノードタイプ名のみを出力してください
3. 最も関連性の高いノードタイプを優先して選択してください
4. 質問の意図を深く理解して、適切なノードの組み合わせを選択してください
                
Graph DBのスキーマは以下の通り

-- graph schema -------------------- 
“{graphdb.structured_schema}”


-- node and relationship description --------------------
Practiceノードは、Document内にて言及されている各種実現方式などのプラクティスに関する情報
Considerationノードには、各Practice内にて言及されている考慮事項に関する情報
Azureresourceノードには、各Practiceを実現するために利用できる、Azureのリソース種別名
Azureimplementationノードには、各Practice内にて言及されているAzureでの実装に関する情報

-----------------------
回答例: 
- input :  Azure ResourceのLogicAppに関連したPracticeが知りたい
- output :  Azureresource,Practice
回答例: 
- input :  バッチ処理を作成するときの考慮事項は？
- output :  Practice,Consideration
"""),
        HumanMessage(content=question)]
    res = llm.invoke(message)
    return res.content


def node_relation_search(nodes, nodetype:str, destnodetype:str):    
    cypher_response = []
    for node in nodes:
        id = node.page_content.split('\n')[1].split(': ')[1].strip()
        # TIPS : この関係を取得するクエリを設問からLLMに生成させる方が良いですが、ここではシンプルに
        cypher = f"""
        MATCH (p1:{nodetype})-[r1]->(p2:{destnodetype})
        WHERE p1.id = '{id}'
        RETURN p1.id, p1.text, p2.id, p2.text, r1.id, r1.text, type(r1) as r1_type
        """
        cypher_response.append(graphdb.query(cypher))
    
    noderel_context = ""
    for rels in cypher_response:
        for rel in rels:
            noderel_context += f"""
    =====================================
    "{rel['p1.id']}"は"{rel['p2.id']}"にとって「{rel['r1_type']}」の関係
    "{rel['p1.id']}の概要は{rel['p1.text']}"
    "{rel['p2.id']}の概要は{rel['p2.text']}"
    "双方の関係性の概要は{rel['r1.text']}"
    =====================================
    """
    return  noderel_context


def node_relation_search_generatequery(question:str, nodes, nodetype:str):

    message = [
        SystemMessage(content= f"""

次の質問に回答するために、Graphを探索する実行可能なCypher Qeuryを作成してください。 
必ずvector検索済みのidを利用するように記載します。

vector検索済みのノードは、"{nodetype}"です。
{nodetype}のノードに対しid値にvectorで検索してください。
"作成されたクエリの先頭にcyptherを記載してはならない"

Graph DBのスキーマは以下の通り

-- graph schema -------------------- 
“{graphdb.structured_schema}”


-- node and relationship description --------------------
Practiceノードには、Document内にて言及されている各種実現方式などのプラクティスに関する情報
Considerationノードには、各Practice内にて言及されている考慮事項に関する情報
AzureResourceノードには、各Practiceを実現するために利用できる、Azureのリソース種別名
AzureImplementationノードには、各Practice内にて言及されているAzureでの実装に関する情報

--回答例1---------------------
vector検索済みのノードがAzureResourceの場合
- input :  Azure resourceの LogicAppsに関連したPracticeが知りたい
- output :
// Azure resourceのLogicAppsに関連したPracticeを取得
MATCH (a:Azureresource)-[r:PRACTICETOAZURERESOURCE]-(p:Practice) 
WHERE a.id = 'vector' 
RETURN a.id, a.text, r.id, r.text, p.id, p.text

--回答例2---------------------
vector検索済みのノードがPracticeの場合
- input :  バッチ処理のプラクティスに関して注意すべき事項
- output :  
// バッチ処理のプラクティスに関して注意すべき事項を取得
MATCH (p:Practice)-[r:PRACTICETOCONSIDERATION]-(c:Consideration)
WHERE p.id = 'vector'
RETURN p.id, p.text, r.id, r.text, c.id, c.text

--回答例3---------------------
vector検索済みのノードがConsiderationの場合
- input :  バッチ処理のプラクティスに関して注意すべき事項
- output :  
// バッチ処理のプラクティスに関して注意すべき事項を取得
MATCH (c:Consideration)-[r:PRACTICETOCONSIDERATION]-(p:Practice)
WHERE c.id = 'vector'
RETURN c.id, c.text, r.id, r.text, p.id, p.text
"""),
        HumanMessage(content=question)]
    res = llm.invoke(message)
    query = res.content
#    print(f"# node type : {nodetype}. generated query : {query}")

    cypher_response = []
    for node in nodes:
        id = node.page_content.split('\n')[1].split(': ')[1].strip()

        query_result = cypher_execute_with_retry(query, question, nodetype, nodeid=id)
        print(f"# node type : {nodetype}, id : {id},\n related result :\n")
        pprint.pprint(query_result)        
        cypher_response.append(query_result)

    noderel_context = ""
    for rels in cypher_response:
        for rel in rels:
            noderel_context += str(rel)
    return  noderel_context


def cypher_execute_with_retry(query, question, nodetype, nodeid, retry_count=3):
    """
    Enhanced Cypher execution with better error handling and logging.
    """
    for attempt in range(retry_count):
        try:
            cypher = query.replace('vector', nodeid)
            print(f"# node type : {nodetype}. attempt {attempt + 1}. generated query : {cypher}")
            result = graphdb.query(cypher)
            if result:
                print(f"# Successfully executed query for node {nodeid}, returned {len(result)} results")
            return result
        except Exception as e:
            error_message = str(e)
            print(f"# Error on attempt {attempt + 1} for node {nodeid}: {error_message}")
            
            if attempt == retry_count - 1:
                print(f"# Final attempt failed for node {nodeid}, returning empty result")
                return []
            correction_message = [
                SystemMessage(content= f"""
あなたは、要求をもとにCypherクエリを作成するエージェントです。
次のエラーを修正するためのCypherクエリを作成してください。
必ずvector検索済みのidを利用するように記載します。
vector検索済みのノードは、{nodetype}です。
{nodetype}はvector検索済みのIDが利用でき、nodeのid値にvectorと入力します。
"作成されたクエリの先頭にcyptherを記載してはならない"

出力内容は、そのまま実行可能なCypherクエリである必要があります。
出力内容にはコメントで、クエリの修正内容を記載してください。
"作成されたクエリの先頭にcyptherを記載してはならない"

Graph DBのスキーマは以下の通り

-- graph schema -------------------- 
“{graphdb.structured_schema}”


-- node and relationship description --------------------
Practiceノードには、Document内にて言及されている各種実現方式などのプラクティスに関する情報
Considerationノードには、各Practice内にて言及されている考慮事項に関する情報
AzureResourceノードには、各Practiceを実現するために利用できる、Azureのリソース種別名
AzureImplementationノードには、各Practice内にて言及されているAzureでの実装に関する情報

"""),
                HumanMessage(content=f"""
次のエラーを修正するためのCypherクエリを作成してください。
ユーザからの問い合わせ: {question}
エラーメッセージ: {error_message}
元のクエリ: {cypher}""")]
            correction_res = llm.invoke(correction_message)
            query = correction_res.content
            print(f"# Corrected query for attempt {attempt + 2}: {query}")
            
    return []


def runQA(question: str):
    """
    Enhanced QA function with better error handling and context formatting.
    """
    # 対象ノードタイプを検討
    target_node_types = get_target_node_types(question)
    target_node_types = [node_type.strip() for node_type in target_node_types.split(',')]

    print(f"# hybrid search node types : {target_node_types}")

    noderel_contexts = ""
    total_nodes_found = 0
    
    # 対象ノードをHybrid検索
    for node_type in target_node_types:
        if node_type not in vector_indexies:
            print(f"# Warning: Node type '{node_type}' not found in vector indices")
            continue
            
        nodes = vector_node_search(vector_indexies[node_type], question)
        print(f"# node type : {node_type} vector search result count : {len(nodes)}")
        
        if not nodes:
            print(f"# Warning: No nodes found for type '{node_type}'")
            continue
            
        total_nodes_found += len(nodes)

        # 対象ノード間の関係を取得
        noderel_context = node_relation_search_generatequery(question, nodes, node_type)
        if noderel_context:
            noderel_contexts += f"\n=== {node_type} 関連情報 ===\n{noderel_context}\n"

    if total_nodes_found == 0:
        return "申し訳ございませんが、質問に関連する情報が見つかりませんでした。別の表現で質問していただけますか？"

    message = [
        SystemMessage(content= f"""
あなたは、Azureに関する技術的な質問に答えるエキスパートです。
次の質問に回答するために、必ず与えられたコンテキストだけを利用して正確に回答してください。

**重要な指示:**
1. コンテキスト内の情報のみを使用してください
2. 情報が不足している場合は、その旨を明確に伝えてください
3. 具体的で実用的な回答を心がけてください
4. 関連する実装方法や考慮事項があれば含めてください

**コンテキスト情報の説明:**
- Practice: 実現方式やプラクティスに関する情報
- Consideration: 考慮事項や注意点
- Azureresource: 使用可能なAzureリソース
- Azureimplementation: Azureでの実装方法

**検索されたコンテキスト:**
{noderel_contexts}
"""),
        HumanMessage(content=question)]
    
    res = llm.invoke(message)
    return res.content


def main():
#    question = "Azure LogicAppsに関連したPracticeは？"
#    runQA(question)

    # コンソールからの入力を受け付ける
    while True:
        question = input("質問を入力してください: ")
        answer = runQA(question)
        print("# graph rag result : " + answer)


if __name__ == "__main__":
    main()

