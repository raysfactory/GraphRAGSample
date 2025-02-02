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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


from dotenv import load_dotenv
from graph_creation import embed_graph
from pprint import pprint


# load env file
load_dotenv()


graphdb = Neo4jGraph()
graphdb.refresh_schema()  # DBからスキーマを取得

vector_indexies = embed_graph()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")



def vector_node_search(vector_index: Neo4jVector, query:str):    
    # クエリを設定して検索を実行
    # (このサンプルではVector Index作成する際にHybrid Indexを作成しているため、Hybrid Indexを使用して検索を行う)
    documents = vector_index.similarity_search(query, k=2)
    return documents




def get_target_node_types(question:str):
    message = [
        SystemMessage(content= """
        次の質問に回答するために、Graph内のどのノードをVector検索するかを教えてください。
        また回答するためにノード間の関係を取得するための対象のノードタイプも教えてください。
        また、複数の関係を指定する場合は、カンマ区切りで指定してください。
        
                      
        Graph DBのスキーマは以下の通り
        -- graph schema -------------------- 
        “{graphdb.structured_schema}”
        -- schema description --------------------
            Charactor(作品登場人物)
            Actor（俳優声優
            Document(グラフ作成元のドキュメント) 
        

        -----------------------
            例: 
            - input :  サザエさんは誰と結婚しているのか？
            - output :  Charactor->Charactor
            例: 
            - input :  マスオさんの声優は？
            - output :  Actor->Charactor
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


def main():

    question = "サザエさんは誰と結婚しているのか？"

    # 対象ノードタイプを検討
    target_node_types = get_target_node_types(question)
    target_node_types = target_node_types.split(',')
    # 対象ノードをHybrid検索
    for node_type in target_node_types:
        node_types = node_type.split('->')
        target_node = node_types[0]
        dest_node = node_types[1]

        nodes = vector_node_search(vector_indexies[node_types[0]], question)

        # 対象ノード間の関係を取得
        noderel_context = node_relation_search(nodes, target_node, dest_node)
        print(noderel_context)

    message = [
        SystemMessage(content= """
        次の質問に回答するために、与えられたコンテキストをもとに回答してください
                      
        -- context --------------------
        "{noderel_context}"
        """),
        HumanMessage(content=question)]
    res = llm.invoke(message)

    print(res)


if __name__ == "__main__":
    main()

