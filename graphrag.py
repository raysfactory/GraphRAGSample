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

# load env file
load_dotenv()


graphdb = Neo4jGraph()
vector_indexies = embed_graph()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# 文章中からエンティティを抽出する
class Entities(BaseModel):
    """エンティティに関する情報の識別"""

    names: List[str] = Field(
        ...,
        description="文章の中に登場する、人物、各人物の性格、各人物間の続柄、各人物が所属する組織、各人物の家族関係",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "テキストから家族と人物のエンティティを抽出してください",
        ),
        (
            "human",
            "指定された形式を使用して、以下から情報を抽出してください"
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

keywords = entity_chain.invoke({"question": "サザエさんの家族構成と人物の性格について教えてください"}).names



def create_full_text_querypart(input: str) -> str:
    """
    フルテキスト検索のクエリ(一部)を作成します。
    与えられた入力を分解し、ANDで接続します
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def structured_retriever(question: str) -> str:
    """
    同名のエンティティを含むNodeを検索し、関連するエンティティを返します
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graphdb.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": create_full_text_querypart(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
#    vector_index.similarity_search(question)
    unstructured_data = [el.page_content for el in vector_indexies["Charactor"].similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data


template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | ChatOpenAI(temperature=0, model_name="gpt-4o")
    | StrOutputParser()
)

chain.invoke("新一の敵は？")


if __name__ == "__main__":
    print(structured_retriever("サザエさんの家族構成と人物の性格について教えてください"))