from langchain_neo4j import Neo4jGraph, Neo4jVector
from typing import Dict, List

node_labels = ["Plactice", "Consideration", "Azureresource", "Azureimprementation"]
relationship_labels = ["PlacticeToConsideration", "PlacticeToAzureResource", "AzureResourceToAzureImprementation", "PlacticeToAzureImprementation"]


def embed_graph(embedding) -> Dict[str, Neo4jVector]:
    index_dict = {}
    index_dict["Document"] = _embed_index(embedding, "Document")
    for node_label in node_labels:
        index_dict[node_label] = _embed_index(embedding, node_label)
    return index_dict


def _embed_index(embedding, node_label: str) -> Neo4jVector:
    return Neo4jVector.from_existing_graph(
        embedding=embedding,
        index_name=f"{node_label}_vector",
        keyword_index_name=f"{node_label}_keyword",
        search_type="hybrid",
        node_label=node_label,
        text_node_properties=["id", "text"],
        embedding_node_property="embedding"
    )
