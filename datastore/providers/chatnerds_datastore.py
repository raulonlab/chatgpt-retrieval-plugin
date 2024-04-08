"""
Chatnerds datastore support for the ChatGPT retrieval plugin.

"""

from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.tracers.stdout import ConsoleCallbackHandler
from chatnerds.langchain.chain_factory import ChainFactory
from chatnerds.config import Config

from datastore.datastore import DataStore
from models.models import (
    DocumentChunk,
    DocumentChunkMetadata,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    Source,
    Query,
    QueryResult,
)

# CHROMA_IN_MEMORY = bool(os.environ.get("CHROMA_IN_MEMORY", "True"))
# CHROMA_PERSISTENCE_DIR = os.environ.get("CHROMA_PERSISTENCE_DIR", "openai")
# CHROMA_HOST = os.environ.get("CHROMA_HOST", "http://127.0.0.1")
# CHROMA_PORT = os.environ.get("CHROMA_PORT", "8000")
# CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "openaiembeddings")

_global_config = Config.environment_instance()

class ChatnerdsDataStore(DataStore):
    def __init__(
        self,
    ):
        self._config = _global_config.get_nerd_config()
        self._retrieve_chain = ChainFactory(self._config).get_pure_retrieve_chain()


    async def query(self, queries: List[Query]) -> List[QueryResult]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        """
        callbacks = []
        if _global_config.VERBOSE > 1:
            callbacks.append(ConsoleCallbackHandler())

        output = await self._retrieve_chain.ainvoke(queries[0].query, config={"callbacks": callbacks})
        if isinstance(output, Dict):
            documents = output.get("documents", [])
        else:
            documents = output
        
        query_results = [{
            "query": queries[0].query,
            "results": [
                {
                    # "id": document.id,
                    "text": document.page_content,
                    "metadata": {
                        "source": "file",
                        "source_id": document.metadata.get("source", None),
                        "url": document.metadata.get("url", None),
                        "created_at": document.metadata.get("created_at", None),
                        "author": document.metadata.get("author", document.metadata.get("artist", None)),
                        "title": document.metadata.get("title", None),
                        "document_id": document.metadata.get("document_id", None),
                    },
                    # "embedding": document.embedding,
                    "score": 1.0,
                }
                for document in documents
            ]
        }]

        return query_results


    def _where_from_query_filter(self, query_filter: DocumentMetadataFilter) -> Dict:
        output = {
            k: v
            for (k, v) in query_filter.dict().items()
            if v is not None and k != "start_date" and k != "end_date" and k != "source"
        }
        if query_filter.source:
            output["source"] = query_filter.source.value
        if query_filter.start_date and query_filter.end_date:
            output["$and"] = [
                {
                    "created_at": {
                        "$gte": int(
                            datetime.fromisoformat(query_filter.start_date).timestamp()
                        )
                    }
                },
                {
                    "created_at": {
                        "$lte": int(
                            datetime.fromisoformat(query_filter.end_date).timestamp()
                        )
                    }
                },
            ]
        elif query_filter.start_date:
            output["created_at"] = {
                "$gte": int(datetime.fromisoformat(query_filter.start_date).timestamp())
            }
        elif query_filter.end_date:
            output["created_at"] = {
                "$lte": int(datetime.fromisoformat(query_filter.end_date).timestamp())
            }

        return output

    def _process_metadata_for_storage(self, metadata: DocumentChunkMetadata) -> Dict:
        stored_metadata = {}
        if metadata.source:
            stored_metadata["source"] = metadata.source.value
        if metadata.source_id:
            stored_metadata["source_id"] = metadata.source_id
        if metadata.url:
            stored_metadata["url"] = metadata.url
        if metadata.created_at:
            try:
                stored_metadata["created_at"] = int(
                    datetime.fromisoformat(metadata.created_at).timestamp()
                )
            except:
                stored_metadata["created_at"] = None
        if metadata.author:
            stored_metadata["author"] = metadata.author
        if metadata.document_id:
            stored_metadata["document_id"] = metadata.document_id

        return stored_metadata

    def _process_metadata_from_storage(self, metadata: Dict) -> DocumentChunkMetadata:
        try:
            created_at=datetime.fromtimestamp(metadata["created_at"]).isoformat() if "created_at" in metadata else None
        except:
            created_at=None

        return DocumentChunkMetadata(
            source=Source(metadata["source"]) if "source" in metadata else None,
            source_id=metadata.get("source", None),
            url=metadata.get("url", None),
            created_at=created_at,
            author=metadata.get("author", metadata.get("artist", None)),
            title=metadata.get("title", None),
        )
    
    
    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        pass

    async def _query(self, queries: List[QueryWithEmbedding]) -> List[QueryResult]:
        pass

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[DocumentMetadataFilter] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        return False

