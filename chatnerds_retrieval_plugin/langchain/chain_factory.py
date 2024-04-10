from typing import Any, Dict, Optional
import io
from contextlib import redirect_stdout
import logging
from langchain.chains.base import Chain
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.embeddings import Embeddings
from chatnerds_retrieval_plugin.stores.store_factory import StoreFactory
from chatnerds_retrieval_plugin.langchain.chain_runnables import (
    retrieve_relevant_documents_runnable,
    get_parent_documents_runnable,
    combine_documents_runnable,
)

class ChainFactory:
    config: Dict[str, Any] = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_pure_retrieve_chain(self) -> Chain:
        embeddings = self.get_embedding_function()

        store_factory = StoreFactory(self.config)
        retrieve_store = store_factory.get_vector_store(embeddings=embeddings)

        retriever = retrieve_store.as_retriever(**self.config["retriever"])

        retrieve_chain_config = self.config.get("retrieve_chain", None)
        if (
            isinstance(retrieve_chain_config, str)
            and retrieve_chain_config in self.config
        ):
            retrieve_chain_config = self.config.get(retrieve_chain_config, None)

        if not isinstance(retrieve_chain_config, dict):
            raise ValueError(
                f"Invalid value in 'retrieve_chain' configuration: {retrieve_chain_config}"
            )

        retrieve_relevant_documents = RunnableParallel(
            documents=retrieve_relevant_documents_runnable.bind(
                retriever=retriever, **retrieve_chain_config
            )
            | get_parent_documents_runnable.bind(
                store=retrieve_store, **retrieve_chain_config
            ),
            question=RunnablePassthrough(),
        )

        return retrieve_relevant_documents


    def get_embedding_function(self) -> Embeddings:
        embeddings_config = {**self.config["embeddings"]}
        
        model_name = str(embeddings_config["model_name"]).lower()
        provider = str(embeddings_config.get("provider", "huggingface")).lower()
        model_kwargs = embeddings_config.get("model_kwargs", {})
        encode_kwargs = embeddings_config.get("encode_kwargs", {})

        if provider == "huggingface":
            from langchain_community.embeddings.huggingface import (
                HuggingFaceInstructEmbeddings,
                HuggingFaceEmbeddings,
                HuggingFaceBgeEmbeddings,
            )
            if "instructor" in model_name:
                provider_class = HuggingFaceInstructEmbeddings
            elif "bge" in model_name:
                provider_class = HuggingFaceBgeEmbeddings
            else:
                provider_class = HuggingFaceEmbeddings

            # capture module stdout and log them as debug level
            trap_stdout = io.StringIO()
            with redirect_stdout(trap_stdout):
                provider_instance = provider_class(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

            logging.debug(trap_stdout.getvalue())

            return provider_instance
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            openai_instance = OpenAIEmbeddings(model=model_name, model_kwargs=model_kwargs)

            return openai_instance
        else:
            raise ValueError(f"Unknown embeddings provider '{provider}'. Please use 'huggingface' or 'openai'.")
