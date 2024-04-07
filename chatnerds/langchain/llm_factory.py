import io
from contextlib import redirect_stdout
import logging
from typing import Any, Callable, Dict, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.huggingface import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceEmbeddings,
)


class LLMFactory:
    config: Dict[str, Any] = {}
    callback: Optional[Callable[[str], None]] = None

    def __init__(
        self, config: Dict[str, Any], callback: Optional[Callable[[str], None]] = None
    ):
        self.config = config
        self.callback = callback

    def get_embedding_function(self) -> Embeddings:
        embeddings_config = {**self.config["embeddings"]}
        if embeddings_config["model_name"].startswith("hkunlp/") or embeddings_config[
            "model_name"
        ].startswith("BAAI/"):
            provider_class = HuggingFaceInstructEmbeddings
        else:
            provider_class = HuggingFaceEmbeddings

        # capture module stdout and log them as debug level
        trap_stdout = io.StringIO()
        with redirect_stdout(trap_stdout):
            provider_instance = provider_class(**embeddings_config)

        logging.debug(trap_stdout.getvalue())

        return provider_instance

    