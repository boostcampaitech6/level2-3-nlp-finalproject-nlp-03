"""**Retriever** class returns Documents given a text **query**.

It is more general than a vector store. A retriever does not need to be able to
store documents, only to return (or retrieve) it. Vector stores can be used as
the backbone of a retriever, but there are other types of retrievers as well.

**Class hierarchy:**

.. code-block::

    BaseRetriever --> <name>Retriever  # Examples: ArxivRetriever, MergerRetriever

**Main helpers:**

.. code-block::

    RetrieverInput, RetrieverOutput, RetrieverLike, RetrieverOutputLike,
    Document, Serializable, Callbacks,
    CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import warnings
from abc import ABC, abstractmethod
from inspect import signature

from langchain_core.documents import Document
from langchain_core.load.dump import dumpd
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
        Callbacks,
    )

RetrieverInput = str
RetrieverOutput = List[Document]
RetrieverLike = Runnable[RetrieverInput, RetrieverOutput]
RetrieverOutputLike = Runnable[Any, RetrieverOutput]

from langchain.retrievers.self_query.base import SelfQueryRetriever


class ReorderSelfQueryRetriever(SelfQueryRetriever):
    async def aget_relevant_documents(
        self,
        query: str,
        *,
        callbacks: Callbacks = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: string to find relevant documents for
            callbacks: Callback manager or list of callbacks
            tags: Optional list of tags associated with the retriever. Defaults to None
                These tags will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
            metadata: Optional metadata associated with the retriever. Defaults to None
                This metadata will be associated with each call to this retriever,
                and passed as arguments to the handlers defined in `callbacks`.
        Returns:
            List of relevant documents
        """
        from langchain_core.callbacks.manager import AsyncCallbackManager

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=tags,
            local_tags=self.tags,
            inheritable_metadata=metadata,
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            dumpd(self),
            query,
            name=run_name,
        )
        try:
            _kwargs = kwargs if self._expects_other_args else {}
            if self._new_arg_supported:
                result = self._get_relevant_documents(query, run_manager=run_manager, **_kwargs)
            else:
                result = self._get_relevant_documents(query, **_kwargs)
        except Exception as e:
            # 에러 발생 시 처리
            run_manager.on_retriever_error(e)
            raise e
        else:
            from langchain_community.document_transformers import LongContextReorder

            reordering = LongContextReorder()
            reordered_result = reordering.transform_documents(result)

            # 검색 종료 이벤트 처리
            run_manager.on_retriever_end(
                reordered_result,  # 재정렬된 결과를 이벤트 처리에 사용
            )
            return reordered_result  # 재정렬된 결과 반환
