
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )


def format_docs(docs):

    return "\n\n".join(
        [doc.page_content for doc in docs]
    )


def build_rag_chain(transcript: str):

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert video and meeting assistant.

            Answer the user's questions using only the provided transcript context.

            If the answer is not present in the context, respond with:
            "I don't know based on the provided transcript."

            Context:
            {context}
            """
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert video and meeting assistant.

            Answer the user's questions using only the provided transcript context.

            If the answer is not present in the context, respond with:
            "I don't know based on the provided transcript."

            Context:
            {context}
            """
        ),
        ("human", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:

    print(f"\nQuestion: {question}")

    answer = rag_chain.invoke(question)

    print(f"\nAnswer: {answer}")

    return answer
