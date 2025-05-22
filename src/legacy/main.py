import chromadb
import sys
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import os

from sympy.codegen.fnodes import dimension

# ATTENTION: This code is deprecated and not in use (please don't grade it)
#            Code is only here for documenting our progress

# Configuration
OLLAMA_MODEL = "llama3.2"  # The model name in Ollama
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
CHROMA_DB_DIR = "../database/chroma_db"  # Your Chroma DB path
COLLECTION_NAME = "recipe_database"


def initialize_llm():
    """Initialize the Ollama Llama 3.2 model."""
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        num_predict=2000,
    )
    return llm


def initialize_retriever():
    """Initialize the Chroma retriever for the existing recipe_database."""
    # Setup Ollama embeddings
    embeddings = OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # Connect to existing Chroma DB with the proper client settings
    db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    print(f"Connected to existing collection: {COLLECTION_NAME}")

    # Create a retriever from the DB
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    return retriever


def create_qa_chain(llm, retriever):
    """Create a QA chain with custom prompt template."""
    # Custom prompt template for recipe questions
    template = """
    You are a helpful culinary assistant that answers questions about recipes.

    Context information is below.
    ---------------------
    {context}
    ---------------------

    Given the context information and not prior knowledge, answer the question.
    Format recipes nicely with ingredients lists and step-by-step instructions when appropriate.

    Question: {question}
    Answer: """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


def main():
    print("Initializing Recipe Chat with Ollama's Llama 3.2...")

    # Initialize components
    try:
        llm = initialize_llm()
        retriever = initialize_retriever()
        qa_chain = create_qa_chain(llm, retriever)

        print("\nRecipe Assistant Ready!")
        print("Ask questions about recipes or type 'exit' to quit.\n")

        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            # Get response
            try:
                result = qa_chain({"query": query})
                print("\nAssistant:", result["result"])

                # Uncomment to show source documents
                # print("\nSources:")
                # for i, doc in enumerate(result["source_documents"]):
                #     print(f"Source {i+1}:", doc.metadata.get("source", "Unknown"))

                print()  # Add space between interactions

            except Exception as e:
                print(f"Error during query: {e}")

    except Exception as e:
        print(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()