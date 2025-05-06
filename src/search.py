import os
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_chroma_db(db_path):
    """
    Load and return the Chroma database.

    Args:
        db_path (str): Path to the Chroma database directory.

    Returns:
        Chroma: The initialized vector store.
    """
    try:
        logging.info(f"Initializing connection to Chroma DB at {db_path}")

        # Initialize the same embeddings model used during creation
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        # Connect to the existing Chroma database
        vector_store = Chroma(
            collection_name="recipe_database",
            persist_directory=db_path,
            embedding_function=embeddings
        )

        return vector_store

    except Exception as e:
        logging.error(f"Error loading Chroma DB: {str(e)}")
        raise


def query_single_result(vector_store, query_text):
    """
    Query the Chroma database with the given text and return one result.

    Args:
        vector_store (Chroma): The Chroma vector store.
        query_text (str): The text query to search for.

    Returns:
        dict: The most similar document with its content and score.
    """
    try:
        logging.info(f"Querying database for: '{query_text}'")

        # Perform similarity search for just one result
        results = vector_store.similarity_search_with_score(
            query=query_text,
            k=1
        )

        if not results:
            return None

        doc, score = results[0]
        return {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": score
        }

    except Exception as e:
        logging.error(f"Error querying Chroma DB: {str(e)}")
        return None


def print_result(result, query):
    """Print a single result in a formatted way."""
    if not result:
        print("\nNo matching results found.")
        return

    print(f"\nSearch Results for: '{query}'")
    print("=" * 80)
    print(f"Score: {result['similarity_score']:.4f}")
    print("-" * 80)
    print(f"Content: {result['content']}")
    print("-" * 80)
    print(f"Metadata: {result['metadata']}")
    print("=" * 80)


def interactive_query_loop(db_path):
    """Run an interactive query loop for the Chroma database."""
    try:
        # Load the database once
        vector_store = load_chroma_db(db_path)

        print("\n===== INTERACTIVE CHROMA DB QUERY =====")
        print("Type your query and press Enter to search.")
        print("To exit, type 'quit', 'exit', or press Ctrl+C.")
        print("=======================================\n")

        while True:
            # Get user query
            query = input("\nEnter your query: ").strip()

            # Check for exit commands
            if query.lower() in ('quit', 'exit', 'q'):
                print("Exiting. Goodbye!")
                break

            # Skip empty queries
            if not query:
                print("Please enter a valid query.")
                continue

            # Get and print one result
            result = query_single_result(vector_store, query)
            print_result(result, query)

    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


def main():
    # Set the path to your Chroma database
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(os.path.dirname(current_dir), "database", "chroma_db")

    # Start the interactive query loop
    interactive_query_loop(db_path)


if __name__ == "__main__":
    main()