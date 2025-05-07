import os
import logging
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_models_and_db(db_path):
    """
    Load and return the Chroma database and LLM.

    Args:
        db_path (str): Path to the Chroma database directory.

    Returns:
        tuple: (vector_store, llm)
    """
    try:
        logging.info("Initializing models and database connections")

        # Initialize the embeddings model
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        # Connect to the existing Chroma database
        vector_store = Chroma(
            collection_name="recipe_database",
            persist_directory=db_path,
            embedding_function=embeddings
        )

        # Initialize the LLM
        llm = OllamaLLM(model="llama3.2")

        return vector_store, llm

    except Exception as e:
        logging.error(f"Error loading models and DB: {str(e)}")
        raise


def enhance_query(llm, user_query):
    """
    Use the LLM to enhance/refine the user query.

    Args:
        llm: The LLM model
        user_query (str): The original user query

    Returns:
        str: Enhanced query
    """
    template = """You are a search query enhancement assistant.
Your task is to refine the user's query to improve vector database search results.
Maintain the original meaning but expand with relevant keywords that might help with retrieval.
Keep it concise (under 100 characters).

Original query: {query}

Enhanced query:"""

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        enhanced_query = chain.run(query=user_query).strip()
        logging.info(f"Enhanced query: '{enhanced_query}'")
        return enhanced_query
    except Exception as e:
        logging.error(f"Error enhancing query: {str(e)}")
        return user_query


def query_chroma_db(vector_store, query_text, num_results=1):
    """
    Query the Chroma database with the given text and return results.

    Args:
        vector_store (Chroma): The Chroma vector store.
        query_text (str): The text query to search for.
        num_results (int): Number of results to return.

    Returns:
        list: The most similar documents with their content and scores.
    """
    try:
        logging.info(f"Querying database for: '{query_text}'")

        # Perform similarity search
        results = vector_store.similarity_search_with_score(
            query=query_text,
            k=num_results
        )

        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            })

        return formatted_results

    except Exception as e:
        logging.error(f"Error querying Chroma DB: {str(e)}")
        return []


def generate_content_summary(llm, content, query):
    """
    Use the LLM to generate a helpful summary of the content based on the query.

    Args:
        llm: The LLM model
        content (str): The content to summarize
        query (str): The original query

    Returns:
        str: A summary of the content
    """
    template = """You are a helpful AI assistant specializing in summarizing content.
Given a user's query and a piece of content, provide a concise and relevant summary that directly addresses what the user was looking for.

USER QUERY: {query}

CONTENT:
{content}

Your task:
1. Summarize the key information from the content that is relevant to the query (3-5 sentences)
2. Extract any specific details that directly answer the user's question
3. Format your response in a clear, easy-to-read manner

SUMMARY:"""

    prompt = PromptTemplate(
        input_variables=["query", "content"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        summary = chain.run(query=query, content=content).strip()
        return summary
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        return "Error generating summary."


def suggest_next_queries(llm, content, current_query):
    """
    Use the LLM to suggest related queries the user might want to try next.

    Args:
        llm: The LLM model
        content (str): The content to analyze
        current_query (str): The current query

    Returns:
        list: Suggested next queries
    """
    template = """Based on the user's current query and the content they just viewed, suggest 3 follow-up queries 
they might find useful for exploring related information.

CURRENT QUERY: {current_query}

CONTENT THEY VIEWED:
{content}

Provide exactly 3 suggested follow-up queries that:
1. Are related to but different from the current query
2. Might help the user explore the topic further
3. Are phrased as complete search queries

SUGGESTED QUERIES:
1."""

    prompt = PromptTemplate(
        input_variables=["current_query", "content"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        suggestions = chain.run(current_query=current_query, content=content).strip()

        # Process the output into a list
        suggestion_list = []
        for line in suggestions.split('\n'):
            line = line.strip()
            # Remove numbering if present
            if line and (line[0].isdigit() and line[1:3] in ['. ', '- ']) or line.startswith('- '):
                line = line[line.find(' ') + 1:].strip()
                suggestion_list.append(line)
            elif line and not line.startswith('SUGGESTED QUERIES:'):
                suggestion_list.append(line)

        # Take only the first 3 suggestions
        return suggestion_list[:3]
    except Exception as e:
        logging.error(f"Error generating query suggestions: {str(e)}")
        return ["No suggestions available."]


def print_enhanced_result(result, query, llm):
    """Print a single result with enhanced information."""
    if not result:
        print("\nNo matching results found.")
        return

    print(f"\n{'=' * 80}")
    print(f"🔍 Search Results for: '{query}'")
    print(f"{'=' * 80}")

    # Generate and display the summary
    content_summary = generate_content_summary(llm, result["content"], query)
    print("\n📝 SUMMARY:")
    print(f"{content_summary}")

    # Display relevance score
    print(f"\nRelevance Score: {result['similarity_score']:.4f}")

    # Display original content (shortened for readability)
    content_preview = result["content"]
    if len(content_preview) > 500:
        content_preview = content_preview[:500] + "..."

    print("\nCONTENT PREVIEW:")
    print(f"{content_preview}")

    # Display metadata
    print("\nMETADATA:")
    print(f"{result['metadata']}")

    # Generate and display suggested next queries
    suggested_queries = suggest_next_queries(llm, result["content"], query)
    print("\nYOU MIGHT ALSO WANT TO ASK:")
    for i, suggestion in enumerate(suggested_queries, 1):
        print(f"  {i}. {suggestion}")

    print(f"{'=' * 80}")


def interactive_query_loop(db_path):
    """Run an enhanced interactive query loop."""
    try:
        # Load the database and LLM once
        vector_store, llm = load_models_and_db(db_path)

        print("\n ENHANCED CHROMA DB QUERY ")
        print("Type your query and press Enter to search.")
        print("To exit, type 'quit', 'exit', or press Ctrl+C.")
        print("For raw results without LLM enhancement, start your query with 'raw:'")
        print("To use a suggested query, type its number (1-3)")
        print("=" * 50)

        last_suggestions = []

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

            # Check if user wants to use a suggested query
            if query in ['1', '2', '3'] and last_suggestions and int(query) <= len(last_suggestions):
                query = last_suggestions[int(query) - 1]
                print(f"Using suggested query: '{query}'")

            # Determine if we should use raw mode
            raw_mode = False
            if query.lower().startswith('raw:'):
                raw_mode = True
                query = query[4:].strip()

            # Enhance the query unless in raw mode
            if not raw_mode:
                enhanced_query = enhance_query(llm, query)
            else:
                enhanced_query = query

            # Get results
            results = query_chroma_db(vector_store, enhanced_query, num_results=1)

            if not results:
                print("\nNo matching results found.")
                continue

            # Display the result
            if raw_mode:
                # Simple display for raw mode
                print(f"\nRaw Results for: '{query}'")
                print("=" * 80)
                print(f"Score: {results[0]['similarity_score']:.4f}")
                print(f"Content: {results[0]['content']}")
                print(f"Metadata: {results[0]['metadata']}")
                print("=" * 80)
            else:
                # Enhanced display with LLM summaries and suggestions
                print_enhanced_result(results[0], query, llm)

                # Store suggestions for next iteration
                last_suggestions = suggest_next_queries(llm, results[0]["content"], query)

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