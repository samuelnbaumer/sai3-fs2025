import os
import logging
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_parquet_files(dataset_dir):
    """Load and combine parquet files from the dataset directory."""
    logging.info(f"Loading parquet files from {dataset_dir}")
    
    parquet_files = [
        "train-00000-of-00004-237b1b1141fdcfa1.parquet",
        "train-00001-of-00004-d46654ac93566129.parquet",
        "train-00002-of-00004-3b4f78b99eedadc2.parquet",
        "train-00003-of-00004-2369b90eb0860a76.parquet"
    ]
    
    dfs = []
    for file in parquet_files:
        file_path = os.path.join(dataset_dir, file)
        try:
            df = pd.read_parquet(file_path)
            logging.info(f"Successfully loaded {file}")
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error loading {file}: {str(e)}")
            raise
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Combined dataset size: {len(combined_df)} rows")
    return combined_df

def create_vector_store(df, db_path):
    """Create and populate the vector store with documents."""
    logging.info("Initializing vector store...")
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Create vector store
    vector_store = Chroma(
        collection_name="recipe_database",
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    # Process documents
    logging.info("Processing documents...")
    documents = []
    ids = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating documents"):
        try:
            recipe_text = row["input"]
            doc = Document(
                page_content=recipe_text,
                metadata={"id": str(idx)},
                id=str(idx)
            )
            documents.append(doc)
            ids.append(str(idx))
            
            # Process in batches of 100
            if len(documents) >= 100:
                vector_store.add_documents(documents=documents, ids=ids)
                documents = []
                ids = []
                
        except Exception as e:
            logging.error(f"Error processing document {idx}: {str(e)}")
            continue
    
    # Add any remaining documents
    if documents:
        vector_store.add_documents(documents=documents, ids=ids)
    
    logging.info("Vector store creation completed")
    return vector_store

def main():
    try:
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(os.path.dirname(current_dir), "datasets")
        db_path = os.path.join(os.path.dirname(current_dir), "database", "chroma_db")
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Load data
        df = load_parquet_files(dataset_dir)
        
        # Create vector store
        vector_store = create_vector_store(df, db_path)
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        logging.info("Vectorization process completed successfully")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()