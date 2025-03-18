import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

def setup_models():
    """Set up LLM and embedding models"""
    # Set up the Ollama LLM with llama3.2
    llm = Ollama(model="llama3.2", temperature=0.1, request_timeout=120.0)

    # Set up the embedding model using Ollama
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")

    # Configure LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model

def load_chromadb_index(db_path="./chroma_db", collection_name="document_collection"):
    """Load index from ChromaDB"""
    # Set up models
    llm, embed_model = setup_models()
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    try:
        # Get the collection
        chroma_collection = chroma_client.get_collection(name=collection_name)
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
        )
        
        return index
    
    except Exception as e:
        print(f"Error loading ChromaDB index: {e}")
        return None

def ask_question(index, question):
    """Query the index with a question"""
    if not index:
        return "Index not loaded properly. Please check your ChromaDB setup."
    
    # Create query engine
    query_engine = index.as_query_engine()
    
    # Execute query
    response = query_engine.query(question)
    
    return response

if __name__ == "__main__":
    # Load the ChromaDB index
    db_path = "./chroma_db"  # Path to your ChromaDB directory
    collection_name = "document_collection"  # Collection name used during indexing
    
    print(f"Loading ChromaDB index from {db_path}...")
    index = load_chromadb_index(db_path, collection_name)
    
    if index:
        print("ChromaDB index loaded successfully!")
        print("Ask questions about your indexed documents (type 'exit' to quit):")
        
        while True:
            user_question = input("\nYour question: ")
            if user_question.lower() in ['exit', 'quit', 'q']:
                break
                
            print("Querying ChromaDB...")
            response = ask_question(index, user_question)
            print(f"\nResponse: {response}")
    else:
        print("Failed to load ChromaDB index. Please check that it exists and is properly set up.")
