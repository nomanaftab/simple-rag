import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

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

def create_enhanced_retriever(index, top_k=10):
    """Create an enhanced vector retriever with improved settings"""
    if not index:
        return None
        
    # Create vector retriever with optimized settings
    vector_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
        # Additional parameters for better retrieval
        alpha=0.5,  # Balance between relevance and recency (if your nodes have timestamps)
        vector_store_query_mode="mmr",  # Maximum Marginal Relevance for diversity
    )
    
    return vector_retriever

def ask_question_enhanced(index, question, top_k=10):
    """Query the index with enhanced vector search approach"""
    if not index:
        return "Index not loaded properly. Please check your ChromaDB setup."
    
    # Create enhanced retriever
    enhanced_retriever = create_enhanced_retriever(index, top_k)
    
    # Create query engine with the enhanced retriever
    query_engine = RetrieverQueryEngine.from_args(
        retriever=enhanced_retriever,
        response_synthesizer_kwargs={
            "verbose": True,
            "node_postprocessors": []  # You can add custom node processors here
        }
    )
    
    # Execute query
    response = query_engine.query(question)
    
    # Print source information if available
    if hasattr(response, 'source_nodes'):
        print(f"\nBased on {len(response.source_nodes)} sources:")
        for i, node in enumerate(response.source_nodes[:3]):
            content = node.node.get_content()
            preview = content[:100].replace('\n', ' ')
            source = node.node.metadata.get("source", "Unknown")
            print(f"Source {i+1} ({source}): {preview}...")
    
    return response

if __name__ == "__main__":
    # Load the ChromaDB index
    db_path = "./chroma_db"  # Path to your ChromaDB directory
    collection_name = "document_collection"  # Collection name used during indexing
    
    print(f"Loading ChromaDB index from {db_path}...")
    index = load_chromadb_index(db_path, collection_name)
    
    if index:
        print("ChromaDB index loaded successfully!")
        print("Using enhanced vector retrieval with Maximum Marginal Relevance")
        print("Ask questions about your indexed documents (type 'exit' to quit):")
        
        while True:
            user_question = input("\nYour question: ")
            if user_question.lower() in ['exit', 'quit', 'q']:
                break
                
            print("Querying ChromaDB with enhanced search...")
            try:
                response = ask_question_enhanced(index, user_question)
                print(f"\nResponse: {response}")
            except Exception as e:
                print(f"Error during retrieval: {e}")
                print("Trying with default retrieval method...")
                # Fall back to the simplest retrieval method
                query_engine = index.as_query_engine()
                response = query_engine.query(user_question)
                print(f"\nResponse: {response}")
    else:
        print("Failed to load ChromaDB index. Please check that it exists and is properly set up.")