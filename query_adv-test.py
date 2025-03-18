from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.retrievers.bm25 import BM25Retriever  # This import path is more reliable
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Set up models
llm = Ollama(model="llama3.2", temperature=0.1, request_timeout=120.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
Settings.llm = llm
Settings.embed_model = embed_model

# Load the existing index
storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
index = load_index_from_storage(storage_context)

# Create vector retriever (semantic search)
vector_retriever = index.as_retriever(similarity_top_k=10)

# Create BM25 retriever (keyword search)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=10
)

# Combine both retrievers
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,  # Final number of results
    num_queries=4,
    use_async=True,
    verbose=True
)

# Create query engine with the hybrid retriever
#query_engine = index.as_query_engine(
 #   retriever=hybrid_retriever,
  #  llm=llm,
   # streaming=True
#)

query_engine = RetrieverQueryEngine.from_args(hybrid_retriever)

# Function to query with source display
def ask_question(question):
    response = query_engine.query(question)
    
    # Print source information if available
    if hasattr(response, 'source_nodes'):
        print(f"\nBased on {len(response.source_nodes)} sources:")
        for i, node in enumerate(response.source_nodes[:3]):
            content = node.node.get_content()
            preview = content[:100].replace('\n', ' ')
            print(f"Source {i+1}: {preview}...")
    
    return response

# Example usage
if __name__ == "__main__":
    print("Ask questions about your indexed documents (type 'exit' to quit):")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ['exit', 'quit', 'q']:
            break
            
        response = ask_question(user_question)
        print(f"\nResponse: {response}")