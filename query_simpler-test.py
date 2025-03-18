from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, load_index_from_storage
from llama_index.core.storage import StorageContext

# Set up models
llm = Ollama(model="llama3.2", temperature=0.1, request_timeout=120.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
Settings.llm = llm
Settings.embed_model = embed_model

# Load the existing index
storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
index = load_index_from_storage(storage_context)

# Create a query engine with increased retrieval settings
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Retrieve more nodes
    llm=llm,
    streaming=True
)

# Simple query function with source display
def ask_question(question):
    response = query_engine.query(question)
    
    # Print source information if available
    if hasattr(response, 'source_nodes'):
        print(f"\nBased on {len(response.source_nodes)} sources:")
        for i, node in enumerate(response.source_nodes[:3]):
            print(f"Source {i+1}: {node.node.get_content()[:100]}...")
    
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