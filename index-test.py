from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser

# Set up the embedding model using Ollama
embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")

# Set up the Ollama LLM
llm = Ollama(model="llama3.2", temperature=0.1, request_timeout=120.0)

# Configure LlamaIndex settings
Settings.llm = llm
Settings.embed_model = embed_model

# Load documents without specifying extractors (rely on built-in defaults)
try:
    documents = SimpleDirectoryReader("./data", recursive=True).load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Print loaded document information
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc.metadata.get('file_name', 'Unknown')}")
    
    # Node parser - split documents into chunks
    node_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
    
    # Create the index with the node parser
    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser
    )
    
    # Save the index
    index.storage_context.persist("./index_storage")
    print("Index created and saved successfully!")
    
except Exception as e:
    print(f"Error during indexing: {e}")