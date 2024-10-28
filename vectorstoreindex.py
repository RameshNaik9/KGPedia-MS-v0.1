import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from dotenv import load_dotenv
from documentreader import DocumentLoader

# Load documents from the specified directory
documents = DocumentLoader("database").load_documents()
textsplitter = DocumentLoader("database").text_splitter()

load_dotenv()

class VectorStore:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.base_persist_dir = "pinecone index"  # Base directory for all categories

    def create_index(self, index_name, dimension, metric, cloud="aws", region="us-east-1"):
        self.pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=ServerlessSpec(cloud=cloud, region=region))

    def get_pinecone_index(self, host_url):
        """Retrieve the Pinecone index using the provided host URL."""
        return self.pc.Index(host=host_url)

    def get_vector_store(self, host_url):
        """Get the vector store with the specified host URL."""
        pinecone_index = self.get_pinecone_index(host_url)
        return PineconeVectorStore(pinecone_index=pinecone_index)

    def load_vector_index(self, host_url, category):
        persist_dir = os.path.join(self.base_persist_dir, category)  # Create a path for the specific category
        vector_store = self.get_vector_store(host_url, host_url)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
        index = load_index_from_storage(storage_context=storage_context)
        return index

    def create_vector_index(self, documents, text_splitter, host_url, category):
        persist_dir = os.path.join(self.base_persist_dir, category)  # Create a path for the specific category
        vector_store = self.get_vector_store(host_url)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        pc_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, transformations=[text_splitter])
        pc_index.storage_context.persist(persist_dir=persist_dir)
        return pc_index
    
    def update_vector_index(self,documents,index):
        for i in range(len(documents)):
            index.insert(documents[i])
        return index
    