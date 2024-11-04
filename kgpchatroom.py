import os
from dotenv import load_dotenv
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.gemini import Gemini
from pinecone import Pinecone
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import Stemmer

load_dotenv()

class KGPChatroomModel:
    """
    Responsible for configuring and providing models and embeddings.
    Adheres to SRP by focusing solely on model management.
    """
    def __init__(self, pinecone_api_key=None, google_api_key=None):
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")

        if not self.pinecone_api_key or not self.google_api_key:
            raise ValueError("No API keys??")

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.base_persist_dir = "pinecone index"
        self.keyword_persist_dir = "keyword index"

        # Map from chat_profile to host_url
        self.chat_profile_url_map = {
            "Career": "https://test1nov-r1ur7n2.svc.aped-4627-b74a.pinecone.io",
            "Academic": "https://profilex-chatroom-url.pinecone.io",
            "Bhaat": "https://profiley-chatroom-url.pinecone.io",
            "Gymkhana": "https://profilez-chatroom-url.pinecone.io",
            # Add more mappings as needed
        }

        self.configure_models()

    def configure_models(self):
        """Configure the embedding and LLM models."""
        Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004",api_key=self.google_api_key)
        Settings.llm = Gemini(model_name="models/gemini-1.5-flash-002",temperature=1,api_key=self.google_api_key)

    def get_model(self):
        """Return the configured LLM model."""
        return Settings.llm

    def get_embedding_model(self):
        """Return the configured embedding model."""
        return Settings.embed_model

    def get_pinecone_index(self, chat_profile):
        """Retrieve the Pinecone index based on the chat_profile."""
        host_url = self.chat_profile_url_map.get(chat_profile)
        if not host_url:
            raise ValueError(f"Host URL for chat profile {chat_profile} not found!")
        return self.pc.Index(host=host_url)

    def get_vector_store(self, chat_profile):
        """Get the vector store based on the chat_profile."""
        pinecone_index = self.get_pinecone_index(chat_profile)
        return PineconeVectorStore(pinecone_index=pinecone_index)

    def load_vector_index(self, chat_profile):
        """Load the vector index for a specific chat profile."""
        persist_dir = os.path.join(self.base_persist_dir, chat_profile)  # Create a path for the specific category
        vector_store = self.get_vector_store(chat_profile)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
        index = load_index_from_storage(storage_context=storage_context)
        return index
    
    def load_keyword_index(self,chat_profile):
        """Load the keyword index for a specific chat profile."""
        persist_dir = os.path.join(self.keyword_persist_dir, chat_profile)  # Create a path for the specific category
        keyword_storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        keyword_index = load_index_from_storage(keyword_storage_context)
        return keyword_index
    
    def get_retriever(self, chat_profile, similarity_top_k=10):
        """Get the retriever for a specific chat profile."""
        vector_retriever = self.load_vector_index(chat_profile).as_retriever(similarity_top_k=similarity_top_k)
        bm25_retriever = BM25Retriever.from_defaults(index = self.load_keyword_index(chat_profile),similarity_top_k=3,stemmer=Stemmer.Stemmer('english'), language='english', verbose=True)
        retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=3,  # Reduced from 5 for more focused retrieval
    mode="reciprocal_rerank",
    use_async=False,  # Enable async for better performance
    verbose=True
)
        return retriever

