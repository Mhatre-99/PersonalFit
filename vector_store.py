import pickle as pkl
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class VectorStore:
    def __init__(self, nodes_file, index_file):
        self.nodes_file = nodes_file
        self.index_file = index_file
        self.nodes = []
        self.index = None

    def read_nodes(self):
        with open(self.nodes_file, 'rb') as file:
            self.nodes = pkl.load(file)
        return self.nodes

    def create_vector_store(self):
        d = len(self.nodes[0].get_embedding()) 
        if d != 1536:
            raise ValueError("Embedding should be 1536. Something Went Wrong!")
        
        faiss_index = faiss.IndexFlatL2(d)
        
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        self.index = VectorStoreIndex(self.nodes, storage_context)

    def save_index(self):
        with open(self.index_file, 'wb') as file:
            pkl.dump(self.index, file)

