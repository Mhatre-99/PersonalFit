import pandas as pd
import pickle as pkl
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, Document, VectorStoreIndex
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import LangchainNodeParser

class ProcessDocuments:
    def __init__(self, file_path, data_path, model='text-embedding-3-small'):
        self.file_path = file_path
        self.model = OpenAIEmbedding(model=model)
        Settings.embed_model = self.model
        self.documents = []
        self.nodes = []
        self.rag_data=None
        self.data=None
        self.data_path = data_path

    def read_data(self):
        with open(self.file_path, 'rb') as file:
            self.rag_data = pkl.load(file)
        self.rag_data = self.rag_data.iloc[:1381] 
        if len(self.rag_data) != 1381:
            raise ValueError("Documents created should be 1381. Something Went Wrong!")
        
        self.data = pd.read_csv(self.data_path)

    def create_documents(self):
        for i in range(self.rag_data.shape[0]):
            text = "Summary: "+ self.rag_data["Summary"][i]
            document = Document(
                text=text,
                metadata={
                    "exercise": self.data['Exercise'][i],
                    "Target Muscle Group": self.data['Target Muscle Group'][i],
                    "Difficulty Level": self.data['Difficulty Level'][i],
                    "Primary Equipment": self.data['Primary Equipment'][i],
                    "Body Region": self.data['Body Region'][i],
                    "Prime Mover Muscle": self.data['Prime Mover Muscle'][i],
                    "Posture": self.data['Posture'][i],
                    "Arms involved": self.data['Single or Double Arm'][i],
                    "Arm movement": self.data['Continuous or Alternating Arms'][i],
                    "Grip": self.data['Grip'][i],
                    "Movement Pattern": self.data['Movement Pattern'][i],
                    "Force Type": self.data['Force Type'][i]
                }
            )
            self.documents.append(document)
        if len(self.documents) != 1381:
            raise ValueError("Documents created should be 1381. Something Went Wrong!")

    def document_chunking_and_embedding(self):
        MARKDOWN_SEPARATORS = [
            "\n#{1,6} ",
            "```\n",
            "\n\\*\\*\\*+\n",
            "\n---+\n",
            "\n___+\n",
            "\n\n",
            "\n",
            " ",
            ""
        ]
        print("Number of Documents: ", len(self.documents))
        print("A",repr(self.documents[0])," A")
        parser = LangchainNodeParser(RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
            is_separator_regex=True
        ))
        self.nodes = parser.get_nodes_from_documents(self.documents)
        print("Number of nodes: ", len(self.nodes))
        for i, node in enumerate(self.nodes):
            if i % 100 == 0:
                print(f"Embedding node {i}")
            node.embedding = Settings.embed_model.get_text_embedding(node.text)

    def save_nodes(self, output_path):
        with open(output_path, 'wb') as file:
            pkl.dump(self.nodes, file)

