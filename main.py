from data_preprocessing import PreprocessData
from document_creator import CreateDocument
from document_processor import ProcessDocuments
from vector_store import VectorStore
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    cleaner = PreprocessData("./data/ex_data.csv", "./data/cleaned_ex_data.csv")
    cleaner.read_data()
    cleaner.clean_data()
    cleaner.save_cleaned_data()
    
    creator = CreateDocument("./data/cleaned_ex_data.csv")
    creator.read_data()
    creator.create_document()
    try:
        df = creator.convert_document_to_df()
        creator.save_pickle(df)
    except ValueError as e:
        print(e)

    processor = ProcessDocuments("./data/summaries.pkl", "./data/cleaned_ex_data.csv")
    processor.read_data()
    processor.create_documents()
    processor.document_chunking_and_embedding()
    processor.save_nodes("./data/nodes_colab.pkl")

    vs = VectorStore("./data/nodes_colab.pkl", "./data/index_colab.pkl")
    try:
        vs.read_nodes()
        vs.create_vector_store()
        vs.save_index()
    except ValueError as e:
        print(e)