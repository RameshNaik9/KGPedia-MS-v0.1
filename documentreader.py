import os
import docx2txt
from typing import List
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.json import JSONReader
from llama_index.core.node_parser import SentenceSplitter  # Adjusted import for SentenceSplitter

class DocumentLoader:
    """Document loader to load files from a directory into documents."""

    def __init__(self, input_dir):
        self.input_dir = input_dir

    def load_documents(self) -> List[Document]:
        """Loads documents from the directory, handling .json files separately from other formats."""
        documents = []

        # Iterate through all files in the directory
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # If the file is a .json, use JSONReader
                if file.lower().endswith('.json'):
                    json_reader = JSONReader()
                    try:
                        # Read the JSON file and load it as a Document
                        json_documents = json_reader.load_data(file_path)
                        documents.extend(json_documents)
                    except Exception as e:
                        print(f"Error loading JSON file {file}: {e}")

                # For all other formats, use SimpleDirectoryReader
                else:
                    try:
                        simple_reader = SimpleDirectoryReader(input_dir=self.input_dir)
                        other_documents = simple_reader.load_data()
                        documents.extend(other_documents)
                    except Exception as e:
                        print(f"Error loading file {file}: {e}")

        return documents

    def text_splitter(self):
        """Splits the text content of a document into sentences."""
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
        return text_splitter
