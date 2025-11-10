import os
import pandas as pd
import json
from PyPDF2 import PdfReader
from langchain_core.documents import Document

def load_documents(folder_path: str):
  
    docs = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        ext = filename.lower().split(".")[-1]

        try:
            
            if ext == "pdf":
                reader = PdfReader(file_path)
               

                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        #creating langchian document obj Adds that document to the list docs.
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": filename, "page": i + 1}
                        ))

            elif ext == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename}
                    ))

            #  CSV Files Pandas library to read data from a Comma Separated Values (CSV) file and load it into a Pandas DataFrame ----
            elif ext == "csv":
                df = pd.read_csv(file_path)
                #Converts that DataFrame into a plain text format (like a table).Useful because the vector store works with text.
             #   Wraps the converted CSV text into a LangChain Document and stores it.
                text = df.to_string()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename}
                ))

            
            elif ext in ["xls", "xlsx"]:
                df = pd.read_excel(file_path)
                text = df.to_string()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename}
                ))

            elif ext == "json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = json.dumps(data, indent=2)
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename}
                    ))

            else:
                print(f"Skipping unsupported file type: {filename}")

        except Exception as e:
            print(f"Error loading {filename}: {e}")

    print(f"\nLoaded {len(docs)} documents from '{folder_path}'")
    return docs
