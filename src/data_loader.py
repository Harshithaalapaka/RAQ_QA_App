import os
import pandas as pd
import json
from PyPDF2 import PdfReader
from langchain_core.documents import Document

def load_documents(folder_path: str):
    """
    Loads PDF, TXT, CSV, Excel, and JSON files from a folder
    and returns a list of LangChain Document objects.
    """
    docs = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        ext = filename.lower().split(".")[-1]

        try:
            # ---- PDF Files ----
            if ext == "pdf":
                reader = PdfReader(file_path)
                #Loops through every page of the PDF.

#enumerate() gives both page index (i) and the page object (page).

                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        #creating langchian document obj Adds that document to the list docs.
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": filename, "page": i + 1}
                        ))

            # ---- Text Files ----
            elif ext == "txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": filename}
                    ))

            # ---- CSV Files Pandas library to read data from a Comma Separated Values (CSV) file and load it into a Pandas DataFrame ----
            elif ext == "csv":
                df = pd.read_csv(file_path)
                #Converts that DataFrame into a plain text format (like a table).Useful because the vector store works with text.
             #   Wraps the converted CSV text into a LangChain Document and stores it.
                text = df.to_string()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename}
                ))

            # ---- Excel Files ----
            elif ext in ["xls", "xlsx"]:
                df = pd.read_excel(file_path)
                text = df.to_string()
                docs.append(Document(
                    page_content=text,
                    metadata={"source": filename}
                ))

            # ---- JSON Files ----
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
"""Returns the final list of all Document objects created.

Each Document is a dict that has:

page_content: text

metadata: info like filename, page number, etc.
 load() is needed to read the JSON file (turn text → Python).

             dumps() is needed to convert it back into a neat text string for LangChain, which only understands text documents.
             here f is file obj and json.load() = “Load JSON file → Convert to Python object”(dict)
               Opens your .json file in read mode ("r") using UTF-8 encoding (so all characters are handled properly).
                   indent=2 means 2 spaces of indentation per level.(line)
                   So json.dumps() = “Dump (convert) Python object → JSON text string”
                   
                   dict = Python’s internal data structure (used inside your code).
                  JSON = Text format for storing or transferring that data outside your code.
                jSON = string version of a dictionary (following strict syntax rules).
                Each format requires a different library to read it correctly.

           A PDF’s text is on pages.

            A CSV’s text is in cells.

             A JSON’s text is in nested keys.

             A TXT’s text is already plain"""
