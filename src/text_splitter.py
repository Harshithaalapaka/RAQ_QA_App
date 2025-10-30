from langchain_text_splitters import RecursiveCharacterTextSplitter
def split_documents(documents,chunk_size=500,chunk_overlap=50):#this is our defined func
        #if isinstance(documents[0],str):
             #documents=[Document(page_content=doc,metadata={})for doc in documents]
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n","\n"," ",""]
        )
        #split_documents method  belongs to the text_splitter object i.e., it is a inbuilt_method of RCTS cls from lanchain.
        #so when we creating obj of rcts cls we are inheriting the methods of that cls into obj
        """
       Why your function(split_documents) is useful

Handles raw text strings automatically (so you don’t have to manually convert to Document objects).

Sets default chunking rules (chunk size, overlap, separators) in one place.

Returns ready-to-use chunks for your RAG pipeline.
        """
        chunks=text_splitter.split_documents(documents)#this is inbuilt func
        print(f"split{len(documents)} documents into {len(chunks)} chunks")

        '''if chunks:
               print(f"\n example chunk:")
               print(f"content:{chunks[0].page_content[:200]}...")
               print(f"metadata:{chunks[0].metadata}")'''
        return chunks