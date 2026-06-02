import os
import sqlite3
from typing import List
from pathlib import Path

# LangChain Document Loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

# Text Splitting & Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# SQLite Vector Store
from langchain_community.vectorstores import SQLiteVec
import sqlite_vec

# Import the MaterialFile model from our previous material_manager
from material_manager import MaterialFile

class RAGManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except AttributeError:
            print("⚠️ Python's sqlite3 module doesn't support extension loading.")
            raise
        return conn

    def _init_tracking_table(self, conn: sqlite3.Connection):
        """FIX: Keeps a log of ingested files to prevent duplication."""
        conn.execute("CREATE TABLE IF NOT EXISTS Ingested_Files (file_path TEXT UNIQUE, course_id TEXT)")
        conn.commit()

    def _load_file(self, file: MaterialFile) -> List[Document]:
        if not os.path.exists(file.file_path): return []
        try:
            if file.extension == ".pdf":
                return PyPDFLoader(file.file_path).load()
            elif file.extension in [".txt", ".md", ".csv"]:
                return TextLoader(file.file_path, encoding="utf-8").load()
        except Exception as e:
            print(f"❌ Error loading {file.file_name}: {e}")
        return []

    def ingest_materials(self, course_id: str, materials: List[MaterialFile]) -> SQLiteVec:
        conn = self._get_connection()
        self._init_tracking_table(conn)
        table_name = f"course_{course_id.lower()}_vec"
        
        all_docs = []
        new_materials = []

        # FIX: Check against tracking table
        for material in materials:
            cursor = conn.execute("SELECT 1 FROM Ingested_Files WHERE file_path = ?", (material.file_path,))
            if cursor.fetchone():
                print(f"⏩ Skipping {material.file_name} (Already Ingested)")
            else:
                new_materials.append(material)

        if not new_materials:
            print("✅ All provided materials are already up-to-date in the Vector DB.")
            return SQLiteVec(table=table_name, connection=conn, embedding=self.embeddings)

        for material in new_materials:
            print(f"Loading {material.file_name}...")
            docs = self._load_file(material)
            for doc in docs:
                doc.metadata["course_id"] = course_id
            all_docs.extend(docs)

        if all_docs:
            chunks = self.text_splitter.split_documents(all_docs)
            print(f"Ingesting {len(chunks)} new chunks into SQLite Vector DB...")
            vectorstore = SQLiteVec.from_documents(documents=chunks, embedding=self.embeddings, table=table_name, connection=conn)
            
            # Log successful ingestions
            for mat in new_materials:
                conn.execute("INSERT INTO Ingested_Files (file_path, course_id) VALUES (?, ?)", (mat.file_path, course_id))
            conn.commit()
            return vectorstore

        return SQLiteVec(table=table_name, connection=conn, embedding=self.embeddings)

    def get_retriever(self, course_id: str, k: int = 4):
        table_name = f"course_{course_id.lower()}_vec"
        conn = self._get_connection()
        vectorstore = SQLiteVec(table=table_name, connection=conn, embedding=self.embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": k})