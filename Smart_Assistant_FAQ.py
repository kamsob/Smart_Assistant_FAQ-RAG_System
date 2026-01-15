import os
import sys
import time
from typing import Any, List, Dict
from dotenv import load_dotenv

# LangChain i narzędzia RAG
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Ładowanie zmiennych środowiskowych
load_dotenv()

def load_university_docs(folder_path: str) -> List[Document]:
    """Ładuje wszystkie PDFy z folderu i dzieli je na fragmenty."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    if not os.path.exists(folder_path):
        print(f"BŁĄD: Folder {folder_path} nie istnieje!")
        return []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs = loader.load()
            # Dodajemy nazwę pliku do metadanych każdego fragmentu
            for doc in docs:
                doc.metadata["source_file"] = file
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
    
    print(f"Załadowano {len(all_chunks)} fragmentów z dokumentów.")
    return all_chunks

def create_knowledge_base(chunks: List[Document]) -> VectorStore:
    """Tworzy bazę wektorową w pamięci."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_store

def ask_bot(query: str, vector_store: VectorStore):
    """Przeszukuje bazę i generuje odpowiedź przez Groq."""
    
    # 1. Pobieranie 5 najbardziej pasujących fragmentów
    relevant_docs = vector_store.similarity_search(query, k=5)
    context = "\n\n".join([f"Źródło ({d.metadata['source_file']}): {d.page_content}" for d in relevant_docs])

    # 2. Konfiguracja modelu Llama 3 przez Groq
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_key=os.getenv("GROK_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1"
    )

    system_prompt = (
        "Jesteś pomocnym asystentem FAQ dla studentów uczelni. "
        "Twoim zadaniem jest odpowiadać na pytania wyłącznie na podstawie dostarczonego kontekstu. "
        "Jeśli w kontekście nie ma odpowiedzi, powiedz uprzejmie, że nie posiadasz takich informacji. "
        "Zawsze podawaj nazwę pliku źródłowego, z którego pochodzi informacja.\n\n"
        f"KONTEKST:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    chain = prompt | llm
    
    print("\n[Asystent AI]: ", end="", flush=True)
    response = chain.invoke({"question": query})
    print(response.content)

def main():
    print("=== UCZELNIANY ASYSTENT FAQ (RAG) ===")
    
    # Krok 1: Budowa bazy wiedzy
    docs_folder = "./docs"
    chunks = load_university_docs(docs_folder)
    
    if not chunks:
        print("Brak dokumentów do analizy. Wrzuć pliki PDF do folderu './docs'.")
        return

    print("Inicjalizacja bazy wiedzy...")
    vector_store = create_knowledge_base(chunks)
    
    print("\nSystem gotowy! O co chcesz zapytać? (wpisz 'exit' aby wyjść)")
    
    # Krok 2: Interaktywna pętla czatu
    while True:
        user_query = input("\n[Ty]: ")
        if user_query.lower() in ['exit', 'quit', 'wyjdź']:
            print("Dziękuję za rozmowę. Do widzenia!")
            break
        
        if not user_query.strip():
            continue

        try:
            ask_bot(user_query, vector_store)
        except Exception as e:
            print(f"\nBŁĄD: {e}")

if __name__ == "__main__":
    main()