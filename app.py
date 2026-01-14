import os
from dotenv import load_dotenv
import chainlit as cl

# Imports LangChain & Co
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Configuration
CHROMA_PATH = "chroma_db_temp"  # On utilisera un dossier temporaire


# --- Fonction utilitaire pour traiter le PDF uploadé ---
def process_pdf_file(pdf_path):
    """Lit le PDF, le découpe et crée la DB vectorielle à la volée"""

    # 1. Load
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 3. Embed (Local)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Store (On recrée une DB propre pour ce fichier)
    # Note: En prod, on gérerait des ID de session pour ne pas mélanger les utilisateurs
    db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    return db


@cl.on_chat_start
async def start():
    # 1. Demander le fichier à l'utilisateur
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="📂 Veuillez déposer votre relevé bancaire ou contrat (PDF) pour commencer l'analyse.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    # 2. Feedback visuel
    msg = cl.Message(content=f"Traitement de `{file.name}` en cours... ⏳")
    await msg.send()

    # 3. Sauvegarder le fichier temporairement sur le disque pour que PyPDFLoader puisse le lire
    # Chainlit nous donne le fichier en mémoire, on l'écrit sur le disque
    temp_pdf_path = f"temp_{file.name}"
    with open(temp_pdf_path, "wb") as f:
        with open(file.path, "rb") as source_file:
            f.write(source_file.read())

    # 4. Création du cerveau (RAG)
    # On appelle notre fonction synchrone dans un thread pour ne pas bloquer l'interface
    db = await cl.make_async(process_pdf_file)(temp_pdf_path)

    # 5. Setup de la chaîne de discussion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    template = """
    Tu es un auditeur bancaire senior.
    Utilise le contexte suivant pour répondre à la question.
    Si tu ne sais pas, dis-le. Sois professionnel et concis.
    
    Contexte : {context}
    Question : {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Sauvegarde en session
    cl.user_session.set("rag_chain", rag_chain)

    # 7. Nettoyage et Prêt
    msg.content = f"✅ Analyse de `{file.name}` terminée ! Je vous écoute."
    await msg.update()

    # Optionnel : Supprimer le fichier temp
    # os.remove(temp_pdf_path)


@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")

    # Appel de la chaîne
    res = await rag_chain.ainvoke(message.content)  # type: ignore

    await cl.Message(content=res).send()
