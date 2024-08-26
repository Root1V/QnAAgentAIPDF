import os
from dotenv import load_dotenv
from config import Config

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import streamlit as st

# Cargar las variables de entorno
load_dotenv()

# Obtener todos los valores de configuración
config = Config.get_all()

# BD persist
directory_db = "./chroma_db"


def update_embeddings():
    path_pdf = "data/AI.pdf"
    loader = PyPDFLoader(path_pdf)
    docs = loader.load()

    # dividimos el texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(docs)

    # Creamos la BD de Vectores
    vectordb = Chroma.from_documents(
        chunked_documents,
        OpenAIEmbeddings(model=config["model_embeddings"]),
        persist_directory=directory_db,
    )

    return vectordb


st.title("QnA with your PDF")
st.write("Este es un agente de AI que responde a preguntas sobre tu archivo PDF")

if st.button("Actualizar memory"):
    vectordb = update_embeddings()
    st.write("Memoria/Embeddings actualizados correctamente ....")


prompt_system = "Eres un agente de IA especializado en respuestas amables a los usuarios. Responde a las preguntas de los usuarios {input} con mucha precision basandote en el contexto {context}, no hagas suposiciones ni proporciones informacion que no esta incluida en el {context}."

llm = ChatOpenAI(model=config["model"], max_tokens=config["max_tokens"])
qna_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_system))

question = st.text_area("Ingresa tu consulta a la IA sobre el documento")

if st.button("Enviar"):
    if question:
        vectordb = Chroma(
            persist_directory=directory_db,
            embedding_function=OpenAIEmbeddings(model=config["model_embeddings"]),
        )
        result_simil = vectordb.similarity_search(question, k=5)
        context = ""

        for doc in result_simil:
            context += doc.page_content

        answer = qna_chain.invoke({"input": question, "context": context})
        result = answer["text"]
        st.write(result)
    else:
        st.write("Por favor, ingresa un pregunta antes de enviar.")
