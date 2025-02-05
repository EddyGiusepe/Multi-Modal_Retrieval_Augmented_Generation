#! /usr/bin/env python3
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq
from uuid import uuid4
import faiss
import os
from dotenv import load_dotenv
import logging
import httpx
import base64
import asyncio

# Inicialização de variáveis de ambiente e logging:
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Função assíncrona para invocar a cadeia:
async def async_invoke_chain(chain, input_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chain.invoke, input_data)

# Inicialização do estado da sessão para mensagens e modelos:
if "messages" not in st.session_state:
    st.session_state.messages = []
if "models" not in st.session_state:
    st.session_state["models"] = {
        "Gemini": ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.8,
            verbose=True,
            api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        ),
        "Deepseek-R1-distill-llama-70b": ChatGroq(
            model="deepseek-r1-distill-llama-70b",
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GROQ_API_KEY"),
        ),
        "Mistral": ChatMistralAI(
            model_name="open-mistral-nemo",
            temperature=0.8,
            verbose=True
        ),
        "Llama": ChatCerebras(
            model="llama-3.3-70b",
            temperature=0.8,
            verbose=True,
            api_key=os.getenv("CEREBRAS_API_KEY")
        )
    }

# Inicialização do modelo de embeddings:
if "embeddings" not in st.session_state:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

st.header("📸📈📊🖼️ Image Content Analysis and Question Answering")

# Breve visão geral para análise de conteúdo de imagem:
description = """
Upload uma imagem, e a AI analisará seu conteúdo e responderá suas perguntas. 
Ele pode interpretar vários tipos de imagens, incluindo:
- Imagens gerais (objetos, pessoas, cenas)
- Diagramas, gráficos e visualizações de dados
- Imagens científicas e médicas
- Imagens baseadas em texto (documentos, capturas de tela)
"""

# Exibir uma breve descrição:
st.write(description)

# Upload da imagem e entrada de URL:
st.header("Upload da imagem para Perguntas e Respostas")
uploaded_file = st.file_uploader("Upload de uma imagem (.jpeg, .jpg, .png, etc.):", type=["jpeg", "jpg", "png"])

st.header("Ou insira a URL de uma imagem:")
image_url = st.text_input("Insira a URL da imagem")

image_data = None

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    image_data = base64.b64encode(uploaded_file.read()).decode("utf-8")
elif image_url:
    try:
        with httpx.Client() as client:
            response = client.get(image_url)
            response.raise_for_status()
            st.image(response.content, caption="Image from URL", use_container_width=True)
            image_data = base64.b64encode(response.content).decode("utf-8")
    except Exception as e:
        st.error(f"Error fetching image from URL: {e}")

if image_data:
    message = HumanMessage(content=[{
            "type": "text", "text": """Descreva o que há na imagem em detalhes.  
                                       Sempre responda em português do Brasil (pt-BR).
                                    """
        }, {
            "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        }])

    # Gerar resposta do modelo:
    response = asyncio.run(async_invoke_chain(st.session_state.models["Gemini"], [message]))
    knowledge = [Document(page_content=response.content)]

    # Dividir o texto em chunks para indexação:
    text_splitter = RecursiveCharacterTextSplitter(separators="\n\n", chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_documents(knowledge)

    # Criar o índice HNSWFlat para indexação de embeddings de imagem:
    index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("Olá Mundo!")))

    # Criar o armazenamento de vetores FAISS para recuperação de documentos:
    vector_store = FAISS(
        embedding_function=st.session_state.embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Gerar IDs únicos e adicionar documentos ao armazenamento:
    ids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=ids)

    # Atualizar o mapeamento entre o índice FAISS e IDs de documentos:
    for idx, doc_id in enumerate(ids):
        vector_store.index_to_docstore_id[idx] = doc_id

    # Criar o retriever de imagem com o índice FAISS:
    image_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def get_retrieved_context(query):
        retrieved_documents = image_retriever.get_relevant_documents(query)
        return "\n".join(doc.page_content for doc in retrieved_documents)

    # Input do usuário para perguntas sobre a imagem:
    user_input = st.chat_input("Pergunte sobre a imagem:")

    prompt = ChatPromptTemplate.from_messages([(
            "system", """Você é um analista de imagens experiente treinado para detectar e explicar as diferenças entre
                         imagens reais e geradas por IA. Sua análise deve ser detalhada, factual, destacando padrões, texturas e
                         anomalias únicas para cada categoria. Esta é a informação sobre a imagem {context}, use-a para
                         responder as perguntaa."""
        ), ("human", "{question}")])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        qa_chain = prompt | st.session_state.models["Deepseek-R1-distill-llama-70b"] | StrOutputParser()
        context = get_retrieved_context(user_input)
        response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))
        st.session_state.messages.append({"role": "assistant", "content": response_message})
        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])
            