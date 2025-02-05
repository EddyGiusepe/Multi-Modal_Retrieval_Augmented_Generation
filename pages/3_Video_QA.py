#! /usr/bin/env python3
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from uuid import uuid4
import whisper
import torch
import tempfile
import faiss
from dotenv import load_dotenv
import logging
import asyncio
import os

# Carregar variáveis de ambiente:
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar estado da sessão para mensagens:
if "messages" not in st.session_state:
    st.session_state.messages = []

async def async_invoke_chain(chain, input_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chain.invoke, input_data)

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


# Inicializar embeddings:
if "embeddings" not in st.session_state:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": False}
    )

# Text splitter recursivo:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

# Prompt template:
prompt = ChatPromptTemplate.from_messages([
    ("system", """
     Você é um especialista em explicar conteúdo de vídeo. Seu objetivo é fornecer respostas completas
     e insights sobre perguntas do usuário com base no transcríto do vídeo fornecido. Você combinará 
     informações do transcríto com seu conhecimento geral para dar uma compreensão bem-rounded.

Aqui está como você deve abordar cada pergunta:

1. **Visão geral do vídeo:** Primeiro, responda diretamente à pergunta do usuário usando trechos relevantes do transcríto fornecido. Use aspas para claramente indicar texto tirado diretamente do transcrito.

2. **Explicação detalhada:** Expanda as informações do transcrito com explicações detalhadas, contexto e informações de fundo do seu conhecimento geral. Explique quaisquer termos técnicos ou conceitos que possam ser desconhecidos para o usuário.

3. **Exemplos e Analogias:** Use exemplos, analogias e cenários do mundo real para ilustrar ideias complexas e torná-las mais fáceis de entender.

4. **Snippets de código/URLs (Se aplicável):** Se o vídeo discute código ou se refere a recursos externos, forneça trechos de código relevantes (formatados para legibilidade) ou URLs para melhorar a explicação.

5. **Estrutura e clareza:** Apresente suas respostas em um formato claro, estruturado e fácil de ler. Use cabeçalhos, listas com marcadores e listas numeradas onde apropriado.

Contexto (Transcrição do vídeo): {context}
     """),
    ("user", "{question}")
])

st.title("Video QA com LangChain 🦜🔗 e Streamlit")

# Upload o arquivo de vídeo:
uploaded_video = st.file_uploader("Upload um arquivo de vídeo", type=["mp4", "mov", "avi"])

video_url = None

if uploaded_video:
    st.video(uploaded_video)
    if st.button("Gera a Transcrição do Vídeo"):
        with st.spinner("Transcrevendo o vídeo ..."):
            try:
                # Salva o arquivo de vídeo enviado em um arquivo temporário:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(uploaded_video.getvalue())
                    temp_file_path = temp_file.name

                # Carrega o modelo Whisper e transcreve o vídeo:
                model = whisper.load_model("small")
                model = model.to("cpu")
                result = model.transcribe(temp_file_path)

                # Obtém o texto da transcrição:
                transcript = result["text"]
                docs = [Document(page_content=transcript)]
                chunks = text_splitter.split_documents(docs)

                # **Limpa o vector store anterior**
                index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("Olá mundo")))
                st.session_state.vector_store = FAISS(
                    embedding_function=st.session_state.embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )

                # Adiciona os novos documentos ao vector store:
                ids = [str(uuid4()) for _ in range(len(chunks))]
                st.session_state.vector_store.add_documents(documents=chunks, ids=ids)
                st.success("Você está pronto para fazer perguntas sobre o vídeo")

            except Exception as e:
                st.error(f"Erro ao obter a transcrição: {e}")

else:
    # URL do vídeo do YouTube:
    video_url = st.text_input("Insira a URL do vídeo do YouTube:")

    if video_url and st.button("Gera a Transcrição do Vídeo"):
        st.video(video_url)
        
        with st.spinner("Obtendo a transcrição ..."):
            try:
                # Carrega a transcrição usando YoutubeLoader:
                loader = YoutubeLoader.from_youtube_url(
                    video_url,
                    add_video_info=False,
                    language=["pt"]
                )
                transcript = loader.load()

                # Divide em documentos para chunking:
                docs = [Document(page_content=entry.page_content) for entry in transcript]
                chunks = text_splitter.split_documents(docs)

                # **Limpa o vector store anterior**
                index = faiss.IndexFlatL2(len(st.session_state.embeddings.embed_query("Olá mundo")))
                st.session_state.vector_store = FAISS(
                    embedding_function=st.session_state.embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )

                # Adiciona os novos documentos ao vector store:
                ids = [str(uuid4()) for _ in range(len(chunks))]
                st.session_state.vector_store.add_documents(documents=chunks, ids=ids)
                st.success("Você está pronto para fazer perguntas sobre o vídeo")

            except Exception as e:
                st.error(f"Erro ao obter a transcrição: {e}")

# Seção de perguntas e respostas:
if "vector_store" in st.session_state:
    def get_retrieved_context(query):
        video_retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )
        
        retrieved_documents = video_retriever.get_relevant_documents(query)
        return "\n".join(doc.page_content for doc in retrieved_documents)

    user_input = st.chat_input("Faça suas perguntas sobre o vídeo:")
    if user_input:
        context = get_retrieved_context(user_input)
        qa_chain = prompt | st.session_state.models[ "Deepseek-R1-distill-llama-70b"] | StrOutputParser()
        response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response_message})

        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])
else:
    st.error("Nenhuma transcrição disponível. Por favor, envie ou processe um vídeo primeiro.")
