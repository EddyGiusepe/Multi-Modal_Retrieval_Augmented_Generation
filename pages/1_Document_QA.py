#! /usr/bin/env python3
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cerebras import ChatCerebras
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader
from streamlit_pdf_viewer import pdf_viewer
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import StrOutputParser
from uuid import uuid4
import faiss
import os
from dotenv import load_dotenv
import logging
import asyncio

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if 'pdf_ref' not in st.session_state:
    st.session_state.pdf_ref = None

# Função async para invocar a cadeia de execução:
async def async_invoke_chain(chain, input_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, chain.invoke, input_data)

# Inicializa o estado da sessão para mensagens e modelos:
if "messages" not in st.session_state:
    st.session_state["messages"] = []


llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b ",
    temperature=0.8,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)


if "models" not in st.session_state:
    st.session_state["models"] = {
        "Gemini": ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
                                         temperature=0.8,
                                         verbose=True,
                                         api_key=os.getenv("GOOGLE_AI_STUDIO_API_KEY")
                                        ),

        "Deepseek-R1-distill-llama-70b": ChatGroq(model="deepseek-r1-distill-llama-70b",
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

if "embeddings" not in st.session_state:
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    st.session_state["embeddings"] = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

# Header (cabeçalho) da interface Streamlit:
st.header("🗂️ ✨ Document Question Answering")
st.write("""Upload um documento e faça perguntas sobre seu conteúdo. Formatos suportados incluem:
- Arquivos PDF (.pdf)
- Arquivos Word (.docx)
""")

# Carregador de arquivos para documento:
uploaded_doc = st.file_uploader("Upload seu documento (.pdf, .docx):", type=["pdf", "docx"])

# Processar o arquivo PDF carregado:
if uploaded_doc and uploaded_doc.name.endswith(".pdf"):
    # Armazenar o arquivo PDF carregado no estado da sessão para visualização no sidebar:
    st.session_state.pdf_ref = uploaded_doc

    # Exibir o preview do PDF no sidebar:
    with st.sidebar:
        binary_data = st.session_state.pdf_ref.getvalue()
        pdf_viewer(input=binary_data, width=700)

    # Processar o arquivo PDF:
    with st.spinner("Processando o arquivo PDF carregado ..."):
        # Salvar o arquivo carregado temporariamente:
        temp_path = f"temp_{uuid4().hex}.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_doc.read())

        # Carregar o documento usando PyMuPDFLoader:
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()

        # Remover o arquivo temporário:
        os.remove(temp_path)

        st.success(f"Sucesso ao carregar {len(documents)} páginas do arquivo PDF carregado.")

        # Incluir (Embed) os documentos no índice FAISS:
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=1200, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        embedding_dim = len(st.session_state["embeddings"].embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        vector_store = FAISS(
            embedding_function=st.session_state["embeddings"],
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        ids = [str(uuid4()) for _ in range(len(chunks))]
        vector_store.add_documents(chunks, ids=ids)

        for idx, doc_id in enumerate(ids):
            vector_store.index_to_docstore_id[idx] = doc_id

        # Criar o retriever com o índice FAISS:
        doc_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        def get_retrieved_context(query):
            retrieved_documents = doc_retriever.get_relevant_documents(query)
            return "\n".join(doc.page_content for doc in retrieved_documents)

        # Entrada do usuário para fazer perguntas sobre o documento:
        user_input = st.chat_input("Faça suas perguntas sobre o documento/documentos:")

        # Define o template de prompt:
        prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        Você é um especialista em análise de documentos com a capacidade de processar grandes
        volumes de texto de forma eficiente. Sua tarefa é extrair insights chave e responder
        perguntas com base no conteúdo do documento fornecido: {context}
        Quando perguntado, você deve fornecer uma resposta factual, direta, detalhada e concisa,
        apenas usando as informações disponíveis do documento. Se a resposta não pode ser encontrada
        diretamente, você deve esclarecer isso e fornecer contexto ou informações relacionadas se 
        aplicável. Focalize em desvendar informações críticas, sejam elas fatos específicos, resumos 
        ou insights ocultos dentro do documento.
        Ademais, você deve fornecer uma resposta sempre em português do Brasil (pt-BR).
    """),
    ("human", "{question}")
])


        # Lidar com o input do usuário e exibir respostas:
        if user_input:
            available_models = list(st.session_state["models"].keys())
            print("Os modelos disponíveis são:", available_models)
            print("")
            st.session_state["messages"].append({"role": "user", "content": user_input})
            qa_chain = prompt_template | st.session_state["models"]["Deepseek-R1-distill-llama-70b"] | StrOutputParser()
            context = get_retrieved_context(user_input)
            response_message = asyncio.run(async_invoke_chain(qa_chain, {"question": user_input, "context": context}))
            st.session_state["messages"].append({"role": "assistant", "content": response_message})

            for message in st.session_state["messages"]:
                st.chat_message(message["role"]).markdown(message["content"])
                