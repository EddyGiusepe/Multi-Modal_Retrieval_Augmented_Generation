#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro
"""
import streamlit as st

# Configurar a página do Streamlit:
st.set_page_config(
    page_title="Multi-Modal RAG",
    page_icon="🚀",
    initial_sidebar_state="expanded",  # "expanded" significa que a barra lateral estará aberta quando a aplicação carregar. A alternativa seria "collapsed" para iniciar fechada
    layout="centered",  # "wide" significa que a aplicação ocupará toda a largura da tela
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",  # Neste exemplo, todos os links apontam para um domínio fictício 'extremelycoolapp.com'
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

# Título da aplicação web:
st.title("Multi-Modal Retrieval-Augmented Generation (RAG)")
st.subheader("🤖 Análise de Vídeos, Imagens e Documentos com IA 🤖")

# Explicação para usuários não técnicos:
st.write(
    """
         **Multi-Modal RAG** significa **Multi-Modal Retrieval-Augmented Generation**.
         É um processo que permite fazer perguntas sobre diferentes tipos de mídia, como vídeos, imagens ou documentos, e obter respostas com IA.
    
         Nesta aplicação, você pode interagir com as seguintes funcionalidades:
    
         1. **Video Question Answering**: Upload um vídeo, e a aplicação irá transcrever o vídeo. Você pode então fazer perguntas sobre o conteúdo do vídeo.
         2. **Image Question Answering**: Upload uma imagem, e a aplicação irá descrever o conteúdo da imagem. Você pode fazer perguntas sobre o conteúdo da imagem.
         3. **Document Question Answering**: Upload um documento (PDF, Word, etc.), e a aplicação irá extrair as informações relevantes para responder suas perguntas.
    
         Aqui está como funciona:
    
         - **Video QA**: A aplicação primeiro transcreve o vídeo para texto. Então, você pode fazer qualquer pergunta sobre o vídeo.
                    Com base na transcrição, ela recupera as informações relevantes para ajudar a responder sua pergunta.
    
         - **Image QA**: Upload uma imagem, e a aplicação irá analisar a imagem, descrevendo seu conteúdo. Você pode então fazer perguntas sobre o conteúdo da imagem.
    
         - **Document QA**: Upload um documento (como um PDF ou Word), e a aplicação irá extrair as informações relevantes para ajudar a responder suas perguntas.
    
         Cada funcionalidade usa uma combinação de modelos de IA e algoritmos sofisticados para fornecer as melhores respostas possíveis.
         
         
         Thank God 🤓!
         """
)
