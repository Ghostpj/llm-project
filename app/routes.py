from flask import render_template, request, redirect, url_for, flash
from app import app
from werkzeug.utils import secure_filename
import os
from .settings import *
from .global_methods import *
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

messages = [{'a' : "Bonjour, comment puis-je vous aider ?"}]

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods = ['GET', 'POST'])
def chat():

    global messages

    if request.method == 'POST':

        # Récupère le message de l'utilisateur depuis le formulaire
        question_content = request.form.get('message')
        file = request.files.get('fichier')
        if question_content and not file:

            llm = HuggingFaceHub(repo_id = "mistralai/Mistral-7B-Instruct-v0.1")

            messages.append({'q' : question_content, 'a' : llm(question_content)})  # Ajoute le message à la liste

        # Gestion du téléchargement des fichiers
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Assurez-vous que le nom de fichier est sûr
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Sauvegarde le fichier dans le dossier de destination

            # Identifier et ajouter le type de fichier au message
            file_type_description = determine_file_type(filename)
            messages.append({'a' : f"Le fichier '{filename}' a été correctement téléchargé, c'est {file_type_description}."})

            # Text load
            loader = TextLoader(file_path)
            pages = loader.load()

            if not question_content :
                # LLM : french and english
                try :
                    language = detect(pages[0].page_content)
                    if language == 'fr' :
                        print('fr')
                        llm = HuggingFaceHub(repo_id = "plguillou/t5-base-fr-sum-cnndm")
                    elif language == 'en' :
                        print('en')
                        llm = HuggingFaceHub(repo_id = "Falconsai/medical_summarization")
                    else :
                        raise Exception
                except :
                    raise Exception

                # Prompt
                prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
                Return your response in bullet points which covers the key points of the text.
                ```{text}```
                BULLET POINT SUMMARY:
                """
                prompt = PromptTemplate.from_template(prompt_template)

                # Stuff
                stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                
                messages.append({'a' : stuff_chain.run(pages)})

            else :
                messages.append({'q' : question_content})

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
                docs = text_splitter.split_documents(pages)
                model_name = "sentence-transformers/all-mpnet-base-v2"
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                llm = HuggingFaceHub(repo_id = "mistralai/Mistral-7B-Instruct-v0.1")

                vectordb = Qdrant.from_documents(documents=docs, embedding=embeddings, location=":memory:", 
                                                prefer_grpc=True, collection_name="my_documents")
                retriever = vectordb.as_retriever()
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)

                messages.append({'a' : qa.run(question_content)})

    return render_template('chat.html', messages = messages)
