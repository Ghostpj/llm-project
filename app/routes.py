#----- IMPORTS -----#

from flask import render_template, request
from app import app
from werkzeug.utils import secure_filename
import os
from .settings import *
from .global_methods import *
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask import session

#----- INTERACTION WITH THE INTERFACE -----#

# History's dictionnary of messages in the interface with 2 types of keys :
# 'a' for answers from the llm
# 'q' for questions from the user
if LANGUAGE_CHOICE == "fr" :
    messages = [{'a' : "Bonjour, comment puis-je vous aider ?"}]
elif LANGUAGE_CHOICE == "en" :
    messages = [{'a' : "Hi, how can I help ?"}]
else :
    raise Exception

# Uploaded documents folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.secret_key = 'session_key'

@app.route('/', methods = ['GET', 'POST'])
def chat():
    
    if 'filename' not in session.keys() :
        session['filename'] = None
        session['chunksize'] = None
        session['chunkoverlap'] = None

    # Launch of the user's request
    if request.method == 'POST':

        # API solution
        if BOOL_API :
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" #HuggingFace key
            llm = HuggingFaceHub(repo_id = LLM_NAME)

        # Local solution : loads the LLM locally if not already loaded in the "locl_llm" folder
        else :
            if BOOL_LLM :
                model = AutoModelForCausalLM.from_pretrained(LLM_FOLDER)
                tokenizer = AutoTokenizer.from_pretrained(LLM_FOLDER)
            else :
                model = AutoModelForCausalLM.from_pretrained(LLM_NAME)
                tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
                model.save_pretrained(LLM_FOLDER)
                tokenizer.save_pretrained(LLM_FOLDER)

            # Creating a runnable LLM
            pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, max_new_tokens = 500)
            llm = HuggingFacePipeline(pipeline = pipe)

        # Getting the user's message from the interface
        question_content = request.form.get('message')

        # Getting the file uploaded form the interface, saving it and choosing the right parameters
        file_article = request.files.get('file_article')
        file_lesson = request.files.get('file_lesson')
        if file_article or file_lesson :

            if file_article and allowed_file(file_article.filename) :
                filename = secure_filename(file_article.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER']+"/unprocessed", filename)
                file_article.save(file_path)
                chunk_size = 500
                chunk_overlap = 10

            elif file_lesson and allowed_file(file_lesson.filename) :
                filename = secure_filename(file_lesson.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER']+"/unprocessed", filename)
                file_article.save(file_path)
                chunk_size = 500
                chunk_overlap = 10

            # Tell's the user that the file has been correctly uploaded
            file_type_description = determine_file_type(filename)
            if LANGUAGE_CHOICE == "fr" :
                messages.append({'a' : f"Le fichier '{filename}' a été correctement téléchargé, c'est {file_type_description}."})
            elif LANGUAGE_CHOICE == "en" :
                messages.append({'a' : f"The file '{filename}' has been correctly uploaded, it's {file_type_description}."})
            else :
                raise Exception
            
            # Loads the text with the according loader (text or PDF)
            if filename.rsplit('.', 1)[1].lower() == "txt" :
                loader = TextLoader(file_path)
            elif filename.rsplit('.', 1)[1].lower() == "pdf" :
                loader = PyPDFLoader(file_path)
            pages = loader.load()

            # Cleaning of the text
            process_text_file(pages[0].page_content, app.config['UPLOAD_FOLDER']+"/processed/"+filename)

            # Saving the text and parameters
            session['filename'] = filename
            session['chunksize'] = chunk_size
            session['chunkoverlap'] = chunk_overlap
            
        else :
            
            # Retrieves the previous text and parameters
            filename = session['filename']
            chunk_size = session['chunksize']
            chunk_overlap = session['chunkoverlap']

        # Getting the summary's buttons clicks
        summary_type = request.form.get('summary_button')

        # Getting the question's button click
        question_click = request.form.get('send_button')

        # Case of a simple question to the LLM without any file
        if question_content and not filename:

            # Adds the question and the computed answer to the chat
            messages.append({'q' : question_content, 'a' : llm(question_content)})

        # Case of an uploaded file
        if filename :

            # Loads the text
            with open(app.config['UPLOAD_FOLDER']+"/processed/"+filename) as source_file :
                source_text = source_file.read()

            # Preprocessing of the text : splitting, embeddings & vectorstores
            # Splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
            texts = text_splitter.create_documents([source_text])
            splitted_texts = text_splitter.split_documents(texts)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(model_name = EMBEDDINGS_NAME)

            # Vectorstores
            vectorstore = Qdrant.from_documents(splitted_texts, embeddings, location = ":memory:")

            # Creates a prompt with the query and the context
            prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            {context}
            Question: {question}
            Helpful Answer:"""
            prompt = PromptTemplate.from_template(prompt_template)

            # Summarize part
            if summary_type :

                # Retrieves the type of summary
                if summary_type == "Short" :
                    max_words = "100"
                elif summary_type == "Medium" :
                    max_words = "250"
                else :
                    max_words = "500"

                # Creates the right summary query
                query = """Write a """+summary_type.lower()+""" summary of the text with less than """+max_words+""" words. \n
                Return your response in """+LANGUAGE_DICT[LANGUAGE_CHOICE]+""" in bullet points which covers the key points of the text.
                """
            
                # Retriever in the vectorstore
                retriever = vectorstore.as_retriever()
                runner = RetrievalQA.from_chain_type(llm = llm, retriever = retriever, 
                                            verbose = True, return_source_documents = True,
                                            chain_type_kwargs = {"prompt": prompt})
                
                # Run the query and adds it into the answers of the interface
                messages.append({'q' : query, 'a' : runner({"query": query})['result']})
            
            # Question-Answering part
            elif question_click :
            
                # Retriever in the vectorstore
                retriever = vectorstore.as_retriever()
                runner = RetrievalQA.from_chain_type(llm = llm, retriever = retriever, 
                                            verbose = True, return_source_documents = True,
                                            chain_type_kwargs = {"prompt": prompt})
                
                # Run the query and adds it into the answers of the interface
                messages.append({'q' : question_content, 'a' : runner({"query": question_content})['result']})
            
            else :
                raise Exception
        
        else :
            
            # Tells to upload a file if no file was uploaded
            if LANGUAGE_CHOICE == "fr" :
                messages.append({'a' : "Veuillez télécharger un fichier en premier lieu."})
            else :
                messages.append({'a' : "Please upload a file first."})


    return render_template('chat.html', messages = messages)
