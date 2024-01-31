#----- IMPORTS -----#

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, Qdrant
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
import pandas as pd
from bert_score import BERTScorer
import csv
import re
from nltk.corpus import stopwords
import sys
sys.path.append("../app/")
from global_methods import *

#----- MAIN SCRIPT -----#

# Initialize BERTScorer
scorer = BERTScorer(model_type='bert-base-uncased')

# Path to the folder containing .xlsx files of each test
folder_path = "/home/cbreant/Documents/5BIM/projet5BIM/llm-project/app/All_files_by_test"  

# List to store results
score_results = []

# API's
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# LLM load
llm = HuggingFaceHub(repo_id = "mistralai/Mistral-7B-v0.1")

# Prompt creation
template = """Use only the text provided in the following context : {context}
Please answer in french to this question : {question}
If you didn't find the answer in the context, just say that you don't know, don't use your own knowledge and learning. 

Answer :"""
prompt = PromptTemplate.from_template(template)

# Parameters load
with open("parameters.csv", "r") as parameters_file :
    all_parameters = list(csv.reader(parameters_file))

# Iterate through all tests
with open(file = "documents_expected_list.csv", mode = "r", newline = '\n') as global_file :
    summary_test = list(csv.reader(global_file))[1:]

# Iterate throught all tests
i = 0
for row in summary_test :

    document_name = row[2]
    print(document_name)
    
    try :
        with open("preprocessed_source_documents/"+document_name, encoding = "utf-8") as source_file :
            source_text = source_file.read()
    except :
        with open("source_documents/"+document_name, encoding = "utf-8") as source_file :
            source_text = source_file.read()
        process_text_file(source_text, "preprocessed_source_documents/"+document_name)
        with open("preprocessed_source_documents/"+document_name, encoding = "utf-8") as source_file :
            source_text = source_file.read()

    temp_list_results = []

    # Iterate on all parameters
    for params in all_parameters :

        # Splitters
        if params[0] == 'recursive' :
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = int(params[2]), chunk_overlap = int(params[3]))
        elif params[0] == 'token' :
            text_splitter = TokenTextSplitter(chunk_size = int(params[2]), chunk_overlap = int(params[3]))

        texts = text_splitter.create_documents([source_text])
        splitted_texts = text_splitter.split_documents(texts)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-mpnet-base-v2')

        # Vectorstores
        if params[1] == 'chroma' :
            vectorstore = Chroma.from_documents(splitted_texts, embeddings)
        elif params[1] == 'qdrant' :
            vectorstore = Qdrant.from_documents(splitted_texts, embeddings, location = ":memory:", 
                                                prefer_grpc = True, collection_name = "my_documents")

        # Retriever
        retriever = vectorstore.as_retriever()
        runner = RetrievalQA.from_chain_type(llm = llm, retriever = retriever, 
                                            verbose = True, return_source_documents = True,
                                            chain_type_kwargs = {"prompt": prompt})

        # Result
        query = row[4]
        candidate = runner({"query": query})['result']
        candidate = candidate.replace('\n', "")
        params.append(candidate)

        # Extract reference from the first row of the "Expected" column
        reference = row[5]

        # Calculate BERT score
        P, R, F1 = scorer.score([candidate], [reference])
        for p in [P.item(), R.item(), F1.item()] :
            params.append(p)

        # Append the results
        temp_list_results.append(params)
        
        i += 1

    # Save the results
    df_results = pd.DataFrame(temp_list_results, columns = ['splitter', 'vectorstore', 'chunk_size', 'chunk_overlap', 'result', 'precision', 'recall', 'f1_score'])
    df_results.to_csv("results/test_"+str(row[0])+".csv")

    break