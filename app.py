from PyPDF2 import PdfReader
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import base64
import streamlit as st

display_file = r"bell_labs.pdf"

with open(display_file, "rb") as f:
  base64_pdf = base64.b64encode(f.read()).decode('utf-8')

pdf_display = f"""<embed
class="pdfobject"
type="application/pdf"
title="Embedded PDF"
src="data:application/pdf;base64,{base64_pdf}"
style="overflow: auto; width: 50%; height: 50%;">"""

st.markdown(pdf_display, unsafe_allow_html = True)

os.environ['OPENAI_API_KEY'] = "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"

# extract data from the PDF
reader = PdfReader(r'bell_labs.pdf')

# read data from the PDF and store in a variable
raw_text = ""

for i, page in enumerate(reader.pages):
  text = page.extract_text()

  if text:
    raw_text += text

# print(raw_text)

# Split the text into sentences / smaller chunks
splitter = CharacterTextSplitter(separator = "\n", chunk_size = 1250, chunk_overlap = 200, length_function = len)
sentences = splitter.split_text(raw_text)

# print(sentences[1])

# Text Embeddings
embeddings = OpenAIEmbeddings()

doc_search = FAISS.from_texts(sentences, embeddings)
chain = load_qa_chain(OpenAI(), chain_type = 'stuff')

while True:
    
  query = st.text_input("\nQUESTION : ", key = 'prompt_1')
  if query != "exit":
    doc = doc_search.similarity_search(query)
    st.write("\nANSWER : {}\n".format(chain.run(input_documents = doc, question = query)))

  else:
    break

