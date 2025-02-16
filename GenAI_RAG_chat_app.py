from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable.config import RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain_community.vectorstores.chroma import Chroma
from langchain.vectorstores import Chroma
from operator import itemgetter
import chainlit as cl
import tempfile
import os
import pandas as pd

# Takes uploaded PDFs, creates document chunks, computes embeddings
# Stores document chunks and embeddings in a Vector DB
# Returns a retriever which can look up the Vector DB
# to return documents based on user input
def configure_retriever(uploaded_files):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        docs = []

        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir, file.name)

            # Ensure file exists before reading
            if not os.path.exists(file.path):
                raise FileNotFoundError(f"Uploaded file not found: {file.path}")

            # Copy file from Chainlit temp path to our temp directory
            with open(temp_filepath, "wb") as f:
                with open(file.path, "rb") as infile:
                    f.write(infile.read())

            # Load PDF into documents
            loader = PyMuPDFLoader(temp_filepath)
            loaded_docs = loader.load()

            if not loaded_docs:
                raise ValueError(f"No text found in PDF: {file.name}")

            docs.extend(loaded_docs)

        print(f"Total documents loaded: {len(docs)}")

        # Ensure documents contain extractable text
        for doc in docs:
            source = doc.metadata.get("source", "Unknown source") 
            if not doc.page_content.strip():
                raise ValueError(f"Document {source} contains no extractable text!")

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        doc_chunks = text_splitter.split_documents(docs)

        print(f"Number of document chunks: {len(doc_chunks)}")

        if not doc_chunks:
            raise ValueError("Document splitting failed! Try adjusting chunk size.")

        # Generate embeddings
        doc_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", task_type="retrieval_document"
        )

        # Create Chroma vector store
        vectordb = Chroma.from_documents(doc_chunks, embedding=doc_embeddings)

        # Define retriever object
        retriever = vectordb.as_retriever()

        return retriever


@cl.on_chat_start
# this function is called when the app starts for the first time
async def when_chat_starts():
  # Create UI element to accept PDF uploads
  uploaded_files = None
  # Wait for the user to upload a file
  while uploaded_files == None:
    uploaded_files = await cl.AskFileMessage(
      content="Please upload PDF documents to continue.",
      accept=["application/pdf"],
      max_size_mb=20, max_files=5, timeout=180
    ).send()

  msg = cl.Message(content=f"Processing files please wait...", disable_feedback=True)
  await msg.send()
  await cl.sleep(2)
  # Create retriever object based on uploaded PDFs
  retriever = configure_retriever(uploaded_files)
  msg = cl.Message(content=f"Processing completed. You can now ask questions!", disable_feedback=True)
  await msg.send()
  


  # Load Gemini LLM
  gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-thinking-exp",
    temperature=0.8,
    max_tokens=1024,
    convert_system_message_to_human=True,
  )

  # Create a prompt template for QA RAG System
  qa_template = """
                Use only the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know,
                don't try to make up an answer. Keep the answer as concise as possible.

                {context}

                Question: {question}
                """
  qa_prompt = ChatPromptTemplate.from_template(qa_template)

  # This function formats retrieved documents before sending to LLM
  def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

  # Create a QA RAG System Chain
  qa_rag_chain = (
    {
      "context": itemgetter("question") # based on the user question get context docs
        |
      retriever
        |
      format_docs,
      "question": itemgetter("question") # user question
    }
      |
    qa_prompt # prompt with above user question and context
      |
    gemini_model # above prompt is sent to the LLM for response
      |
    StrOutputParser() # to parse the output to show on UI
  )

  # Set session variables to be accessed when user enters prompts in the app
  cl.user_session.set("qa_rag_chain", qa_rag_chain)

@cl.on_message
# this function is called whenever the user sends a prompt message in the app
async def on_user_message(message: cl.Message):

  # get the chain and memory objects from the session variables
  qa_rag_chain = cl.user_session.get("qa_rag_chain")

  # this will store the response from ChatGPT LLM
  chatgpt_message = cl.Message(content="")

  #Callback handler for handling the retriever and LLM processes.
  # Used to post the sources of the retrieved documents as a Chainlit element.
  class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: cl.Message):
      BaseCallbackHandler.__init__(self)
      self.msg = msg
      self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
      source_ids = []
      for d in documents: # retrieved documents from retriever based on user query
        metadata = {
          "source": d.metadata["source"],
          "page": d.metadata["page"],
          "content": d.page_content[:200]
        }
        idx = (metadata["source"], metadata["page"])
        if idx not in source_ids: # store unique source documents
          source_ids.append(idx)
          self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
      if len(self.sources):
          sources_table = pd.DataFrame(self.sources[:3]).to_markdown()
          self.msg.elements.append(
            cl.Text(name="Sources", content=sources_table, display="inline")
          )

  # Stream the response from ChatGPT and show in real-time
  async with cl.Step(type="run", name="QA Assistant"):
    async for chunk in qa_rag_chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[
            cl.LangchainCallbackHandler(),
            PostMessageHandler(chatgpt_message)
        ]),
    ):
        await chatgpt_message.stream_token(chunk)
  await chatgpt_message.send()
