from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
import gradio as gr
import os.path

# Grab the LLM and create an object.
llm = Ollama(model="llama3", request_timeout=300.0)
# use an embedding model to create embeddings from the data.
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Add some global LlamaIndex rleated settings, we can also define chunk size here.
Settings.llm = llm
Settings.embed_model = embed_model

# Create the vector store, using llamaindex.
# Save it to disk, so we don't have to keep on reloading it.
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader(
        input_files=["./data/opensocial.txt"]
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Define the query engine, it's LLM powered by the vector store.
# Using the Settings defined above.
# For speed purposes, we use similarity_top_k 1, basically
# retrieve the top 1 most relevant documents to the query.
query_engine = index.as_query_engine(similarity_top_k=1)

# run the query using the text from the interface.
def query(text):
    z = query_engine.query(text)
    return z

def interface(text):
    z = query(text)
    response = z.response
    return response

# provided by gradio.
interface = gr.Interface(
        fn = interface,
        inputs = gr.Textbox(lines=4,placeholder="Enter your Prompt"),
        outputs = 'text'
    )
interface.launch()        
