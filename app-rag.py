from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
import gradio as gr

llm = Ollama(model="codellama", request_timeout=300.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(
    input_files=["./data/paul_graham_essay.txt"]
).load_data()

index = VectorStoreIndex.from_documents(documents)

index.storage_context.persist(persist_dir="data")

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="data")

# load index
vector_index = load_index_from_storage(storage_context)

query_engine = vector_index.as_query_engine(similarity_top_k=1)


def query(text):
    z = query_engine.query(text)
    return z

def interface(text):
    z = query(text)
    response = z.response
    return response

with gr.Blocks(theme=gr.themes.Glass().set(block_title_text_color= "black", body_background_fill="black", input_background_fill= "black", body_text_color="white")) as demo:
    with gr.Row():
        output_text = gr.Textbox(lines=20)
        
    with gr.Row():
        input_text = gr.Textbox(label='Enter your query here')
        
    input_text.submit(fn=interface, inputs=input_text, outputs=output_text)
                      
demo.launch()
