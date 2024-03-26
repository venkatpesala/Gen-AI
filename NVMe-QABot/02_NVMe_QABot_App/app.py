# https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/
import streamlit as st
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore")

model_id = 'meta-llama/Llama-2-7b-chat-hf'
hf_auth = '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'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
root_path = '/content/'
chat_history = []

def initialize_HF_model():
    
    global model

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # begin initializing HF items, you need an access token
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,                                                                                                                                        
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )

    # enable evaluation mode to allow model inference
    model.eval()

    print(f"Model loaded on {device}")

def initialize_HF_tokenizer():
    
    global tokenizer

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

def initialize_stopping_criteria():

    global stopping_criteria

    stop_list = ['\nHuman:', '\n```\n']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]

    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    # define custom stopping criteria object
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

def initialize_HF_pipeline():

    global generate_text

    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

def initialize_langchain_HF_pipeline():

    global chain

    llm = HuggingFacePipeline(pipeline=generate_text)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # storing embeddings in the vector store
    vectorstore = FAISS.load_local(root_path + 'faiss_index/', embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

def get_answer(query):
    
    result = chain({"question": query, "chat_history": chat_history})
    
    chat_history.append((query, result['answer']))

    return result['answer']

def init_QABot():
    initialize_HF_model()
    initialize_HF_tokenizer()
    initialize_stopping_criteria()
    initialize_HF_pipeline()
    initialize_langchain_HF_pipeline()

init_QABot()

# App title
st.set_page_config(page_title="NVMe - QABot")

# Hugging Face Credentials
with st.sidebar:
    st.title('NVMe - QABot')
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input):
    return get_answer(prompt_input)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)