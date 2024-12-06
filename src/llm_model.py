"""
llm_model.py
-------------

Description:
    This module loads a large language model and answers user queries.

Author:
    Umair Cheema <cheemzgpt@gmail.com>

Version:
    1.0.0

License:
    Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

Date Created:
    2024-11-30

Last Modified:
    2024-11-30

Python Version:
    3.8+

Usage:
    Instantiate this model and answer user question
    Example:
        from llm_model import VRAGLLMModel

Dependencies:

"""


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from vragconfig import VRAGConfig


class VRAGLLMModel:
    def __init__(self):
        self.config = VRAGConfig(file_path='./vrag.yaml').read()
        self.model_name = self.config['llm_name']
        self.model_path = self.config['llm_path']
        self.device = self.config['llm_device']
        self.model = None
        self.system_prompt = self.config['llm_system_prompt']
        self.rag_top_k = self.config['rag_top_k']
        self.rag_top_p = self.config['rag_top_p']
        self.rag_repition_penalty = self.config['rag_repition_penalty']
        self.rag_streaming = self.config['rag_streaming']
        self.rag_temperature = self.config['rag_temperature']
        self.rag_max_new_tokens = self.config['rag_max_new_tokens']
        self.rag_do_sample = self.config['rag_do_sample']
        self.rag_retriever_documents = self.config['rag_retriever_documents']
        self.debug = self.config['rag_debug']
       
        self.tokenizer = None
        self.terminators = None

    def load_model(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.model = AutoModelForCausalLM.from_pretrained(
                   self.model_path,
                   torch_dtype=torch.bfloat16,
                   device_map=self.device
            )
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config['vs_embeddings_model_path'])
        self.vectordb = Chroma(persist_directory=self.config['vs_output_folder'], embedding_function=self.embeddings)

     
    def generate_response(self, user_query):
        
        #Find similar documents
        retrieved_docs = self.vectordb.similarity_search(user_query, k=self.rag_retriever_documents)
        context = "\n\n".join(
                     (f"{doc.page_content}")
                        for doc in retrieved_docs
                    )
        
        prompt = f"""Use the following pieces of context to answer the question in less than 150 words. If no context provided, say I dont know.
                   {context}
                   Question: {user_query}
                   """ 
        
        if(self.debug):
            print(prompt)

        messages = [
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": prompt},
         ]
        

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.rag_max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=self.rag_do_sample,
            temperature=self.rag_temperature,
            top_p= self.rag_top_p,
            top_k = self.rag_top_k

        )
        
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
