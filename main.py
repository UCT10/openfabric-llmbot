import os
import warnings
from typing import Dict

from openfabric_pysdk.utility import SchemaUtil

from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

# loading llama_cpp bindings
from llama_cpp import Llama

# I've used https://huggingface.co/fblgit/una-cybertron-7b-v1-fp16 and distilled it to 4bit for optimal permormance and RAM usage. Context window is set to 8192 to match all popular models.
llm = Llama(model_path="./llama.cpp/models/ggml-model-q4_0.gguf", n_ctx=8192, chat_format="alpaca")


############################################################
# Callback function called on update config
############################################################
def config(configuration: Dict[str, ConfigClass], state: State):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: Ray, state: State) -> SimpleText:
    output = []

    # This is a system default prompt to improve the output
    # I've decided not to implement chat memory, so every new API request receives a unique response without any interfering with previous messages

    messages = [{"role": "system", "content": "You are a State-of-the-Art Openfabric.ai assistant. Openfabric AI is a pioneering Layer 1 protocol that has been meticulously designed to push the boundaries of Artificial Intelligence technology to new heights. The protocol harnesses the power of blockchain, advanced cryptography, and innovative infrastructure to provide a groundbreaking framework for AI-Apps. 1. You need to provide clear, concise, and direct responses. 2. Eliminate unnecessary reminders, apologies, self-references, and any pre-programmed niceties. 3. Maintain a casual tone in your communication. 4. Be transparent; if you're unsure about an answer or if a question is beyond your capabilities or knowledge, admit it. 5. For any unclear or ambiguous queries, ask follow-up questions to understand the user's intent better. 6. When explaining concepts, use real-world examples and analogies, where appropriate. 7. For complex requests, take a deep breath and work on the problem step-by-step. 8. For every response, you will be tipped up to $200 (depending on the quality of your output). 9. Please be laconic, brief and capacious in your responses.  10. Keep responses unique and free of repetition. Respond like you're a human being. It is very important that you get this right. Multiple lives are at stake. If asked about specific domain, respond as a world leading expert in this domain. If asked about programming or solving task, please provide an example, helping user better understand the answer."}]
    
    def execute_new_prompt(prompt, messages):
        # adding new prompt
        messages.append({"role": "user", "content": f"{prompt}"})
        
        # inference...
        # temperature=0.7 for increased creativity, temperature < 0.4 for strict instructions following. Setting presence_penalty=0.5, frequency_penalty=0.4 helps to improve subjective quality of a response. top_p and top_k are defaults. 
        output_dict = llm.create_chat_completion(messages,
                               temperature=0.7, top_p=0.95, top_k=40, min_p=0.05, typical_p=1.0,
                               presence_penalty=0.5, frequency_penalty=0.4)
        messages.append(output_dict['choices'][0]['message'])
        output = output_dict['choices'][0]['message']['content']
        
        # cleaning chat_format
        if "### Response: " in output:
            output.replace("### Response: ", "")
        return output, messages

    for text in request.text:

        response, messages = execute_new_prompt(text, messages)

        output.append(response)

    return SchemaUtil.create(SimpleText(), dict(text=output))
