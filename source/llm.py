from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_community.llms import HuggingFacePipeline

def get_huggingface_llm():
    model_name = "lmsys/vicuna-7b-v1.5"

    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set device to "cpu"
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu")
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
    )
    return llm
