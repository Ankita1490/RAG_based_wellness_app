import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def load_generation_pipeline(model_name: str):
    """
    Load Tokenizer, model, and generation pipeline for Qwen instaruct model.
    """
    tokenizer =AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        
    )

    generator = pipeline(
        task = "text-generation",
        model = model,
        tokenizer = tokenizer,
        device =-1
    )

    return generator, tokenizer

def generate_answer(generator, tokenizer, prompt: str, max_new_tokens: int =256) -> str:
    """
    Generate an answer using the provided generation pipeline and prompt.

    Args:
        generator: The text generation pipeline
        tokenizer: The tokenizer associated with the model
        prompt: The input prompt for generation
        max_new_tokens: Maximum number of new tokens to generate
    Returns:
        The generated answer as a string
    """
    messages = [
        {
            "role" : "user",
            "content" : prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )

    output = generator(
        formatted_prompt,
        max_new_tokens = max_new_tokens,
        do_sample = False,
        return_full_text = False,
        pad_token_id = tokenizer.eos_token_id
    )

    return output[0]["generated_text"].strip()