"""
Model loader for RLHF training
loads and configures language models for code generation
"""

from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch

def model_tokenizer(
        model_name:str="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        use_4bit:bool= True
                        ):
    """
    load a code generation model and tokenizer
    
    
    Args:
        model_name(str): name of the model to be used to generate code
        use_4bit(bool):use 4 bit quantization for better memory usage
        
    Returns:
        Tuple:(model, Tokenizer)
        
        
    example:
        >>>model, teokenizer=model_tokenizer()
        >>>print(f"Model has been loaded {model_name})"""
    print(f"Model has been loaded {model_name}")


    if use_4bit:
        q_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        q_config=None
#---------------TOKENIZER-----------------------
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

#---------------MODEL---------------------------
    model=AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=q_config,
        device_map="auto",
        trust_remote_code=True
        )
    
    print("MODEL LOADED SUCCESFULLY")
    print(f"DEVICE IS:{model.device}")
    print(f"Parameters for the model:{model.num_parameters()}")

    return model,tokenizer

def gen_code(model, tokenizer , prompt:str, max_length:int=150):
    """generate code from thew given prompt

    Args:
        model : Th emodel that will egenrate code
        tokenizer : The tokenizer that the model will use
        prompt : User prompt given by the user
        max_length : Maximum legth for the generated output code, Defaults to 150. 

    Returns:
        str: generated code with precision

        example:
            >>>model, tokenizer= model_tokenizer()
            >>>code=gen_code(model, tokenizer, prompt="def add(a,b)")
            >>>print(code)
"""
#-----------------------INPUTS AND OUTPUTS------------------------0-    
    inputs=tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs=model.generate(
        **inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id
    )
#------------------------------------------------------------------
    final_output=tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("CODE GENERATED SUCCESFULLY")
    return final_output

if __name__ == "__main__":
    """
    Test model loading and generation.
    """
    print("=" * 60)
    print("Testing Model Loader")
    print("=" * 60)
    
    # Load model
    model, tokenizer = model_tokenizer()
    
    # Test generation
    print("\n[Test] Generating code for 'def add(a, b):'")
    prompt = "Add two numbers "
    generated = gen_code(model, tokenizer, prompt, max_length=50)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated:\n{generated}")
    
    print("\n✅ Model loader working!")
    