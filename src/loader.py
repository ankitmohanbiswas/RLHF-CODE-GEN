"""
Model loader for RLHF training
loads and configures language models for code generation
"""

from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch
from executor import load_problems


def model_tokenizer(model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct"):
    """
    Load model and tokenizer with GPU support.
    """
    import torch
    
    print(f"Model has been loaded {model_name}")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model to GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 for speed
        device_map="auto"           # Automatically uses GPU
    )
    
    print("MODEL LOADED SUCCESSFULLY")
    print(f"DEVICE IS: {model.device}")
    print(f"Parameters for the model: {model.num_parameters()}")
    
    return model, tokenizer

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
    il=inputs['input_ids'].shape[1]
    outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # Maximum NEW tokens only
            min_new_tokens=10,           # At least generate something
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            early_stopping=True,         # Stop at EOS token
            num_beams=1,   
    )

    new_tokens = outputs[0][il:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
#------------------------------------------------------------------
    
    final_output=prompt+ "\n" + generated
    lines = final_output.split('\n')
    function_lines = []
    in_function = False
    for line in lines:
        # Start of function
        if line.strip().startswith('def '):
            in_function = True
            function_lines.append(line)
        # Inside function (indented)
        elif in_function and (line.startswith('    ') or line.startswith('\t') or line.strip() == ''):
            function_lines.append(line)
        # End of function (non-indented line)
        elif in_function and line.strip() and not line.startswith(' '):
            break
    
    # Return only the function
    clean_code = '\n'.join(function_lines)
    # ==================================================
    
    print("CODE GENERATED SUCCESSFULLY")
    return clean_code if clean_code else final_output  # Fallback to full if extraction fails

if __name__ == "__main__":
    """
    Test model loading and generation.
    """
    print("=" * 60)
    print("Testing Model Loader")
    print("=" * 60)
    
    # Load model
    model, tokenizer = model_tokenizer()
    problem=load_problems()
    
    # Test generation
    print("\n[Test] Generating code for 'def add(a, b):'")
    prompt = problem[1]
    generated = gen_code(model, tokenizer, prompt, max_length=50)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated:\n{generated}")
    
    print("\n Model loader working!")
    