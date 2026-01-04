from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

mn="Qwen/Qwen2.5-Coder-0.5B-Instruct"
print("loading model (it takes  few seconds)")
tokenizer=AutoTokenizer.from_pretrained(mn)
model=AutoModelForCausalLM.from_pretrained(
    mn,
    torch_dtype=torch.float64,
    device_map="auto"

)

print("Model rendered successfully")
print(f"Model has {model.num_parameters} parameters")

pmpt="you are lovely but you are not Panchachuli lovely"

input=tokenizer(pmpt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **input,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id
    )

    

result=tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\ngenerated text:")
print(result)


