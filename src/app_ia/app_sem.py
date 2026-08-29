from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype="auto"
)

prompt = "Representa a estrutura do caractere chinês para água."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=50
)

generated = output[0][inputs["input_ids"].shape[1]:]

print("Tokens generados:", generated.tolist(), end="\n\n")
print("Resultado:", repr(tokenizer.decode(generated, skip_special_tokens=False)))