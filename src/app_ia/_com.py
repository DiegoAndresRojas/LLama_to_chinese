import torch
import xgrammar as xgr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)

# model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

config = AutoConfig.from_pretrained(model_name)

with open("grammar.ebnf", encoding="utf-8") as f:
    grammar = f.read()

tokenizer_info = xgr.TokenizerInfo.from_huggingface(
    tokenizer,
    vocab_size=config.vocab_size
)

compiler = xgr.GrammarCompiler(tokenizer_info)
compiled_grammar = compiler.compile_grammar(grammar)

processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

prompt = "Representa a estrutura do caractere chinês para água."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=50,
    logits_processor=[processor]
)

generated = output[0][inputs["input_ids"].shape[1]:]

print("Tokens generados:", generated.tolist())
print("Resultado:", repr(tokenizer.decode(generated, skip_special_tokens=False)))