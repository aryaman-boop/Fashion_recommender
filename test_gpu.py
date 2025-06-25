from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # will auto-distribute to GPU if available
    load_in_4bit=True,  # efficient loading using bitsandbytes
    torch_dtype="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """### Instruction:
Extract dress features and RGB color from the description.

### Input:
is a long black dress

### Output:"""

print("Generating output...\n")
out = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.3)[0]['generated_text']
print(out)