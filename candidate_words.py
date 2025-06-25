import os
import json
import re
from collections import Counter

INPUT_DIR = "data/json_files"
files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json") or f.endswith(".jsonl")]

skip_words = set([
    "is", "a", "with", "and", "the", "more", "has", "have", "in", "to", "it", "of", "dress", "color", "colored",
    "this", "that", "as", "for", "on", "off", "i", "my", "than", "no", "one", "s", "t", "was", "are", "you", "but"
])

word_counter = Counter()
phrase_counter = Counter()

def load_json_file(path):
    if path.endswith(".jsonl"):
        return [json.loads(line) for line in open(path) if line.strip()]
    else:
        return json.load(open(path))

for file_name in files:
    file_path = os.path.join(INPUT_DIR, file_name)
    try:
        data = load_json_file(file_path)
    except Exception as e:
        print(f"⚠️ Skipping {file_name}: {e}")
        continue

    for item in data:
        if not isinstance(item, dict):  # 👈 safeguard here
            continue
        captions = item.get("captions", [])
        for caption in captions:
            caption = caption.lower()
            words = re.findall(r'\b[a-z]+\b', caption)
            words = [w for w in words if w not in skip_words]
            word_counter.update(words)

            tokens = caption.split()
            bigrams = zip(tokens, tokens[1:])
            trigrams = zip(tokens, tokens[1:], tokens[2:])
            phrase_counter.update([" ".join(b) for b in bigrams])
            phrase_counter.update([" ".join(t) for t in trigrams])

print("🔠 Top 50 Words:")
print("----------------")
for word, count in word_counter.most_common(50):
    print(f"{word:20} {count}")

print("\n🧩 Top 30 Descriptive Phrases:")
print("-------------------------------")
for phrase, count in phrase_counter.most_common(30):
    print(f"{phrase:30} {count}")