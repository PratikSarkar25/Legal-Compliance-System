import json
import re
from pathlib import Path

#INPUT_FILE = "./cuad-data/CUADv1.json"
INPUT_FILE = "./data/cuad-data/CUADv1.json"
OUTPUT_FILE = "./src/legal_nlp/prompts/cuad_questions.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = []

for idx, qa in enumerate(data["data"][0]["paragraphs"][0]["qas"], start=1):

    prompt = qa["question"]

    match = re.search(r'related to "(.*?)"', prompt)

    if not match:
        continue

    display_name = match.group(1)

    clause_type = (
        display_name.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )

    while "__" in clause_type:
        clause_type = clause_type.replace("__", "_")

    questions.append(
        {
            "id": idx,
            "clause_type": clause_type,
            "display_name": display_name,
            "prompt": prompt,
        }
    )

output_path = Path(OUTPUT_FILE)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)

print(f"Saved {len(questions)} CUAD questions to {OUTPUT_FILE}")