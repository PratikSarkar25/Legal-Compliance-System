from typing import Any, Dict, List

from transformers import PreTrainedTokenizerBase


def chunk_text_with_stride(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping token windows using the tokenizer.

    Returns a list of chunks together with their token offsets.
    """

    if not text.strip():
        return []

    encoding = tokenizer(
        text,
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )

    chunks = []

    for i in range(len(encoding["input_ids"])):
        offsets = encoding["offset_mapping"][i]

        start_char = offsets[0][0]

        end_char = offsets[-1][1]

        chunks.append(
            {
                "input_ids": encoding["input_ids"][i],
                "attention_mask": encoding["attention_mask"][i],
                "start_char": start_char,
                "end_char": end_char,
                "text": text[start_char:end_char],
            }
        )

    return chunks