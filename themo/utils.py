import functools
import typing as tp
import warnings

import pyarrow as pa
import torch
import tqdm
import transformers

_TQDM_WIDTH = 120

# This is a temporal script to store the translation function

# opus-en-es model: "Helsinki-NLP/opus-mt-en-es"
# opus-es-en model: "Helsinki-NLP/opus-mt-es-en"
# nllb model: "facebook/nllb-200-distilled-600M"
# nllb en code -> "eng_Latn"
# nllb es code -> "spa_Latn"


def translate_target_sentences(
    target_sentences: tp.Sequence[pa.StringScalar],
    model_path: str,
    batch_size: int,
    num_workers: int,
    src_lang: str = None,
    target_lang: str = None,
) -> tp.Sequence[pa.StringScalar]:
    print("Translating target features, this might take a while...")

    if src_lang:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, src_lang=src_lang
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    collate_fn = functools.partial(
        tokenizer,
        truncation=True,
        max_length=77,  # not sure
        padding="max_length",
        return_tensors="pt",
    )

    dloader = torch.utils.data.DataLoader(
        target_sentences,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("No GPU found, switching to CPU mode", RuntimeWarning)
    model.to(device)
    results = []
    for batch in tqdm.tqdm(
        dloader,
        desc="Computing features",
        unit="batch",
        ncols=_TQDM_WIDTH,
    ):
        if target_lang:
            generated_ids = model.generate(
                **batch.to(device),
                forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
            )
        else:
            generated_ids = model.generate(**batch.to(device))
        results.extend(
            tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        )
    return pa.Table.from_pylist(results)
