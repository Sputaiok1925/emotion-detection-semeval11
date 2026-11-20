"""Local German/Spanish -> English translation with a resumable cache."""
import json
from pathlib import Path

MODELS = {
    "deu": "Helsinki-NLP/opus-mt-de-en",
    "esp": "Helsinki-NLP/opus-mt-es-en",
}


def split_long_text(text, tokenizer, limit):
    """Split long texts instead of silently cutting off their ending."""
    if len(tokenizer.encode(text)) <= limit:
        return [text]
    words = text.split()
    if len(words) < 2:
        raise ValueError("A single word exceeds the translation model's input limit")
    middle = len(words) // 2
    return (split_long_text(" ".join(words[:middle]), tokenizer, limit)
            + split_long_text(" ".join(words[middle:]), tokenizer, limit))


def save_cache(path, model_name, translations):
    # Replace the file only after the new version has been written completely.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"model": model_name, "translations": translations},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def translate_batch(texts, tokenizer, model, device, limit, depth=0):
    """Retry a translation that hit the limit using smaller source chunks."""
    import torch

    inputs = tokenizer(texts, padding=True, truncation=False,
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, num_beams=4, do_sample=False,
                                max_new_tokens=limit - 1)
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    results = []
    for text, tokens, translation in zip(texts, output, decoded):
        generated = tokens[1:].tolist()
        eos = model.config.eos_token_id
        complete = (eos in generated and generated.index(eos) + 1 < limit - 1)
        if complete:
            results.append(translation.strip())
            continue
        words = text.split()
        if depth >= 4 or len(words) < 2:
            # An empty result is removed and recorded by prepare_extended_data.py.
            print("Skipped a text that repeatedly reached the translation limit:",
                  repr(text[:100]), flush=True)
            results.append("")
            continue
        middle = len(words) // 2
        print("Translation limit reached; retrying smaller chunks.", flush=True)
        chunks = translate_batch([" ".join(words[:middle]), " ".join(words[middle:])],
                                 tokenizer, model, device, limit, depth + 1)
        results.append(" ".join(chunks) if all(chunks) else "")
    return results


def translate_texts(texts, language, cache_dir, batch_size=8, device="auto"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{language}_to_eng.json"
    model_name = MODELS[language]
    translations = {}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["model"] != model_name:
            raise ValueError(f"Translation model differs from the cache: {path}")
        translations = saved["translations"]

    # Repeated source texts are translated only once, even across source splits.
    missing = [text for text in dict.fromkeys(texts) if text not in translations]
    print(f"{language}: {len(missing)} new texts to translate", flush=True)
    if missing:
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device).eval()
        # Only max_new_tokens controls output length; avoid conflicting defaults.
        model.generation_config.max_length = None
        limit = min(512, model.config.max_position_embeddings)
        print(f"Translation device: {device}", flush=True)
        for start in range(0, len(missing), batch_size):
            originals = missing[start:start + batch_size]
            parts = [split_long_text(text, tokenizer, limit) for text in originals]
            flat_parts = [part for group in parts for part in group]
            translated_parts = []
            for offset in range(0, len(flat_parts), batch_size):
                translated_parts.extend(translate_batch(
                    flat_parts[offset:offset + batch_size], tokenizer, model, device, limit))
            offset = 0
            for original, group in zip(originals, parts):
                pieces = translated_parts[offset:offset + len(group)]
                translations[original] = " ".join(pieces).strip() if all(pieces) else ""
                offset += len(group)
            save_cache(path, model_name, translations)
            print(f"{language}: {min(start + batch_size, len(missing))}/{len(missing)} translated", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return [translations[text] for text in texts]
