"""Create a tiny BERT with random weights for OFFLINE testing of the code logic
(no download of bert-base-uncased). For logic tests only, never for real results."""
import sys
from pathlib import Path
from transformers import BertConfig, BertModel, BertTokenizerFast

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny_bert")
out.mkdir(parents=True, exist_ok=True)
words = ("i am happy you came what a surprise leave me alone so sad angry afraid "
         "the of and to in it is was this that very much !").split()
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(words)) + list(".,?")
(out / "vocab.txt").write_text("\n".join(vocab))
BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=True).save_pretrained(out)
cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                 num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
BertModel(cfg).save_pretrained(out)
print("tiny BERT saved to", out)
