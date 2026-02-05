from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path


def make_vocab(tokenizer_path: str,
               max_docs: int,
               vocab_size: int,
               min_frequency: int):

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="default",
        split="train",
        streaming=True
    )

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train_from_iterator(
        iterator=text_iterator(dataset, max_docs),
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )

    # save tokenizer files
    token_dir = Path(tokenizer_path)
    token_dir.mkdir(exist_ok=True)
    tokenizer.save_model(str(token_dir))


def text_iterator(dataset, max_docs):
    for i, example in enumerate(dataset):
        if i > max_docs:
            break
        text = example['text']
        if text is not None and len(text) > 0:
            yield text
