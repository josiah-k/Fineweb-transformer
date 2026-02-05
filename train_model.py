from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from torch.utils.data import DataLoader
import torch
from functools import partial
from multi_head_attn import Transformer
import torch.nn as nn
from tqdm import trange
from make_vocab import make_vocab
import wandb

"""
Todo:
- add load prev model funcs etc
"""


class Train_Transformer():
    def __init__(self, n_heads: int,
                        n_layers: int,
                        hidden_dim: int,
                        max_seq_len: int,
                        vocab_size: int):

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.transformer_model = Transformer(n_heads=n_heads,
                                n_layers=n_layers,
                                hidden_dim=hidden_dim,
                                max_len=max_seq_len,
                                vocab_size=vocab_size)

    def make_and_save_vocab(self, tokenizer_path="project/fineweb_bytebpe",
                             max_docs=5_000, min_frequency=2):

        make_vocab(tokenizer_path, max_docs, self.vocab_size, min_frequency)

    def initialize_tokenizer(self, tokenizer_path="project/fineweb_bytebpe"):

        token_dir = Path(tokenizer_path)
        vocab = token_dir / "vocab.json"
        merges = token_dir / "merges.txt"

        self.tok = ByteLevelBPETokenizer(str(vocab), str(merges))
        self.tok.enable_truncation(max_length=self.max_seq_len+1) # add one because trimming one when creating test/train set
        self.pad_id = self.tok.token_to_id("<pad>")

    def train(self, learning_rate: float,
              n_epochs: int,
              batch_size: int,
              model_dir="project/model"):

        # Initialize dataloader
        training_data = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="default",
            split="train",
            streaming=True
        )
        training_data = training_data.shuffle(buffer_size=2_000)
        training_data = training_data.map(partial(tokenize, tokenizer=self.tok))

        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)

        train_dataloader = DataLoader(
            training_data, batch_size=batch_size,
            collate_fn=partial(collate_fn, pad_id=self.pad_id))


        optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        train_loss = []
        for epoch in trange(n_epochs, desc='Training model'):
            losses = 0
            self.transformer_model.train()

            for data in train_dataloader:
                src, tgt, mask = data['input_ids'], data['output_ids'], data['pad_mask']

                logits = self.transformer_model(src, pad_mask=mask)

                optimizer.zero_grad()

                ce_loss = loss_fn(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt.reshape(-1)
                )
                losses += ce_loss.item()
                ce_loss.backward()

                optimizer.step()

            train_loss.append(losses)
            self.transformer_model.eval()

            # save model and log loss after every epoch
            model_filename = model_dir / f"{epoch:02d}"

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_filename)

            wandb.log({
                "loss":losses
            })



def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch) - 1
    input_ids, pad_mask, output_ids = [], [], []

    for x in batch:
        ids = x["input_ids"][:-1]
        pad_len = max_len - len(ids)

        input_ids.append(
            ids + [pad_id]*pad_len
        )
        pad_mask.append(
            [1]*len(ids) + [0]*pad_len
        )
        output_ids.append(
            x["input_ids"][1:] + [pad_id]*pad_len
        )

    return {
        "input_ids": torch.LongTensor(input_ids),
        "pad_mask": torch.tensor(pad_mask).int(),
        "output_ids": torch.LongTensor(output_ids)
    }

def tokenize(example: dict, tokenizer: callable):
    enc = tokenizer.encode(example['text'])
    return {"input_ids": enc.ids}