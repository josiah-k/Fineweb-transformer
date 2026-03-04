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
"""

"""
config:
n_heads: int,
n_layers: int,
hidden_dim: int,
max_seq_len: int,
vocab_size: int
"""

class Train_Transformer():
    def __init__(self, load_model: bool=False,
                 config: dict=None, # if load_model=False, must provide config
                 model_dir: str="model",
                 epoch_num: int=None):
        if load_model:
            model_state = load_model_data(model_dir, epoch_num)
        else:
            model_state = None
        self.initialize_model(config, model_state)


    def initialize_model(self, config, model_state: dict=None):
        self.config = config
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_seq_len']
        self.transformer_model = Transformer(n_heads=config['n_heads'],
                                n_layers=config['n_layers'],
                                hidden_dim=config['hidden_dim'],
                                max_len=config['max_seq_len'],
                                vocab_size=config['vocab_size'])
        if model_state is not None:
            self.transformer_model.load_state_dict(model_state)


    def make_and_save_vocab(self, tokenizer_path="fineweb_bytebpe",
                             max_docs=5_000, min_frequency=2):

        make_vocab(tokenizer_path, max_docs, self.vocab_size, min_frequency)


    def initialize_tokenizer(self, tokenizer_path="fineweb_bytebpe"):

        token_dir = Path(tokenizer_path)
        vocab = token_dir / "vocab.json"
        merges = token_dir / "merges.txt"

        self.tok = ByteLevelBPETokenizer(str(vocab), str(merges))
        self.tok.enable_truncation(max_length=self.max_seq_len+1) # add one because trimming one when creating test/train set
        self.pad_id = self.tok.token_to_id("<pad>")


    def train(self, learning_rate: float,
              n_epochs: int,
              batch_size: int,
              model_dir="model"):

        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)

        train_dataloader = get_dataloader(batch_size, self.pad_id, self.tok)

        optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        train_loss = []
        for epoch in trange(n_epochs, desc='Training model'):
            losses = 0
            self.transformer_model.train()

            for data in train_dataloader:
                src, tgt, pad_mask = data['input_ids'], data['output_ids'], data['pad_mask']

                logits, _ = self.transformer_model(src, pad_mask, kv_cache=None) # (B, T, vocab_size)

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
                'config': self.config,
                'model_state_dict': self.transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_filename)

            wandb.log({
                "loss":losses
            })


    def run_inference(self, batch_size: int,
                      max_gen_len: int = 50,
                      greedy: bool = False): #300

        test_dataloader = get_dataloader(batch_size, self.pad_id, self.tok)

        prompt = next(iter(test_dataloader))
        src, mask = prompt['input_ids'], prompt['pad_mask']
        trunc_ind = self.max_seq_len - max_gen_len
        src_prompt, mask = src[:,:trunc_ind], mask[:,:trunc_ind] # truncating so can test
        src_prompt_test = src_prompt ######

        self.stop_id = self.tok.token_to_id("</s>")
        """confirm this is actually stop token"""

        self.transformer_model.eval()
        kv_cache = None
        for i in range(max_gen_len):
            tkns = src_prompt if i == 0 else next_tkn

            mask = torch.ones_like(tkns)
            logits, kv_cache = self.transformer_model(tkns, mask, kv_cache) # (B, len(tkns), vocab_size)
            if greedy:
                next_tkn = torch.argmax(logits, dim=-1).unsqueeze(-1)
            else:
                probs = torch.softmax(logits[:,-1], dim=-1)
                next_tkn = torch.multinomial(probs, num_samples=1)

            src_prompt = torch.cat((src_prompt,next_tkn), dim=-1)

        ###### for testing kv_cache implementation ######
        kv_cache = None
        for _ in range(max_gen_len):

            mask = torch.ones_like(src_prompt_test)
            logits, _ = self.transformer_model(src_prompt_test, mask, kv_cache) # (B, T, vocab_size)
            probs = torch.softmax(logits[:,-1], dim=-1)
            # next_tkn = torch.multinomial(probs, num_samples=1)
            next_tkn = torch.argmax(probs, dim=-1).unsqueeze(-1)

            src_prompt_test = torch.cat((src_prompt_test,next_tkn), dim=-1)

        print(torch.eq(src_prompt,src_prompt_test))
        """may need to clear output after stop token"""
        # src_seq = [seq[seq != stop_id/pad_id].tolist() for seq in src] # to clear unwanted ouput tokens
        decoded_src = self.tok.decode_batch(src.tolist())
        decoded_gen = self.tok.decode_batch(src_prompt.tolist())
        decoded_gen_test = self.tok.decode_batch(src_prompt_test.tolist()) ########
        return decoded_src, decoded_gen, decoded_gen_test



def get_dataloader(batch_size: int,
                   pad_id,
                   tokenizer: callable,
                   subset_size: int = 2560,
                   buffer_size: int = 2_000):

    # Initialize dataloader
    data = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="default",
        split="train",
        streaming=True
    )
    data = data.take(subset_size) # choose subset size to load
    data = data.shuffle(buffer_size=buffer_size)
    data = data.map(partial(tokenize, tokenizer=tokenizer))

    dataloader = DataLoader(
        data, batch_size=batch_size,
        collate_fn=partial(collate_fn, pad_id=pad_id))

    return dataloader


def load_model_data(model_dir, epoch_num):
    model_dir = Path(model_dir)
    model_path = model_dir / epoch_num
    all_data = torch.load(model_path)

    return all_data['config'], all_data['model_state_dict']


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