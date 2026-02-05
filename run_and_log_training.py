import os
import sys
import wandb

from train_model import Train_Transformer

MAIN = __name__ == "__main__"

if MAIN:
    ## ----- training settings ----- ##
    # batch_size = float(sys.argv[1])
    # learning_rate = float(sys.argv[2])
    batch_size = 64
    learning_rate = 1e-4

    max_seq_len = 512
    n_epochs = 2
    build_vocab = True


    ## ----- model settings ----- ##
    vocab_size = 1000
    n_heads = 4
    n_layers = 3
    hidden_dim = 8


    ## ----- wandb settings ----- ##
    trial_name = f"lr{learning_rate}_bs{batch_size}_msl{max_seq_len}_vs{vocab_size}_nh{n_heads}_nl{n_layers}_hd{hidden_dim}"
    folder_name = f"results/{trial_name}/"
    os.makedirs(folder_name, exist_ok=True)

    wandb_config = {"batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_seq_len": max_seq_len,
                    "n_epochs": n_epochs,
                    "vocab_size": vocab_size,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "hidden_dim": hidden_dim}

    wandb.login(key="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")
    wandb.init(project="fineweb-transformer-test",
               dir=folder_name,
               name=str(trial_name),
               config=wandb_config,
               settings=wandb.Settings(symlink=False))

    model = Train_Transformer(n_heads, n_layers, hidden_dim, max_seq_len, vocab_size)
    if build_vocab:
        model.make_and_save_vocab()
    model.initialize_tokenizer()
    model.train(learning_rate, n_epochs, batch_size)

    wandb.finish()
    print("Done")
    sys.exit(0)