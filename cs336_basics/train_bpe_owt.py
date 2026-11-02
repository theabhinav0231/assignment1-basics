import os
from cs336_basics.train_bpe_tinystories import train_and_save

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    dataset_path = "data/owt_train.txt"
    vocab_size = 32000  # 32K vocab size specified in the assignment
    output_prefix = "data/owt"

    train_and_save(dataset_path, vocab_size, output_prefix)