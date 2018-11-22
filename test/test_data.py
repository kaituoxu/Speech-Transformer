import json

from data.data import AudioDataset
from data.data import AudioDataLoader


if __name__ == "__main__":
    train_json = "data/data.json"
    batch_size = 2
    max_length_in = 1000
    max_length_out = 1000
    num_batches = 10
    num_workers = 2

    train_dataset = AudioDataset(
        train_json, batch_size, max_length_in, max_length_out, num_batches)
    # NOTE: must set batch_size=1 here.
    train_loader = AudioDataLoader(
        train_dataset, batch_size=1, num_workers=num_workers)

    for i, (data) in enumerate(train_loader):
        inputs, inputs_lens, targets = data
        print(i)
        # print(inputs)
        print(inputs_lens)
        # print(targets)
        print("*"*20)
