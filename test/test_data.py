import json

from data import AudioDataset
from data import AudioDataLoader


if __name__ == "__main__":
    train_json = "data/data.json"
    batch_size = 2
    max_length_in = 1000
    max_length_out = 1000
    num_batches = 10
    num_workers = 2
    batch_frames = 2000

    # test batch_frames
    train_dataset = AudioDataset(
        train_json, batch_size, max_length_in, max_length_out, num_batches,
        batch_frames=batch_frames)
    for i, minibatch in enumerate(train_dataset):
        print(i)
        print(minibatch)
    exit(0)

    # test
    train_dataset = AudioDataset(
        train_json, batch_size, max_length_in, max_length_out, num_batches)
    # NOTE: must set batch_size=1 here.
    train_loader = AudioDataLoader(
        train_dataset, batch_size=1, num_workers=num_workers, LFR_m=4, LFR_n=3)

    import torch
    #torch.set_printoptions(threshold=10000000)
    for i, (data) in enumerate(train_loader):
        inputs, inputs_lens, targets = data
        print(i)
        # print(inputs)
        print(inputs.size())
        print(inputs_lens)
        # print(targets)
        print("*"*20)
