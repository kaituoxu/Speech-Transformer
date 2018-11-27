import argparse

import torch

from encoder import Encoder
from decoder import Decoder
from transformer import Transformer


if __name__ == "__main__":
    D = 3
    beam_size = 5
    nbest = 5
    defaults = dict(beam_size=beam_size,
                    nbest=nbest,
                    decode_max_len=0,
                    d_input = D,
                    LFR_m = 1,
                    n_layers_enc = 2,
                    n_head = 2,
                    d_k = 6,
                    d_v = 6,
                    d_model = 12,
                    d_inner = 8,
                    dropout=0.1,
                    pe_maxlen=100,
                    d_word_vec=12,
                    n_layers_dec = 2,
                    tgt_emb_prj_weight_sharing=1)
    args = argparse.Namespace(**defaults)
    char_list = ["a", "b", "c", "d", "e", "f", "g", "h", "<sos>", "<eos>"]
    sos_id, eos_id = 8, 9
    vocab_size = len(char_list)
    # model
    encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout, pe_maxlen=args.pe_maxlen)
    decoder = Decoder(sos_id, eos_id, vocab_size,
                      args.d_word_vec, args.n_layers_dec, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout,
                      tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                      pe_maxlen=args.pe_maxlen)
    model = Transformer(encoder, decoder)

    for i in range(3):
        print("\n***** Utt", i+1)
        Ti = i + 20
        input = torch.randn(Ti, D)
        length = torch.tensor([Ti], dtype=torch.int)
        nbest_hyps = model.recognize(input, length, char_list, args)

    file_path = "./temp.pth"
    optimizer = torch.optim.Adam(model.parameters())
    torch.save(model.serialize(model, optimizer, 1, LFR_m=1, LFR_n=1), file_path)
    model, LFR_m, LFR_n = Transformer.load_model(file_path)
    print(model)

    import os
    os.remove(file_path)
