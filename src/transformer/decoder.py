import torch
import torch.nn as nn

from attention import MultiHeadAttention
from module import PositionalEncoding, PositionwiseFeedForward
from utils import IGNORE_ID, pad_list, get_subsequent_mask, get_attn_key_pad_mask, get_non_pad_mask


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, sos_id, eos_id,
            n_tgt_vocab, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            pe_maxlen=5000):
        super(Decoder, self).__init__()
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

    def forward(self, padded_input, encoder_padded_outputs,
                encoder_input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H

        Returns:
        """
        # START -- Get Input and Output
        # from espnet/Decoder.forward()
        # TODO: need to make more smart way
        ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, IGNORE_ID)
        # print("ys_in_pad", ys_in_pad.size())
        assert ys_in_pad.size() == ys_out_pad.size()
        batch_size = ys_in_pad.size(0)
        output_length = ys_in_pad.size(1)
        # max_length = ys_in_pad.size(1) - 1  # TODO: should minus 1(sos)?
        # END -- Get Input and Output

        # -- Prepare masks
        # TODO: encapsulate mask into functions
        # padding position is set to 0, unsqueeze(-1) for broadcast
        assert ys_in_pad.dim() == 2
        non_pad_mask = ys_in_pad.ne(self.eos_id).type(torch.float).unsqueeze(-1)
        # padding position is set to 1
        # mask position will be filled in -np.inf in ScaledDotProductAttention
        slf_attn_mask_subseq = get_subsequent_mask(ys_in_pad)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=ys_in_pad,
                                                     seq_q=ys_in_pad,
                                                     pad_idx=self.eos_id)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        # padding position is set to 1
        dec_enc_attn_mask = get_non_pad_mask(encoder_padded_outputs, # NxTixD
                                             encoder_input_lengths) # NxTix1, padding position is set to 0
        dec_enc_attn_mask = dec_enc_attn_mask.clone().squeeze(-1).lt(1) # padding position is set to 1
        dec_enc_attn_mask = dec_enc_attn_mask.unsqueeze(1).expand(-1, output_length, -1)

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.dropout(self.tgt_word_emb(ys_in_pad) +
                                  self.positional_encoding(ys_in_pad))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        # before softmax
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        # return
        pred, gold = seq_logit, ys_out_pad

        if return_attns:
            return pred, gold, dec_slf_attn_list, dec_enc_attn_list
        return pred, gold


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
