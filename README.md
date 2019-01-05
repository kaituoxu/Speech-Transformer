# End-to-End Automatic Speech Recognition with Transformer
This is a PyTorch implementation of End-to-End ASR with [Transformer](https://arxiv.org/abs/1706.03762) model.

## Install
- Python3 (Recommend Anaconda)
- PyTorch 0.4.1+
- [Kaldi](https://github.com/kaldi-asr/kaldi) (Just for feature extraction)
- `pip install -r requirements.txt`
- `cd tools; make KALDI=/path/to/kaldi`
- If you want to run `egs/aishell/run.sh`, download [aishell](http://www.openslr.org/33/) dataset for free.

## Usage
1. `$ cd egs/aishell` and modify aishell data path to your path in `run.sh`.
2. `$ bash run.sh`, that's all!

You can change hyper-parameter by `$ bash run.sh --parameter_name parameter_value`, egs, `$ bash run.sh --stage 3`. See parameter name in `egs/aishell/run.sh` before `. utils/parse_options.sh`.
### More detail
```bash
$ cd egs/aishell/
$ . ./path.sh
```
Train
```bash
$ train.py -h
```
Decode
```bash
$ recognize.py -h
```
### Workflow
Workflow of `egs/aishell/run.sh`:
- Stage 0: Data Preparation
- Stage 1: Feature Generation
- Stage 2: Dictionary and Json Data Preparation
- Stage 3: Network Training
- Stage 4: Decoding
### Visualize loss
If you want to visualize your loss, you can use [visdom](https://github.com/facebookresearch/visdom) to do that:
- Open a new terminal in your remote server (recommend tmux) and run `$ visdom`.
- Open a new terminal and run `$ bash run.sh --visdom 1 --visdom_id "<any-string>"` or `$ train.py ... --visdom 1 --vidsdom_id "<any-string>"`.
- Open your browser and type `<your-remote-server-ip>:8097`, egs, `127.0.0.1:8097`.
- In visdom website, chose `<any-string>` in `Environment` to see your loss.

## Results
| Model | CER | Config |
| :---: | :-: | :----: |
| LSTMP | 9.85| 4x(1024-512) |
| Listen, Attend and Spell | 13.2 | See my repo ListenAttendSpell's egs/aishell/run.sh |
| SpeechTransformer | 12.8 | See egs/aishell/run.sh |

## Related papers
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, 2017, pp. 5998–6008.
- Linhao Dong, Shuang Xu,and Bo Xu,“Speech-transformer:A no-recurrence sequence-to-sequence model for speech recognition,” in International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018, pp. 5884–5888.
- Shiyu Zhou, Linhao Dong, Shuang Xu, and Bo Xu, “Syllable-based sequence-to-sequence speech recognition with the transformer in mandarin chinese,” in Proc. Interspeech 2018, 2018, pp. 791–795.
- Shiyu Zhou, Linhao Dong, Shuang Xu, and Bo Xu, “A comparison of modeling units in sequence-to-sequence speech recognition with the transformer on mandarin chinese,” arXiv preprint arXiv:1805.06239, 2018.
