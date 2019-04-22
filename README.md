# Speech Transformer: End-to-End ASR with Transformer
A PyTorch implementation of Speech Transformer [1][2][3], an end-to-end automatic speech recognition with [Transformer](https://arxiv.org/abs/1706.03762) [4] network, which directly converts acoustic features to character sequence using a single nueral network.

## Install
- Python3 (recommend Anaconda)
- PyTorch 0.4.1+
- [Kaldi](https://github.com/kaldi-asr/kaldi) (just for feature extraction)
- `pip install -r requirements.txt`
- `cd tools; make KALDI=/path/to/kaldi`
- If you want to run `egs/aishell/run.sh`, download [aishell](http://www.openslr.org/33/) dataset for free.

## Usage
### Quick start
```bash
$ cd egs/aishell
# Modify aishell data path to your path in the begining of run.sh 
$ bash run.sh
```
That's all!

You can change parameter by `$ bash run.sh --parameter_name parameter_value`, egs, `$ bash run.sh --stage 3`. See parameter name in `egs/aishell/run.sh` before `. utils/parse_options.sh`.
### Workflow
Workflow of `egs/aishell/run.sh`:
- Stage 0: Data Preparation
- Stage 1: Feature Generation
- Stage 2: Dictionary and Json Data Preparation
- Stage 3: Network Training
- Stage 4: Language Model Training
- Stage 5: Decoding
### More detail
`egs/aishell/run.sh` provide example usage.
```bash
# Set PATH and PYTHONPATH
$ cd egs/aishell/; . ./path.sh
# Train
$ train.py -h
# Decode
$ recognize.py -h
```
#### How to visualize loss?
If you want to visualize your loss, you can use [visdom](https://github.com/facebookresearch/visdom) to do that:
1. Open a new terminal in your remote server (recommend tmux) and run `$ visdom`.
2. Open a new terminal and run `$ bash run.sh --visdom 1 --visdom_id "<any-string>"` or `$ train.py ... --visdom 1 --vidsdom_id "<any-string>"`.
3. Open your browser and type `<your-remote-server-ip>:8097`, egs, `127.0.0.1:8097`.
4. In visdom website, chose `<any-string>` in `Environment` to see your loss.
![loss](egs/aishell/figures/train-k0.2-bf15000-shuffle-ls0.1.png)
#### How to resume training?
```bash
$ bash run.sh --continue_from <model-path>
```
#### How to solve out of memory?
When happened in training, try to reduce `batch_size`. `$ bash run.sh --batch_size <lower-value>`.

## Results
| Model | CER | Config |
| :---: | :-: | :----: |
| LSTMP | 9.85| 4x(1024-512). See [kaldi-ktnet1](https://github.com/kaituoxu/kaldi-ktnet1/blob/ktnet1/egs/aishell/s5/local/nnet1/run_4lstm.sh)|
| Listen, Attend and Spell | 13.2 | See [Listen-Attend-Spell](https://github.com/kaituoxu/Listen-Attend-Spell)'s egs/aishell/run.sh |
| SpeechTransformer | 12.8 | See egs/aishell/run.sh |
| SpeechTransformer + 3-gram | 12.1 | insert lm_score in recognize |
## Reference
- [1] Linhao Dong, Shuang Xu,and Bo Xu. “Speech-transformer:A no-recurrence sequence-to-sequence model for speech recognition” in ICASSP 2018
- [2] Shiyu Zhou, Linhao Dong, et al. “Syllable-based sequence-to-sequence speech recognition with the transformer in mandarin chinese” in Interspeech 2018
- [3] Shiyu Zhou, Linhao Dong, et al. “A comparison of modeling units in sequence-to-sequence speech recognition with the transformer on mandarin chinese” arXiv preprint arXiv:1805.06239
- [4] Ashish Vaswani, Noam Shazeer, et al. “Attention is all you need” in NIPS 2017
