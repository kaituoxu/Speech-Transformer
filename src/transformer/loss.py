import torch.nn.functional as F

from utils import IGNORE_ID


def cal_performance(pred, gold, smoothing=False):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """

    pred = pred.view(-1, pred.size(2))
    gold = gold.view(-1)

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def cal_loss(pred, gold, smoothing=False):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """

    if smoothing:
        pass
        # TODO: add label smoothing
        # eps = 0.1
        # n_class = pred.size(1)

        # one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        # one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        # log_prb = F.log_softmax(pred, dim=1)

        # non_pad_mask = gold.ne(Constants.PAD)
        # loss = -(one_hot * log_prb).sum(dim=1)
        # loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=IGNORE_ID,
                               reduction='elementwise_mean')

    return loss
