# If I'm not sure waht some function or class actually doing, I will write
# snippet codes to confirm my unstanding.

import torch
import torch.nn.functional as F


def learn_cross_entropy():
    IGNORE_ID = -1
    torch.manual_seed(123)

    input = torch.randn(4, 5, requires_grad=True)  # N x C
    target = torch.randint(5, (4,), dtype=torch.int64)  # N
    target[-1] = IGNORE_ID
    print("input:\n", input)
    print("target:\n", target)

    # PART 1: confirm F.cross_entropy() == F.log_softmax() + F.nll_loss()
    ce = F.cross_entropy(
        input, target, ignore_index=IGNORE_ID, reduction='elementwise_mean')
    print("### Using F.cross_entropy()")
    print("ce =", ce)
    ls = F.log_softmax(input, dim=1)
    nll = F.nll_loss(ls, target, ignore_index=IGNORE_ID,
                     reduction='elementwise_mean')
    print("### Using F.log_softmax() + F.nll_loss()")
    print("nll =", nll)
    print("### [CONFIRM] F.cross_entropy() == F.log_softmax() + F.nll_loss()\n")

    # PART 2: confirm log_softmax = log + softmax
    print("log_softmax():\n", ls)
    softmax = F.softmax(input, dim=1)
    log_softmax = torch.log(softmax)
    print("softmax():\n", softmax)
    print("log() + softmax():\n", log_softmax)
    print("### [CONFIRM] log_softmax() == log() + softmax()\n")

    # PART 3: confirm ignore_index works
    non_ignore_index = target[target != IGNORE_ID]
    print(non_ignore_index)
    print(log_softmax[target != IGNORE_ID])
    loss_each_sample = torch.stack([log_softmax[i][idx]
                                    for i, idx in enumerate(non_ignore_index)], dim=0)
    print(loss_each_sample)
    print(-1 * torch.mean(loss_each_sample))
    print("### [CONFIRM] ignore_index in F.cross_entropy() works\n")

    # PART 4: confirm cross_entropy()'s backward() works correctly when set ignore_index
    # nll = 1/N * -1 * sum(log(softmax(input, dim=1))[target])
    # d_nll / d_input = 1/N * (softmax(input, dim=1) - target)
    print("softmax:\n", softmax)
    print("non ignore softmax:")
    print(softmax[:len(non_ignore_index)])
    print(softmax[range(len(non_ignore_index)), non_ignore_index])
    print("target\n", target)
    grad = softmax
    grad[range(len(non_ignore_index)), non_ignore_index] -= 1
    grad /= len(non_ignore_index)
    grad[-1] = 0.0  # IGNORE_ID postition
    print("my gradient:\n", grad)
    ce.backward()
    print("pytorch gradient:\n", input.grad)
    print("### [CONFIRM] F.cross_entropy()'s backward() works correctly when "
          "set ignore_index")


if __name__ == "__main__":
    learn_cross_entropy()
