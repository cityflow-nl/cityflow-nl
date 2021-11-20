import unittest

import torch
from third_party.lr import TriStageLRScheduler


class TestWarmUpLr(unittest.TestCase):
    def test_warmup_lr(self):
        optimizer = torch.optim.AdamW(params=[torch.tensor(1)],
                                      lr=0.001)
        warmup_scheduler = TriStageLRScheduler(optimizer,
                                               0.001,
                                               0.01,
                                               0.0001,
                                               0.1,
                                               2000,
                                               2000,
                                               2000)
        for epoch in range(5):
            for step in range(2000):
                optimizer.step()
                warmup_scheduler.step()
                if step % 200 == 0:
                    print(warmup_scheduler.get_lr())


if __name__ == '__main__':
    unittest.main()
