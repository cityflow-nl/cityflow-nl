# MIT License
#
# Copyright (c) 2021 Soohwan Kim
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

from torch.optim import Optimizer


class LearningRateScheduler(object):
    r"""
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class TriStageLRScheduler(LearningRateScheduler):
    r"""
    Tri-Stage Learning Rate Scheduler. Implement the learning rate scheduler in "SpecAugment"

    Args:
        optimizer (Optimizer): Optimizer.
        init_lr (float): Initial learning rate.
        peak_lr (float): Maximum learning rate.
        final_lr (float): Final learning rate.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        hold_steps (int): Hold the learning rate for the N updates.
        decay_steps (int): Decay the learning rate linearly for the first N updates.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            init_lr: float,
            peak_lr: float,
            final_lr: float,
            warmup_steps: int,
            hold_steps: int,
            decay_steps: int,
            final_lr_scale: float = 0.05,
            global_step: int = 0,
    ):
        super(TriStageLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps

        self.warmup_rate = (self.peak_lr - self.init_lr) / \
            self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_steps = global_step

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        offset = self.warmup_steps

        if self.update_steps < offset + self.hold_steps:
            return 1, self.update_steps - offset

        offset += self.hold_steps

        if self.update_steps <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_steps - offset

        offset += self.decay_steps

        return 3, self.update_steps - offset

    def step(self):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * \
                math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_steps += 1

        return self.lr
