from dataclasses import dataclass, asdict
import math


@dataclass
class EarlyStopping:
    patience: int = 3
    best: float = -math.inf
    bad_evaluations: int = 0
    best_epoch: int = 0

    def update(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError('Nonfinite early-stopping metric')
        if score > self.best:
            self.best = score
            self.bad_evaluations = 0
            self.best_epoch = epoch
        else:
            self.bad_evaluations += 1
        return self.patience > 0 and self.bad_evaluations >= self.patience

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls, state):
        return cls(**state)
