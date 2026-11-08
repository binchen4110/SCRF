import hashlib
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F


def read_ids(path):
    pairs = [line.split() for line in path.read_text().splitlines() if line.strip()]
    mapping = {name: int(index) for name, index in pairs}
    if len(mapping) != len(pairs) or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError(f'IDs must be unique and contiguous: {path}')
    return mapping


def reciprocal(triples, relation_count):
    """加入逆关系查询，使头实体和尾实体预测使用同一训练与评估流程。"""
    inverse = triples[:, [2, 1, 0]].clone()
    inverse[:, 1] += relation_count
    return torch.cat((triples, inverse))


class GroupedTrainingQueries:
    """合并相同实体与关系的训练查询，将其全部训练答案作为多标签监督。
    验证集和测试集答案只参与评估过滤，不用于构造训练标签。"""
    def __init__(self, training_triples, num_entities):
        answers = defaultdict(set)
        for h, r, t in training_triples.tolist():
            answers[(h, r)].add(t)
        self.queries = torch.tensor(list(answers), dtype=torch.long).reshape(-1, 2)
        self.answers = [sorted(values) for values in answers.values()]
        self.num_entities = num_entities

    def targets(self, indices, device, smoothing=0.1):
        """为当前批次构造平滑后的多标签；保留多答案权重，不按每行答案数量归一化。"""
        labels = torch.zeros((len(indices), self.num_entities), device=device)
        rows, columns = [], []
        for row, index in enumerate(indices.tolist()):
            values = self.answers[index]
            rows.extend([row] * len(values))
            columns.extend(values)
        labels[rows, columns] = 1.

        return (1 - smoothing) * labels + 1 / self.num_entities


class DistillData:
    def __init__(self, root, dataset, features='original', normalize=True, filter_scope='all'):
        if features not in ('original', 'clip'):
            raise ValueError(f'Unknown feature family: {features}')
        if filter_scope not in ('train', 'all'):
            raise ValueError('filter_scope must be train or all')
        self.path = Path(root) / dataset
        self.entity2id = read_ids(self.path / 'entity2id.txt')
        self.relation2id = read_ids(self.path / 'relation2id.txt')
        self.num_entities, self.num_relations = len(self.entity2id), len(self.relation2id)
        self.splits = {}
        for split in ('train', 'valid', 'test'):
            rows = []
            for line in (self.path / f'{split}.txt').read_text().splitlines():
                if line.strip():
                    head, rel, tail = line.split()[:3]
                    rows.append((self.entity2id[head], self.relation2id[rel], self.entity2id[tail]))
            self.splits[split] = torch.tensor(rows, dtype=torch.long)
        suffix = '_clip' if features == 'clip' else ''
        self.features = {}
        self.manifest = {'dataset': dataset, 'features': features, 'normalized': normalize,
                         'filter_scope': filter_scope,
                         'entities': self.num_entities, 'relations': self.num_relations,
                         'splits': {k: len(v) for k, v in self.splits.items()}, 'files': {}}
        for kind in ('img', 'text'):
            file = self.path / f'{kind}_features{suffix}.pth'
            x = torch.load(file, map_location='cpu', weights_only=True)
            if not isinstance(x, torch.Tensor) or x.shape[0] != self.num_entities:
                raise ValueError(f'Expected an entity-indexed tensor: {file}')
            x = x.float().reshape(self.num_entities, -1)
            if not torch.isfinite(x).all():
                raise ValueError(f'Nonfinite features: {file}')
            self.features[kind] = F.normalize(x, dim=-1) if normalize else x
            self.manifest[f'{kind}_shape'] = list(x.shape)
        paths = [self.path / f'{s}.txt' for s in ('entity2id', 'relation2id', 'train', 'valid', 'test')]
        paths += [self.path / f'{k}_features{suffix}.pth' for k in ('img', 'text')]
        for file in paths:
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            self.manifest['files'][file.name] = digest.hexdigest()
        self.train = reciprocal(self.splits['train'], self.num_relations)
        self.known_train = defaultdict(set)
        self.known_all = defaultdict(set)
        for split, triples in self.splits.items():
            for h, r, t in reciprocal(triples, self.num_relations).tolist():
                self.known_all[(h, r)].add(t)
                if split == 'train':
                    self.known_train[(h, r)].add(t)
        self.known = self.known_train if filter_scope == 'train' else self.known_all


def filtered_ranks(scores, queries, known):
    """保留当前目标答案，过滤其他已知正确实体；并列得分使用平均排名。"""
    targets = queries[:, 2]
    row = torch.arange(len(queries), device=scores.device)
    target_scores = scores[row, targets].clone()
    filtered = scores.clone()
    rows, cols = [], []
    for index, (head, relation, _) in enumerate(queries.tolist()):
        tails = known.get((head, relation), ())
        rows.extend([index] * len(tails))
        cols.extend(tails)
    if rows:
        filtered[rows, cols] = -torch.inf
    filtered[row, targets] = target_scores
    better = (filtered > target_scores[:, None]).sum(-1)
    tied = (filtered == target_scores[:, None]).sum(-1)
    return 1.0 + better.double() + 0.5 * (tied.double() - 1)


def rank_metrics(ranks):
    ranks = torch.as_tensor(ranks, dtype=torch.float64)
    return {'MRR': (1 / ranks).mean().item(), 'MR': ranks.mean().item(),
            **{f'Hits@{k}': (ranks <= k).double().mean().item() for k in (1, 3, 10)}}


def probability_ensemble(logits, temperature=1., weights=None):
    """各分支分别经过温度校准后融合概率，权重顺序为结构、图像、文本、融合。
    消融时对剩余分支的权重重新归一化，不改变训练损失。"""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('prediction temperature must be finite and positive')
    if weights is None:
        return torch.stack([(x/temperature).softmax(-1) for x in logits.values()]).mean(0)
    if len(weights)!=4 or any(not math.isfinite(w) or w<0 for w in weights) or not math.isclose(sum(weights),1.,abs_tol=1e-8):
        raise ValueError('S,V,T,MM weights must be finite, nonnegative and sum to one')
    by_name=dict(zip(('s','i','t','mm'),weights))
    active=[by_name[name] for name in logits]
    total=sum(active)
    if total<=0:raise ValueError('Active branches must have positive total weight')
    probabilities=torch.stack([(x/temperature).softmax(-1) for x in logits.values()])
    if len(set(active))==1:
        return probabilities.mean(0)
    factors=probabilities.new_tensor(active)/total
    return (probabilities*factors[:,None,None]).sum(0)


@torch.no_grad()
def evaluate(model, data, split, device, batch_size=256, limit=0, filter_scope=None,
             prediction_temperature=1., ensemble_weights=None):
    """复用候选表示完成双向预测，汇总各分支及概率集成的过滤排名指标。"""
    if prediction_temperature <= 0:
        raise ValueError('prediction_temperature must be positive')
    model.eval()
    triples = data.splits[split]
    if limit:
        triples = triples[:limit]
    queries = reciprocal(triples, data.num_relations).to(device)
    bank = model.candidate_bank()
    ranks = defaultdict(list)
    if filter_scope not in (None, 'train', 'all'):
        raise ValueError('filter_scope must be train or all')
    known = data.known if filter_scope is None else getattr(data, 'known_' + filter_scope)
    for batch in queries.split(batch_size):
        logits = model(batch, bank)

        logits['mean'] = probability_ensemble(logits,prediction_temperature,ensemble_weights)
        for name, scores in logits.items():
            if not torch.isfinite(scores).all():
                raise FloatingPointError(f'Nonfinite scores in {split}/{name}')
            ranks[name].extend(filtered_ranks(scores, batch, known).cpu().tolist())
    return {name: rank_metrics(values) for name, values in ranks.items()}
