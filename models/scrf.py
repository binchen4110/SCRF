import torch
from torch import nn
from torch.nn import functional as F

from layers.tucker import TuckERLayer


def ffn(input_dim, dim):
    """将各模态映射到统一维度，两层线性映射之间使用非线性激活。"""
    return nn.Sequential(nn.Linear(input_dim, dim), nn.ReLU(), nn.Linear(dim, dim))


class UnitNorm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)


class StructureAwareFusion(nn.Module):
    """以结构表示控制图像和文本的贡献，再通过拼接映射获得融合表示。"""
    def __init__(self, dim, gated=True, normalization='l2', enabled=True):
        super().__init__()
        self.gated = gated
        self.enabled = enabled
        self.image_gate = nn.Linear(2 * dim, 1, bias=False)
        self.text_gate = nn.Linear(2 * dim, 1, bias=False)
        self.project = ffn(3 * dim, dim)
        if normalization not in ('l2', 'layernorm'):
            raise ValueError('fusion normalization must be l2 or layernorm')
        self.norm = UnitNorm() if normalization == 'l2' else nn.LayerNorm(dim)

    def modulate(self, structure, image, text):
        """两个门控独立输出标量，只缩放对应模态，不改变特征方向。"""
        if self.gated:
            image = image * torch.sigmoid(self.image_gate(torch.cat((structure, image), -1)))
            text = text * torch.sigmoid(self.text_gate(torch.cat((structure, text), -1)))
        return structure, image, text

    def combine(self, structure, image, text):
        """拼接结构与门控后的模态表示，经前馈网络和归一化得到融合输出。"""
        if not self.enabled:
            return (structure + image + text) / 3

        return self.norm(self.project(torch.cat((structure, image, text), -1)))

    def forward(self, structure, image, text):
        return self.combine(*self.modulate(structure, image, text))


def topk_distillation(logits, top_k=500, temperature=1.0, scale_t2=False):
    """结构教师选取高分候选，图像与文本在同一候选集合内学习教师分布。
    教师得分停止梯度；三个分布在候选集合内重新归一化。"""
    if temperature <= 0 or top_k < 1:
        raise ValueError('temperature and top_k must be positive')
    if 's' not in logits:
        return next(iter(logits.values())).new_zeros(())
    teacher = logits['s'].detach()
    indices = teacher.topk(min(top_k, teacher.shape[-1]), dim=-1).indices
    log_teacher = F.log_softmax(teacher.gather(1, indices) / temperature, dim=-1)
    loss = teacher.new_zeros(())
    for name in ('i', 't'):
        if name in logits:
            student = F.log_softmax(logits[name].gather(1, indices) / temperature, dim=-1)
            loss = loss + F.kl_div(student, log_teacher, reduction='batchmean', log_target=True)
    return loss * (temperature ** 2 if scale_t2 else 1.0)


class SCRF(nn.Module):
    def __init__(self, num_entities, num_relations, image_features, text_features,
                 dim=200, ablation='none', decoder_dropout=(0.3, 0.4, 0.5),
                 student_input='gated', candidate_input='raw', decoder_relation_mode='shared',
                 fusion_norm='l2', decoder_batchnorm=True):
        super().__init__()
        if ablation not in ('none', 'rwf', 'saf', 'sgr', 's', 'i', 't'):
            raise ValueError(f'Unknown ablation: {ablation}')
        self.ablation = ablation
        if student_input not in ('raw', 'gated') or candidate_input not in ('raw', 'gated'):
            raise ValueError('student_input/candidate_input must be raw or gated')
        self.student_input = student_input
        self.candidate_input = candidate_input
        if decoder_relation_mode not in ('shared', 'separate'):
            raise ValueError('decoder_relation_mode must be shared or separate')
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(2 * num_relations, dim)

        self.register_buffer('image_features', image_features, persistent=False)
        self.register_buffer('text_features', text_features, persistent=False)
        self.encoders = nn.ModuleDict({
            's': ffn(dim, dim),
            'i': ffn(image_features.shape[1], dim),
            't': ffn(text_features.shape[1], dim),
        })
        self.fusion = StructureAwareFusion(dim, gated=ablation != 'saf', normalization=fusion_norm,
                                          enabled=ablation != 'saf')
        self.active = tuple(m for m in ('s', 'i', 't', 'mm') if m != ablation)
        self.decoders = nn.ModuleDict({m: TuckERLayer(dim, dim) for m in self.active})
        for decoder in self.decoders.values():
            decoder.input_drop.p, decoder.hidden_drop.p, decoder.out_drop.p = decoder_dropout
            if not decoder_batchnorm:
                decoder.bn0 = nn.Identity()
                decoder.bn1 = nn.Identity()
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_normal_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)


        self.decoder_relations = nn.ModuleDict({
            m: nn.Embedding.from_pretrained(self.relation.weight.detach().clone(), freeze=False)
            for m in self.active
        } if decoder_relation_mode == 'separate' else {})

    def candidate_bank(self):
        """所有实体共用模态编码器，构建不依赖当前查询关系的候选表示。"""
        raw = {'s': self.entity.weight, 'i': self.image_features, 't': self.text_features}
        bank = {m: self.encoders[m](x) for m, x in raw.items()}
        if self.ablation in ('s', 'i', 't'):
            bank[self.ablation] = torch.zeros_like(bank[self.ablation])
        bank['mm'] = self.fusion(bank['s'], bank['i'], bank['t'])


        if self.candidate_input == 'gated':
            _, bank['i_candidate'], bank['t_candidate'] = self.fusion.modulate(
                bank['s'], bank['i'], bank['t'])
        return bank

    def forward(self, queries, bank=None):
        """查询表示先加入关系，再进行门控融合；四个解码分支分别对全部候选实体评分。"""
        if bank is None:
            bank = self.candidate_bank()
        relation = self.relation(queries[:, 1])
        h = {m: bank[m][queries[:, 0]] for m in ('s', 'i', 't')}
        if self.ablation != 'rwf':
            h = {m: x + relation if m != self.ablation else x for m, x in h.items()}
        gated = self.fusion.modulate(h['s'], h['i'], h['t'])
        h['mm'] = self.fusion.combine(*gated)
        if self.student_input == 'gated':
            h['i'], h['t'] = gated[1], gated[2]
        return {m: self.decoders[m](h[m], self.decoder_relations[m](queries[:, 1])
                                   if m in self.decoder_relations else relation)
                   @ bank.get(m + '_candidate', bank[m]).T
                for m in self.active}

    def objective(self, logits, targets, epoch, warmup_epochs=50,
                  kd_weight=1e-6, top_k=500, temperature=1.0, scale_t2=False,
                  supervision='triple_ce'):
        """预热阶段联合训练四个预测分支，预热结束后加入结构指导的模态蒸馏。"""
        if supervision == 'triple_ce':
            kgc = 2 * sum(F.cross_entropy(x, targets) for x in logits.values())
        elif supervision == 'query_ce':
            kgc = 2 * sum(-(targets * F.log_softmax(x, dim=-1)).sum(-1).mean()
                          for x in logits.values())
        elif supervision == 'query_bce':
            kgc = 2 * targets.shape[-1] * sum(
                F.binary_cross_entropy(x.softmax(-1), targets) for x in logits.values())
        else:
            raise ValueError(f'Unknown supervision: {supervision}')
        kd = kgc.new_zeros(())
        if epoch > warmup_epochs and self.ablation not in ('sgr', 's') and kd_weight:
            kd = 2 * topk_distillation(logits, top_k, temperature, scale_t2)
        return kgc + kd_weight * kd, {'kgc': kgc.detach(), 'kd': kd.detach()}
