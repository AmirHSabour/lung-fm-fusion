"""Tile-level multiple-instance fusion (v04, decisions/v04.md): attention pooling over cached tiles.

Each patient's row carries an `ID` column, its row in the tile cache (src/embeddings/tile_cache.py).
Tiles are standardised with the training slides' tile statistics, projected to d numbers, pooled
by gated attention (Ilse et al., 2018) and mapped to class scores; age and sex add a log-prior
from their tokens (v03) and, in M3, modulate the tile features before attention (FiLM).

- M1 `concat`: one branch on [u_t ; v_t] (per-tile early fusion of the models).
- M2 `branches`: one attention branch per model; class scores averaged.
- M3 `film`: as M1, with tile features modulated by age and sex before attention.

v06 (decisions/v06.md): `aggregator` selects the tile→slide aggregator. `attention` (default) is
the network above. `comp` and `wcomp` (`CompositionNet`) give every tile a class distribution
and the slide a mixture of them: the uniform mean over tiles (an estimate of class area shares)
or a gated-attention-weighted mean. These take no age and sex (they are used as experts).

v07 (decisions/v07.md): `clam_sb` and `clam_mb` (`ClamNet`, Lu et al., 2021): gated attention with
one branch (SB) or one branch per class (MB) and an instance clustering loss on the most and least
attended tiles, added to the slide loss with weight 0.3. No age and sex (experts).

v10 (decisions/v10.md): `xattn` (`CrossAttnNet`): queries attend the tiles in one pre-norm residual
cross-attention block (Perceiver, Jaegle et al., 2021). One learned query; with age and sex, their
tokens (src/fusion/metadata.py) are added as queries, so they choose which tiles are read.

v11 (decisions/v11.md): `aux_groups` adds a grade-group head on the pooled slide feature during
training (In situ; Lepidic; Acinar, Papillary; Micropapillary, Solid, Cribriform), weight 0.5.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn

from src.embeddings.tile_cache import load
from src.fusion.blocks import ID, META, Blocks
from src.fusion.metadata import MetaTokenizer, PiecewiseLinear

MIL_KINDS = ("concat", "branches", "film")
AGGREGATORS = ("attention", "comp", "wcomp", "clam_sb", "clam_mb", "xattn")
TAKES_META = ("attention", "xattn")
GRADE_GROUPS = (0, 1, 2, 2, 3, 3, 3)      # group of each class index in src.data.cohort.CLASSES (v11)
CLAM = ("clam_sb", "clam_mb")


class GatedAttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.A, self.B, self.w = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:          # h: (n, T, d) -> (n, d)
        a = torch.softmax(self.w(torch.tanh(self.A(h)) * torch.sigmoid(self.B(h))).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * h).sum(dim=1)


class CompositionNet(nn.Module):
    """Slide log-distribution log p = log Σ_t a_t softmax(V h_t), with a_t = 1/T (`comp`) or
    gated attention (`wcomp`). Returns log p, which are valid logits (log-softmax leaves them unchanged)."""

    def __init__(self, width: int, d: int, dropout: float, K: int, weighted: bool):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(d, K)
        self.weighted = weighted
        if weighted:
            self.A, self.B, self.w = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 1)

    def forward(self, tiles: list[torch.Tensor], meta: dict | None = None) -> torch.Tensor:
        h = self.proj(torch.cat(tiles, dim=-1))                                  # (n, T, d)
        logq = torch.log_softmax(self.head(h), dim=-1)                          # (n, T, K)
        if self.weighted:
            loga = torch.log_softmax(self.w(torch.tanh(self.A(h)) * torch.sigmoid(self.B(h))), dim=1)   # (n, T, 1)
        else:
            loga = torch.full_like(logq[..., :1], -float(np.log(h.shape[1])))
        return torch.logsumexp(logq + loga, dim=1)                              # (n, K)


class ClamNet(nn.Module):
    """CLAM (Lu et al., 2021). `forward` returns slide scores; `instance_loss` the clustering loss.

    Branches B = 1 (SB: one attention map, linear classifier on the pooled feature) or K (MB: class k
    pools with its own map and scores with its own vector). Instance loss (subtyping form): for a
    slide of class y and each class k, the k_sample most attended tiles of branch k (branch 0 for SB)
    are positives for classifier k if k = y (with the k_sample least attended as negatives), and
    negatives otherwise. Cross-entropy averaged per (slide, class), then over classes and slides.
    """

    def __init__(self, width: int, d: int, dropout: float, K: int, multi_branch: bool, k_sample: int = 8):
        super().__init__()
        self.K, self.B, self.k_sample = K, (K if multi_branch else 1), k_sample
        self.proj = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Dropout(dropout))
        self.A, self.G, self.w = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, self.B)
        if multi_branch:
            self.v = nn.Parameter(torch.empty(K, d))
            self.c = nn.Parameter(torch.zeros(K))
            nn.init.kaiming_uniform_(self.v, a=5 ** 0.5)
        else:
            self.head = nn.Linear(d, K)
        self.inst = nn.Parameter(torch.empty(K, d, 2))          # one Linear(d -> 2) per class
        self.inst_b = nn.Parameter(torch.zeros(K, 2))
        nn.init.normal_(self.inst, std=d ** -0.5)

    def attend(self, tiles: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(torch.cat(tiles, dim=-1))                                       # (n, T, d)
        a = torch.softmax(self.w(torch.tanh(self.A(h)) * torch.sigmoid(self.G(h))), dim=1)   # (n, T, B)
        return h, a

    def scores(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        M = torch.einsum("ntb,ntd->nbd", a, h)                                        # (n, B, d)
        if self.B == 1:
            return self.head(M[:, 0])
        return (M * self.v).sum(dim=-1) + self.c                                      # (n, K)

    def forward(self, tiles: list[torch.Tensor], meta: dict | None = None) -> torch.Tensor:
        return self.scores(*self.attend(tiles))

    def instance_loss(self, h: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n, T, d = h.shape
        k = min(self.k_sample, T // 2)
        att = a.expand(n, T, self.K) if self.B == 1 else a                            # (n, T, K)
        top = att.topk(k, dim=1).indices                                              # (n, k, K)
        bot = (-att).topk(k, dim=1).indices
        gather = lambda idx: torch.gather(h.unsqueeze(2).expand(n, T, self.K, d), 1,
                                          idx.unsqueeze(-1).expand(n, k, self.K, d))   # (n, k, K, d)
        logit = lambda z: torch.einsum("njkd,kdc->njkc", z, self.inst) + self.inst_b   # (n, k, K, 2)
        own = (torch.arange(self.K, device=h.device)[None, :] == y[:, None])           # (n, K)
        ce = nn.functional.cross_entropy
        lt, lb = logit(gather(top)), logit(gather(bot))
        t_top = own[:, None, :].expand(n, k, self.K).long()
        loss_top = ce(lt.reshape(-1, 2), t_top.reshape(-1), reduction="none").view(n, k, self.K)
        loss_bot = ce(lb.reshape(-1, 2), torch.zeros(n * k * self.K, dtype=torch.long, device=h.device),
                      reduction="none").view(n, k, self.K)
        # own class: mean over 2k tiles (top positive, bottom negative); other classes: mean over top k
        per = torch.where(own, (loss_top.mean(1) + loss_bot.mean(1)) / 2, loss_top.mean(1))   # (n, K)
        return per.mean()


class CrossAttnNet(nn.Module):
    """O = Q + MHA(LN(Q), LN(h), LN(h)); z = V · mean_q LN(O)_q. Q = [learned query ; age-and-sex tokens].
    Attention written out (einsum + softmax) so it is deterministic on the GPU."""

    def __init__(self, width: int, d: int, dropout: float, K: int, encoding: str | None = None, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.proj = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Dropout(dropout))
        self.query = nn.Parameter(torch.randn(1, 1, d) / d ** 0.5)
        self.tok = MetaTokenizer(encoding, d, dropout) if encoding else None
        self.ln_q, self.ln_h, self.ln_o = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wq, self.Wk, self.Wv, self.Wo = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, K)

    def queries(self, n: int, meta: dict | None) -> torch.Tensor:
        Q = self.query.expand(n, 1, -1)
        return torch.cat([Q, self.tok(**meta)], dim=1) if self.tok is not None else Q

    def forward(self, tiles: list[torch.Tensor], meta: dict | None = None) -> torch.Tensor:
        h = self.ln_h(self.proj(torch.cat(tiles, dim=-1)))                       # (n, T, d)
        Q = self.queries(h.shape[0], meta)                                       # (n, q, d)
        split = lambda x: x.view(x.shape[0], x.shape[1], self.heads, -1).transpose(1, 2)   # (n, H, ·, dh)
        q, k, v = split(self.Wq(self.ln_q(Q))), split(self.Wk(h)), split(self.Wv(h))
        a = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)  # (n, H, q, T)
        o = (self.drop(a) @ v).transpose(1, 2).reshape(Q.shape)                  # (n, q, d)
        O = Q + self.Wo(o)
        return self.head(self.ln_o(O).mean(dim=1))


class MILNet(nn.Module):
    def __init__(self, widths: list[int], kind: str, encoding: str | None, d: int, dropout: float, K: int):
        super().__init__()
        self.kind = kind
        n_in = [sum(widths)] if kind in ("concat", "film") else widths
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(p, d), nn.GELU(), nn.Dropout(dropout)) for p in n_in])
        self.pool = nn.ModuleList([GatedAttentionPool(d) for _ in n_in])
        self.heads = nn.ModuleList([nn.Linear(d, K) for _ in n_in])
        self.tok = MetaTokenizer(encoding, d, dropout) if encoding else None
        if self.tok is not None:
            self.meta_head = nn.Linear(d, K)
            if kind == "film":
                self.film = nn.Linear(d, 2 * d)
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
        elif kind == "film":
            raise ValueError("FiLM needs age and sex")

    def forward(self, tiles: list[torch.Tensor], meta: dict | None) -> torch.Tensor:
        xs = [torch.cat(tiles, dim=-1)] if self.kind in ("concat", "film") else tiles
        m = self.tok(**meta).mean(dim=1) if self.tok is not None else None          # (n, d)
        z = 0
        for x, proj, pool, head in zip(xs, self.proj, self.pool, self.heads):
            h = proj(x)                                                             # (n, T, d)
            if self.kind == "film":
                gamma, beta = self.film(m).chunk(2, dim=-1)
                h = (1 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
            z = z + head(pool(h))
        z = z / len(xs)
        return z + self.meta_head(m) if m is not None else z


class MILFusion(BaseEstimator, ClassifierMixin):
    def __init__(self, blocks: Blocks, kind: str = "concat", encoding: str = "ft", cache_dir: str = "artifacts/features/tiles",
                 d: int = 32, weight_decay: float = 0.01, epochs: int = 100, lr: float = 1e-3, dropout: float = 0.25,
                 tiles_per_step: int = 256, n_seeds: int = 3, seed: int = 0, device: str = "cpu",
                 aggregator: str = "attention", aux_groups: bool = False):
        self.blocks = blocks
        self.kind = kind
        self.encoding = encoding
        self.cache_dir = cache_dir
        self.d = d
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.tiles_per_step = tiles_per_step
        self.n_seeds = n_seeds
        self.seed = seed
        self.device = device
        self.aggregator = aggregator
        self.aux_groups = aux_groups

    def _pooled(self, net: nn.Module, batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """(class scores, pooled slide feature, CLAM attention state) of a one-stream expert (v11)."""
        if isinstance(net, ClamNet):
            h, a = net.attend(batch)
            return net.scores(h, a), torch.einsum("ntb,ntd->nbd", a, h)[:, 0], (h, a)
        p = net.pool[0](net.proj[0](torch.cat(batch, dim=-1)))
        return net.heads[0](p), p, ()

    def _tiles(self, X: np.ndarray) -> list[torch.Tensor]:
        rows = self.blocks.take(X, ID)[:, 0].astype(int)
        out = []
        for m in self.blocks.images:
            T = np.asarray(load(self.cache_dir, m)[rows], dtype=np.float32)
            out.append(torch.as_tensor((T - self.mu_[m]) / self.sd_[m], device=self.device))
        return out

    def _meta(self, X: np.ndarray) -> dict | None:
        if META not in self.blocks.names:
            return None
        M = self.blocks.take(X, META)
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        return {"age_z": t((M[:, 0] - self.age_[0]) / self.age_[1]), "sex": t(M[:, 1].round(), torch.long),
                "sex_z": t((M[:, 1] - self.sex_[0]) / self.sex_[1]), "age_ple": t(self.ple_.transform(M[:, 0]))}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MILFusion":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        if self.aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}")
        if self.aggregator != "attention" and self.kind != "concat":
            raise ValueError("aggregators other than attention take one tile stream (kind concat)")
        if self.aggregator not in TAKES_META and META in self.blocks.names:
            raise ValueError(f"aggregator {self.aggregator} takes no age and sex")
        if self.aux_groups and (self.aggregator not in ("attention", "clam_sb") or META in self.blocks.names
                                or self.kind != "concat"):
            raise ValueError("the grade-group head is for one-stream attention or CLAM-SB experts without age and sex")
        self.classes_ = np.unique(y)
        rows = self.blocks.take(X, ID)[:, 0].astype(int)
        self.mu_, self.sd_ = {}, {}
        for m in self.blocks.images:
            T = np.asarray(load(self.cache_dir, m)[rows], dtype=np.float32).reshape(-1, load(self.cache_dir, m).shape[2])
            self.mu_[m], self.sd_[m] = T.mean(axis=0), np.maximum(T.std(axis=0), 1e-6)
        if META in self.blocks.names:
            M = self.blocks.take(X, META)
            self.age_ = (M[:, 0].mean(), max(M[:, 0].std(), 1e-6))
            self.sex_ = (M[:, 1].mean(), max(M[:, 1].std(), 1e-6))
            self.ple_ = PiecewiseLinear().fit(M[:, 0])
        tiles, meta = self._tiles(X), self._meta(X)
        yi = torch.as_tensor(np.searchsorted(self.classes_, y), device=self.device)
        K = len(self.classes_)
        counts = np.bincount(yi.cpu().numpy(), minlength=K)
        loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(len(y) / (K * counts), dtype=torch.float32,
                                                             device=self.device))
        if self.aux_groups:
            groups = np.asarray(GRADE_GROUPS)[self.classes_.astype(int)]
            gi = torch.as_tensor(groups[yi.cpu().numpy()], device=self.device)
            gc = np.bincount(gi.cpu().numpy(), minlength=4)
            gloss = nn.CrossEntropyLoss(weight=torch.as_tensor(len(y) / (4 * np.maximum(gc, 1)), dtype=torch.float32,
                                                               device=self.device))
        widths = [t.shape[2] for t in tiles]
        n_tiles = tiles[0].shape[1]
        self.nets_ = []
        for s in range(self.n_seeds):
            torch.manual_seed(self.seed * 1000 + s)
            gen = torch.Generator(device="cpu").manual_seed(self.seed * 1000 + s)
            if self.aggregator == "attention":
                net = MILNet(widths, self.kind, self.encoding if meta is not None else None, self.d, self.dropout,
                             K).to(self.device)
            elif self.aggregator == "xattn":
                net = CrossAttnNet(sum(widths), self.d, self.dropout, K,
                                   self.encoding if meta is not None else None).to(self.device)
            elif self.aggregator in CLAM:
                net = ClamNet(sum(widths), self.d, self.dropout, K, self.aggregator == "clam_mb").to(self.device)
            else:
                net = CompositionNet(sum(widths), self.d, self.dropout, K, self.aggregator == "wcomp").to(self.device)
            params = list(net.parameters())
            if self.aux_groups:
                net.group_head = nn.Linear(self.d, 4).to(self.device)        # registered: saved with the net
                params = list(net.parameters())
            opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            net.train()
            for _ in range(self.epochs):
                pick = torch.randperm(n_tiles, generator=gen)[:self.tiles_per_step].to(self.device)
                opt.zero_grad()
                batch = [t[:, pick] for t in tiles]
                if self.aux_groups:
                    z, pooled, state = self._pooled(net, batch)
                    loss = 0.5 * gloss(net.group_head(pooled), gi)
                    if state:
                        loss = loss + 0.7 * loss_fn(z, yi) + 0.3 * net.instance_loss(*state, yi)
                    else:
                        loss = loss + loss_fn(z, yi)
                elif self.aggregator in CLAM:
                    h, a = net.attend(batch)
                    loss = 0.7 * loss_fn(net.scores(h, a), yi) + 0.3 * net.instance_loss(h, a, yi)
                else:
                    loss = loss_fn(net(batch, meta), yi)
                loss.backward()
                opt.step()
            self.nets_.append(net.eval())
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tiles, meta = self._tiles(X), self._meta(X)
        P = torch.stack([torch.softmax(net(tiles, meta), dim=1) for net in self.nets_]).mean(dim=0)
        return P.double().cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
