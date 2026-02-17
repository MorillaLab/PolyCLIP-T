from __future__ import annotations

from libs import *

class DistributedLabeledCollisionBatchSampler(Sampler[List[int]]):


    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        labeled_per_batch: int = 32,
        m_per_class: int = 2,
        unlabeled_val: int = -1,
        seed: int = 42,
        drop_last: bool = True,
        steps_per_epoch: int = 1000,
        dataset_size: Optional[int] = None,
    ):
        self.labels = np.asarray(labels).astype(int)
        self.batch_size = int(batch_size)
        self.labeled_per_batch = int(labeled_per_batch)
        self.m_per_class = int(m_per_class)
        self.unlabeled_val = int(unlabeled_val)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.steps_per_epoch = int(steps_per_epoch)

        self.dataset_size = int(dataset_size) if dataset_size is not None else int(len(self.labels))
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")

        if len(self.labels) < self.dataset_size:
            raise ValueError(f"labels length ({len(self.labels)}) < dataset_size ({self.dataset_size})")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if self.labeled_per_batch < 0 or self.labeled_per_batch > self.batch_size:
            raise ValueError("labeled_per_batch must be in [0, batch_size]")

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.epoch = 0

        idxs = np.arange(self.dataset_size, dtype=int)
        y = self.labels[: self.dataset_size]

        self.labeled_idxs = idxs[y != self.unlabeled_val]
        self.unlabeled_idxs = idxs[y == self.unlabeled_val]

        self.by_class: Dict[int, np.ndarray] = {}
        for c in np.unique(y[y != self.unlabeled_val]):
            self.by_class[int(c)] = idxs[y == int(c)]

        if self.labeled_per_batch > 0 and len(self.labeled_idxs) == 0:
            raise ValueError("No labeled samples found but labeled_per_batch > 0")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return int(math.ceil(self.steps_per_epoch / float(self.world_size)))

    def _rng(self):
        return np.random.default_rng(self.seed + 1000 * self.epoch + 17 * self.rank)

    def __iter__(self):
        rng = self._rng()

        def sample(pool: np.ndarray, k: int) -> np.ndarray:
            if k <= 0:
                return np.array([], dtype=int)
            if len(pool) == 0:
                return np.array([], dtype=int)
            replace = len(pool) < k
            return rng.choice(pool, size=k, replace=replace).astype(int)

        total_batches = self.steps_per_epoch
        for b in range(total_batches):
            if (b % self.world_size) != self.rank:
                continue

            batch: List[int] = []

            L = min(self.labeled_per_batch, self.batch_size)

            if L > 0:
                if self.m_per_class <= 1 or len(self.by_class) == 0:
                    chosen = sample(self.labeled_idxs, L)
                    batch.extend(chosen.tolist())
                else:
                    classes = list(self.by_class.keys())
                    rng.shuffle(classes)

                    remaining = L
                    ci = 0
                    while remaining > 0 and ci < 10_000:  
                        c = classes[ci % len(classes)]
                        take = min(self.m_per_class, remaining)
                        chosen = sample(self.by_class[c], take)
                        batch.extend(chosen.tolist())
                        remaining -= len(chosen)
                        ci += 1

                    if len(batch) < L:
                        batch.extend(sample(self.labeled_idxs, L - len(batch)).tolist())

            R = self.batch_size - len(batch)
            if R > 0:
                if len(self.unlabeled_idxs) > 0:
                    batch.extend(sample(self.unlabeled_idxs, R).tolist())
                else:
                    batch.extend(sample(np.arange(self.dataset_size), R).tolist())

            if len(batch) != len(set(batch)):
                seen = set()
                dedup = []
                for x in batch:
                    if x not in seen:
                        dedup.append(x)
                        seen.add(x)
                refill = self.batch_size - len(dedup)
                if refill > 0:
                    pool = self.unlabeled_idxs if len(self.unlabeled_idxs) > 0 else np.arange(self.dataset_size)
                    extra = sample(pool, refill).tolist()
                    for x in extra:
                        if len(dedup) >= self.batch_size:
                            break
                        if x not in seen:
                            dedup.append(x)
                            seen.add(x)
                    while len(dedup) < self.batch_size:
                        x = int(rng.integers(0, self.dataset_size))
                        if x not in seen:
                            dedup.append(x)
                            seen.add(x)
                batch = dedup

            if (not self.drop_last) or (len(batch) == self.batch_size):
                yield batch
