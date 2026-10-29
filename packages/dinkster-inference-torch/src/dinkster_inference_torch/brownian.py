"""Brownian-tree step noise: the native BrownianTreeNoiseSampler port.

Three reference layers are transcribed here, because the validation
environments deliberately carry neither torchsde nor numpy:

1. ``SeedSequence`` - numpy's entropy-mixing hash
   (numpy/random/bit_generator.pyx, itself derived from Melissa
   O'Neill's C++11 seed_seq work; numpy pins the output stream as a
   compatibility guarantee, which is what makes this transcription
   sound). torchsde derives every tree node's torch seed from it.
2. ``_Interval`` / the tree in ``_BatchedTree`` - torchsde 0.2.6
   BrownianInterval (torchsde/_brownian/brownian_interval.py) in the
   exact configuration torchsde.BrownianTree pins and ComfyUI uses:
   ``halfway_tree=True``, ``tol=1e-6``, ``pool_size=24``, levy area
   ``"none"``, ``W=None``. Only that configuration is ported; the
   adaptive dependency-tree machinery, space-time levy areas, and
   point evaluation are all unreachable from it. The actual random
   draws are plain ``torch.randn`` from a per-node seeded generator,
   exactly like the reference.
3. ``BrownianTreeNoise`` - k_diffusion's BatchedBrownianTree +
   BrownianTreeNoiseSampler wrappers
   (comfy/k_diffusion/sampling.py @ b78cec87) with the ``cpu=True``
   tree every ported SDE solver constructs: sort/sign handling,
   float32-truncated query times (the reference's
   ``t.detach().cpu().float()``), and the ``1/sqrt(|t1-t0|)``
   increment scaling.

Deviations from the reference, all deliberate:

- The seed is a required int. The reference draws a fallback seed
  from global torch RNG when ``seed=None``; Dinkster seeds are explicit
  engine data and nothing may touch global RNG state.
- Per-batch-item seed lists (BatchedBrownianTree's ``batched`` mode)
  are not ported - no ported consumer can reach them (ComfyUI's
  samplers always pass the single int from ``extra_args["seed"]``).
  Revive with a consumer (ROADMAP: Native inference).
- The GPU-resident tree (``cpu=False``, only reachable through the
  unported ``*_gpu`` solver variants) is not ported; the tree always
  lives on CPU and the result moves to the latent's device, matching
  every solver Dinkster actually ships.
- Searches descend from the root instead of the reference's
  last-queried-interval hint. Value-neutral: every node's noise is
  deterministic in its seed, so the search entry point cannot change
  results (torchsde documents exactly this property; the 45-entry
  increment cache itself is transcribed, see ``_LRUCache``).
- Recursion is plain Python instead of the reference's trampoline.
  A single descent is bounded by ``log2((t1 - t0) / tol)`` - about 24
  levels for the largest SD sigma range at the pinned tol of 1e-6 -
  and the buck-passing bounces (see ``loc``) add at most one re-route
  per level, so the stack stays a few hundred frames deep in the
  worst case, far inside any recursion limit. The trampoline exists
  upstream because tol is caller-controlled there; here it is pinned
  by the BrownianTree configuration.
- Query times overhanging the rounded tree bounds by at most one
  1e-6 grid step clamp to the bound exactly like the reference (this
  is a REACHABLE input: the bounds round inward at construction, so
  querying at the raw t0/t1 overhangs by up to half a step). Larger
  overhang cannot come from rounding and refuses loudly instead of
  the reference's warn-and-clamp; silent clamping there would change
  the noise stream on a caller bug.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "BrownianNoiseError",
    "BrownianTreeNoise",
    "SeedSequence",
]


class BrownianNoiseError(ValueError):
    """A brownian noise request this port cannot execute: a negative
    or non-int seed, reversed tree bounds, or a query outside the
    tree's interval."""


# ---------------------------------------------------------------------------
# numpy SeedSequence (numpy/random/bit_generator.pyx), pure Python.
# All arithmetic is uint32 with wraparound; Python ints are masked
# after every multiply/subtract.

_MASK32 = 0xFFFFFFFF
_INIT_A = 0x43B0D7E5
_MULT_A = 0x931E8875
_INIT_B = 0x8B51F9DD
_MULT_B = 0x58F38DED
_MIX_MULT_L = 0xCA01F9DD
_MIX_MULT_R = 0x4973F715
_XSHIFT = 16


def _int_to_uint32_words(value: int, what: str) -> list[int]:
    """numpy's _int_to_uint32_array: a non-negative int as uint32
    words, lowest bits first (zero is a single zero word)."""
    if isinstance(value, bool):
        # bool passes int annotations; a True seed is a caller bug.
        raise BrownianNoiseError(f"{what} must be an int, got {type(value).__name__}")
    if value < 0:
        raise BrownianNoiseError(f"{what} must be non-negative, got {value}")
    if value == 0:
        return [0]
    words = []
    while value > 0:
        words.append(value & _MASK32)
        value >>= 32
    return words


class SeedSequence:
    """numpy.random.SeedSequence's hash mixing, transcribed: the same
    entropy pool and ``generate_state`` word stream for int entropy
    and int spawn keys (the only shapes torchsde's trees produce).
    numpy freezes this algorithm as a stream-compatibility guarantee,
    so pinning against it is stable."""

    def __init__(
        self,
        entropy: int,
        *,
        spawn_key: tuple[int, ...] = (),
        pool_size: int = 4,
    ) -> None:
        run_entropy = _int_to_uint32_words(entropy, "entropy")
        spawn_entropy: list[int] = []
        for key in spawn_key:
            spawn_entropy.extend(_int_to_uint32_words(key, "spawn_key element"))
        if spawn_entropy and len(run_entropy) < pool_size:
            # numpy pads the entropy with zeros to the pool size when a
            # spawn key is present (gh-16539 stream fix, numpy >= 1.19).
            run_entropy = run_entropy + [0] * (pool_size - len(run_entropy))
        assembled = run_entropy + spawn_entropy

        pool = [0] * pool_size
        hash_const = _INIT_A

        def hashmix(value: int) -> int:
            nonlocal hash_const
            value ^= hash_const
            hash_const = (hash_const * _MULT_A) & _MASK32
            value = (value * hash_const) & _MASK32
            value ^= value >> _XSHIFT
            return value

        def mix(x: int, y: int) -> int:
            result = (_MIX_MULT_L * x - _MIX_MULT_R * y) & _MASK32
            result ^= result >> _XSHIFT
            return result

        # Add in the entropy up to the pool size.
        for i in range(pool_size):
            pool[i] = hashmix(assembled[i] if i < len(assembled) else 0)
        # Mix all bits together so late bits can affect earlier bits.
        for i_src in range(pool_size):
            for i_dst in range(pool_size):
                if i_src != i_dst:
                    pool[i_dst] = mix(pool[i_dst], hashmix(pool[i_src]))
        # Add any remaining entropy, mixing each new word with each
        # pool word.
        for i_src in range(pool_size, len(assembled)):
            for i_dst in range(pool_size):
                pool[i_dst] = mix(pool[i_dst], hashmix(assembled[i_src]))

        self._pool = pool

    def generate_state(self, n_words: int) -> list[int]:
        """The uint32 word stream numpy's generate_state produces."""
        hash_const = _INIT_B
        state = []
        pool = self._pool
        for i_dst in range(n_words):
            data_val = pool[i_dst % len(pool)]
            data_val ^= hash_const
            hash_const = (hash_const * _MULT_B) & _MASK32
            data_val = (data_val * hash_const) & _MASK32
            data_val ^= data_val >> _XSHIFT
            state.append(data_val)
        return state


# ---------------------------------------------------------------------------
# torchsde BrownianInterval, halfway-tree/none-levy configuration.

#: torchsde.BrownianTree's pinned constructor facts (derived.py):
#: the query grid and the per-node SeedSequence pool size.
_TREE_TOL = 1e-6
_TREE_POOL_SIZE = 24
#: round() digits for _TREE_TOL, computed exactly like the reference
#: (brownian_interval.py: ``ndigits = -int(math.log10(tol))``).
_TREE_NDIGITS = -int(math.log10(_TREE_TOL))


def _randn(
    size: tuple[int, ...], dtype: torch.dtype, device: torch.device, seed: int
) -> torch.Tensor:
    """brownian_interval.py _randn: a fresh generator per draw, seeded
    with the SeedSequence-derived uint32."""
    generator = torch.Generator(device).manual_seed(int(seed))
    return torch.randn(size, dtype=dtype, device=device, generator=generator)


def _round(x: float) -> float:
    return round(x, _TREE_NDIGITS)


#: brownian_interval.py ``cache_size`` default; torchsde.BrownianTree
#: does not override it. LOAD-BEARING for memory, not values: without
#: the bound, one latent-sized increment per materialized node stays
#: alive for the tree's whole life and memory grows with every new
#: schedule boundary queried (steps x depth x latent size).
_CACHE_SIZE = 45


class _LRUCache:
    """brownian_interval.py _LRUDict: insertion-ordered eviction where
    only ``__setitem__`` refreshes recency - reads deliberately do NOT
    reorder, exactly like the reference. Eviction is value-neutral:
    a dropped increment recomputes bit-identically from its parent
    chain (each node's noise is deterministic in its stored seed)."""

    __slots__ = ("_data", "_max_size")

    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: dict[_Interval, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: _Interval) -> torch.Tensor | None:
        return self._data.get(key)

    def __setitem__(self, key: _Interval, value: torch.Tensor) -> None:
        data = self._data
        if key in data:
            del data[key]
        elif len(data) >= self._max_size:
            del data[next(iter(data))]
        data[key] = value


class _Interval:
    """One node of the halfway tree: a subinterval of [t0, t1] whose
    Brownian increment derives from its parent's increment plus noise
    seeded by the parent's SeedSequence spawn (brownian_interval.py
    _Interval, the ``_have_H=False`` leg only)."""

    __slots__ = (
        "start",
        "end",
        "parent",
        "is_left",
        "midway",
        "spawn_key",
        "depth",
        "w_seed",
        "left_child",
        "right_child",
    )

    def __init__(
        self,
        start: float,
        end: float,
        parent: _Interval | None,
        is_left: bool,
    ) -> None:
        self.start = _round(start)
        self.end = _round(end)
        self.parent = parent
        self.is_left = is_left
        self.midway: float | None = None
        self.left_child: _Interval | None = None
        self.right_child: _Interval | None = None

    def increment(self, top: _BrownianTree) -> torch.Tensor:
        """W over [start, end] - the reference's
        _increment_and_space_time_levy_area without the unused H leg,
        cached in the tree's bounded LRU (a cache miss recomputes
        bit-identically from the parent chain: the noise is
        deterministic in the parent's stored seed)."""
        cached = top.cache.get(self)
        if cached is not None:
            return cached
        parent = self.parent
        assert parent is not None and parent.midway is not None
        w = parent.increment(top)
        h_reciprocal = 1 / (parent.end - parent.start)
        left_diff = parent.midway - parent.start
        right_diff = parent.end - parent.midway

        mean = left_diff * w * h_reciprocal
        var = left_diff * right_diff * h_reciprocal
        noise = top.randn(parent.w_seed)
        left_w = mean + math.sqrt(var) * noise

        out_w = left_w if self.is_left else w - left_w
        top.cache[self] = out_w
        return out_w

    def loc(self, ta: float, tb: float, out: list[_Interval], top: _BrownianTree) -> None:
        """_loc_inner: collect the ordered existing/created
        subintervals that exactly cover [ta, tb], splitting on the
        halfway grid where needed. ``ta``/``tb`` arrive rounded and
        inside the TREE's [t0, t1] (the top-level query validates),
        but not necessarily inside THIS node: the halfway _split
        bisects at midpoints, so the post-split descent below lands
        in a child whose jurisdiction excludes the query, and that
        child must pass the buck back up to its parent (which now
        has a midway and routes through the ordinary branches)."""
        if ta < self.start or tb > self.end:
            parent = self.parent
            assert parent is not None
            parent.loc(ta, tb, out, top)
            return
        if ta == self.start and tb == self.end:
            out.append(self)
            return
        if self.midway is None:
            if ta == self.start:
                self.split(tb, top)
                assert self.left_child is not None
                self.left_child.loc(ta, tb, out, top)
                return
            self.split(ta, top)
            assert self.right_child is not None
            self.right_child.loc(ta, tb, out, top)
            return
        if tb <= self.midway:
            assert self.left_child is not None
            self.left_child.loc(ta, tb, out, top)
            return
        if ta >= self.midway:
            assert self.right_child is not None
            self.right_child.loc(ta, tb, out, top)
            return
        assert self.left_child is not None and self.right_child is not None
        self.left_child.loc(ta, self.midway, out, top)
        self.right_child.loc(self.midway, tb, out, top)

    def split(self, midway: float, top: _BrownianTree) -> None:
        """The halfway-tree _split: always bisect at the (rounded)
        midpoint, then recurse toward the requested split point until
        it lands on the halfway grid."""
        self.split_exact(0.5 * (self.end + self.start), top)
        assert self.midway is not None
        if midway > self.midway:
            assert self.right_child is not None
            self.right_child.split(midway, top)
        elif midway < self.midway:
            assert self.left_child is not None
            self.left_child.split(midway, top)

    def split_exact(self, midway: float, top: _BrownianTree) -> None:
        """_split_exact: create both children and derive this node's
        draw seed from (entropy, (spawn_key, depth)) - the reference
        generates four words (W, H, left/right levy) and this
        configuration consumes only the first."""
        self.midway = _round(midway)
        self.set_spawn_key_and_depth()
        generator = SeedSequence(
            top.entropy,
            spawn_key=(self.spawn_key, self.depth),
            pool_size=_TREE_POOL_SIZE,
        )
        self.w_seed = generator.generate_state(4)[0]
        self.left_child = _Interval(start=self.start, end=midway, parent=self, is_left=True)
        self.right_child = _Interval(start=midway, end=self.end, parent=self, is_left=False)

    def set_spawn_key_and_depth(self) -> None:
        parent = self.parent
        assert parent is not None
        self.spawn_key = 2 * parent.spawn_key + (0 if self.is_left else 1)
        self.depth = parent.depth + 1


class _BrownianTree(_Interval):
    """torchsde BrownianInterval in the BrownianTree configuration:
    the root node plus the query surface. Holds the tensor facts
    (size/dtype/device), the global entropy, and the bounded increment
    cache; the root increment is drawn at construction and stored
    permanently OUTSIDE the cache, exactly like the reference (the
    BrownianInterval override returns the stored top-level W). A
    zero-width tree (t0 == t1) is legal like the reference (only
    t0 > t1 refuses): the reference SDE solvers build one for a
    single-step schedule (sigma, 0) and never query it."""

    __slots__ = ("size", "dtype", "device", "entropy", "cache", "root_w")

    def __init__(
        self,
        t0: float,
        t1: float,
        size: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        entropy: int,
    ) -> None:
        if t0 > t1:
            raise BrownianNoiseError(f"brownian tree needs t0 <= t1, got t0={t0} and t1={t1}")
        super().__init__(start=t0, end=t1, parent=None, is_left=False)
        self.size = size
        self.dtype = dtype
        self.device = device
        self.entropy = entropy
        self.cache = _LRUCache(_CACHE_SIZE)
        # The reference also draws the top-level space-time levy area H
        # from the second word; this configuration never reads it, and
        # skipping the draw cannot shift any other value (every draw
        # runs its own seeded generator).
        initial_w_seed = SeedSequence(entropy, pool_size=_TREE_POOL_SIZE).generate_state(3)[0]
        self.root_w = self.randn(initial_w_seed) * math.sqrt(t1 - t0)

    def randn(self, seed: int) -> torch.Tensor:
        return _randn(self.size, self.dtype, self.device, seed)

    def increment(self, top: _BrownianTree) -> torch.Tensor:
        return self.root_w

    def set_spawn_key_and_depth(self) -> None:
        self.spawn_key = 0
        self.depth = 0

    def query(self, ta: float, tb: float) -> torch.Tensor:
        """BrownianInterval.__call__ for an interval query: W over
        [ta, tb] with ta <= tb, both inside [t0, t1]. The tree bounds
        are rounded to the 1e-6 grid at construction, so a query at
        the RAW t0/t1 can overhang a rounded-inward bound by up to
        half a grid step; the reference warns and clamps, and this
        port clamps identically (same numerics: the clamped time is
        the rounded bound). Deviation: overhang beyond one grid step
        cannot come from rounding, only from a caller bug, and
        refuses instead of the reference's warn-and-clamp."""
        if ta < self.start:
            if self.start - ta > _TREE_TOL:
                raise BrownianNoiseError(
                    f"brownian query [{ta}, {tb}] leaves the tree interval"
                    f" [{self.start}, {self.end}]"
                )
            ta = self.start
        if tb > self.end:
            if tb - self.end > _TREE_TOL:
                raise BrownianNoiseError(
                    f"brownian query [{ta}, {tb}] leaves the tree interval"
                    f" [{self.start}, {self.end}]"
                )
            tb = self.end
        if ta > tb:
            raise BrownianNoiseError(f"brownian query needs ta <= tb, got [{ta}, {tb}]")
        if ta == tb:
            return torch.zeros(self.size, dtype=self.dtype, device=self.device)
        intervals: list[_Interval] = []
        self.loc(_round(ta), _round(tb), intervals, self)
        w = intervals[0].increment(self)
        for interval in intervals[1:]:
            w = w + interval.increment(self)
        return w


# ---------------------------------------------------------------------------
# The k_diffusion wrappers (comfy/k_diffusion/sampling.py @ b78cec87).


def _sort(a: float, b: float) -> tuple[float, float, int]:
    """BatchedBrownianTree.sort."""
    return (a, b, 1) if a < b else (b, a, -1)


class BrownianTreeNoise:
    """``NoiseSampler`` producing brownian-tree increments scaled by
    ``1/sqrt(|t1 - t0|)`` - the native BrownianTreeNoiseSampler +
    BatchedBrownianTree (@ b78cec87) over the transcribed torchsde
    machinery above.

    ``like`` fixes the noise shape/dtype/device (the reference's
    ``x``); ``sigma_min``/``sigma_max`` bound the tree exactly like
    the reference solvers' ``sigmas[sigmas > 0].min(), sigmas.max()``
    - on SNR-offset flow schedules pass the PRE-offset bounds, because
    the reference computes them before applying the offset. ``cpu``
    selects the reference's CPU tree default or its ``*_gpu``
    device-resident tree. Query times are truncated to float32 exactly like the
    reference's ``t.detach().cpu().float()`` before snapping to the
    1e-6 tree grid.
    """

    def __init__(
        self,
        like: torch.Tensor,
        sigma_min: float,
        sigma_max: float,
        *,
        seed: int,
        cpu: bool = True,
    ) -> None:
        self._dtype = like.dtype
        self._device = like.device
        # The reference's t0/t1 travel as tensors and hit the tree
        # through .float(): build the tree over the float32-truncated
        # bounds, sorted with the reference's sign. Equal bounds are
        # legal (a zero-width tree): the reference solvers construct
        # one for a single-step (sigma, 0) schedule and never query
        # it - a query over a zero-width span divides by sqrt(0) in
        # the reference too, so nothing here adds a refusal on top.
        t0, t1, self._sign = _sort(self._as_float32(sigma_min), self._as_float32(sigma_max))
        self._tree = _BrownianTree(
            t0=t0,
            t1=t1,
            size=tuple(like.shape),
            dtype=like.dtype,
            device=torch.device("cpu") if cpu else like.device,
            entropy=seed,
        )

    @staticmethod
    def _as_float32(value: float) -> float:
        """The reference's query-time dtype path: sigma -> float32
        tensor -> ``.float()`` -> ``float(t)``. Exact float32
        truncation, widened back to a Python float."""
        return float(torch.tensor(value, dtype=torch.float32))

    def __call__(self, sigma_from: float, sigma_to: float) -> torch.Tensor:
        qa = self._as_float32(sigma_from)
        qb = self._as_float32(sigma_to)
        t0, t1, sign = _sort(qa, qb)
        w = self._tree.query(t0, t1) * (self._sign * sign)
        w = w.to(device=self._device, dtype=self._dtype)
        # The reference receives device-resident schedule tensors and
        # computes the float32 normalization on that device.
        denominator = (
            (
                torch.tensor(qb, device=self._device, dtype=torch.float32)
                - torch.tensor(qa, device=self._device, dtype=torch.float32)
            )
            .abs()
            .sqrt()
        )
        return w / denominator
