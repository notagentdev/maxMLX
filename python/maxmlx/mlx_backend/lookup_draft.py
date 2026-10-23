"""Prompt lookup for the MTP speculative loop: a second draft source, and the
policy that decides, round by round, which source fills the verify window.

When the answer repeats an earlier stretch of its context (code edits, quoted
input, tool-call JSON), the tokens that followed that stretch are usually the
next tokens too. ``SuffixDrafter`` finds them in microseconds and without a
model. ``DraftPolicy`` takes such a window only where it pays. It compares the
expected committed tokens per millisecond of the MTP's window against the
lookup window's, both learned online:

- MTP: the mean tokens committed by MTP windows of that size (EMA).
- Lookup: ``E(k) = 1 + q + ... + q^k`` for ``k`` up to the proposal, where
  ``q`` is the acceptance rate of lookup drafts whose match was about as long.
- Cost: the measured round time per window size (EMA). Sizes not seen yet are
  scaled from the seen ones by a prior shape.

Ported from Strata (github.com/Niko1221/Strata, MIT):
``src/spec/suffix_drafter.cpp`` and ``src/spec/draft_policy.cpp``. Their
measurements on a 125B MoE showed code edits 6-11 % faster and ordinary text
unchanged. As there, a lookup window is only considered when the MTP's own
first draft agrees with the lookup's first token. Verification decides every
emitted token, so the output is unchanged: drafts only decide how many tokens
a round yields.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# -- the suffix drafter ----------------------------------------------------------


class SuffixDrafter:
    """Proposes the tokens that followed the longest earlier occurrence of the
    current suffix.

    Every trigram of the history (the prompt, then each committed token) maps
    to its ``WAYS`` most recent end positions. A proposal checks those
    candidates and extends each match backwards up to ``max_match``. It takes
    the longest match; on ties it keeps the most recent. A match shorter than
    ``min_match`` proposes nothing.
    """

    WAYS = 4

    def __init__(self, min_match: int = 3, max_match: int = 64):
        self.min_match = max(3, int(min_match))
        self.max_match = max(self.min_match, int(max_match))
        self._hist: List[int] = []
        self._index: Dict[Tuple[int, int, int], List[int]] = {}
        self.last_match = 0

    def reset(self) -> None:
        self._hist = []
        self._index = {}
        self.last_match = 0

    def __len__(self) -> int:
        return len(self._hist)

    def append(self, tokens: Sequence[int]) -> None:
        hist, index, ways = self._hist, self._index, self.WAYS
        for t in tokens:
            hist.append(int(t))
            end = len(hist) - 1
            if end < 2:
                continue
            key = (hist[end - 2], hist[end - 1], hist[end])
            slot = index.get(key)
            if slot is None:
                index[key] = [end]
            else:
                slot.insert(0, end)
                del slot[ways:]

    def propose(self, max_k: int) -> List[int]:
        """Up to ``max_k`` proposed next tokens (empty when no match of at
        least ``min_match``). ``last_match`` is set to the match length."""
        self.last_match = 0
        hist = self._hist
        n = len(hist)
        if n < 4 or max_k <= 0:
            return []
        cur = n - 1
        slot = self._index.get((hist[cur - 2], hist[cur - 1], hist[cur]))
        if not slot:
            return []
        best_end, best_len = 0, 0
        for p in slot:  # most recent first, so ties keep the newer
            if p >= cur:  # the current suffix itself
                continue
            length = 0
            while (
                length < self.max_match
                and length <= p
                and hist[p - length] == hist[cur - length]
            ):
                length += 1
            if length > best_len:
                best_len, best_end = length, p
        if best_len < self.min_match:
            return []
        self.last_match = best_len
        # The continuation may run into the current suffix (periodic text);
        # reading the history up to `cur` is valid.
        return hist[best_end + 1 : min(cur, best_end + max_k) + 1]


# -- the draft policy ------------------------------------------------------------

MAX_T = 8
BUCKETS = 4
# A round's cost by window size relative to one token, used only for sizes not
# measured yet (the measured round times replace it). Strata's values; on this
# machine a verify row of Ornith costs ~5.3 ms of a ~26 ms round, a flatter
# curve, which the measured times take over after a few rounds.
_SHAPE = (0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6)
_COST_ALPHA = 0.1  # EMA weight of a new round time
_TOK_ALPHA = 0.05  # EMA weight of a new MTP window outcome
_DECAY = 0.97  # lookup counts: older windows fade
# Before a bucket has data: the longer the match, the likelier its
# continuation. Worth 4 observations, so a few real windows override it.
_PRIOR_Q = (0.75, 0.88, 0.93, 0.96)
_PRIOR_N = 4.0
_PROBES = 3  # a lookup size is tried this often before its guessed cost can veto it


class DraftPolicy:
    """Per round: the MTP's window, or a lookup window? Learned over the whole
    process (window costs are a property of the machine and model, not of a
    request)."""

    def __init__(self, max_t: int, margin: float = 0.03):
        self.max_t = max(1, min(int(max_t), MAX_T))
        self.margin = float(margin)
        self._cost = [0.0] * (MAX_T + 1)
        self._cost_n = [0.0] * (MAX_T + 1)
        self._mtp_tok = [0.0] * (MAX_T + 1)
        self._mtp_n = [0.0] * (MAX_T + 1)
        self._ok = [0.0] * BUCKETS
        self._bad = [0.0] * BUCKETS

    @staticmethod
    def bucket(match: int) -> int:
        return 0 if match < 6 else 1 if match < 12 else 2 if match < 24 else 3

    def lookup_rate(self, match: int) -> float:
        b = self.bucket(match)
        return (self._ok[b] + _PRIOR_N * _PRIOR_Q[b]) / (
            self._ok[b] + self._bad[b] + _PRIOR_N
        )

    def cost_ms(self, t: int) -> float:
        t = max(1, min(int(t), MAX_T))
        if self._cost_n[t] > 0:
            return self._cost[t]
        # scale from the measured sizes, weighting each by how often it was seen
        num = den = 0.0
        for u in range(1, MAX_T + 1):
            if self._cost_n[u] > 0:
                w = min(self._cost_n[u], 20.0)
                num += w * self._cost[u] * _SHAPE[t] / _SHAPE[u]
                den += w
        return num / den if den > 0 else _SHAPE[t]

    def mtp_tokens(self, t: int) -> float:
        if self._mtp_n[t] > 0:
            return self._mtp_tok[t]
        return 1.0 + 0.7 * (t - 1)  # before any MTP window of this size

    def choose(self, t_mtp: int, lookup_k: int, match: int) -> Tuple[bool, int]:
        """``(use_lookup, window size)``. ``t_mtp``: the MTP's window;
        ``lookup_k``: the lookup proposal's length; ``match``: its match
        length."""
        t = max(1, min(int(t_mtp), self.max_t))
        if lookup_k <= 0:
            return False, t
        base = self.mtp_tokens(t) / self.cost_ms(t)
        q = self.lookup_rate(match)
        e = qi = 1.0
        best, best_t = 0.0, 0
        for k in range(1, min(lookup_k, self.max_t - 1) + 1):
            qi *= q
            e += qi
            r = e / self.cost_ms(k + 1)
            if r > best:
                best, best_t = r, k + 1
        if best_t > 0 and best > base * (1.0 + self.margin):
            return True, best_t
        # A guessed cost can keep the policy from ever measuring a size: the
        # first few times a confident lookup would need a size not measured
        # yet, it is tried (verification keeps the output; only the one
        # round's speed is at stake).
        t_full = min(lookup_k, self.max_t - 1) + 1
        if t_full > t and self._cost_n[t_full] < _PROBES and q >= 0.85:
            return True, t_full
        return False, t

    def observe(
        self, lookup: bool, t: int, accepted: int, match: int, round_ms: float
    ) -> None:
        """After a round: the source it used, its window, the drafts accepted
        and the round's time."""
        t = max(1, min(int(t), MAX_T))
        if round_ms > 0:
            if self._cost_n[t] > 0:
                self._cost[t] = (1 - _COST_ALPHA) * self._cost[t] + _COST_ALPHA * round_ms
            else:
                self._cost[t] = round_ms
            self._cost_n[t] += 1.0
        if lookup:
            b = self.bucket(match)
            self._ok[b] = _DECAY * self._ok[b] + accepted
            self._bad[b] = _DECAY * self._bad[b] + (1.0 if accepted < t - 1 else 0.0)
        else:
            got = accepted + 1.0
            if self._mtp_n[t] > 0:
                self._mtp_tok[t] = (1 - _TOK_ALPHA) * self._mtp_tok[t] + _TOK_ALPHA * got
            else:
                self._mtp_tok[t] = got
            self._mtp_n[t] += 1.0
