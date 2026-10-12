"""Prompt lookup: the suffix drafter, the draft policy, and the acceptance of a
deterministic draft.

The drafter and the policy are host-only bookkeeping (lookup_draft.py, ported
from Strata). Speed is not tested here; that needs the served model. What is
tested is that a lookup window cannot change what is emitted: under sampling,
a point-mass draft accepted by ``spec_sample.accept`` still commits tokens
distributed exactly as the target's ``p``.
"""

import numpy as np

from maxtoken.mlx_backend import spec_sample
from maxtoken.mlx_backend.lookup_draft import DraftPolicy, SuffixDrafter


# -- the suffix drafter ----------------------------------------------------------


def test_a_repeat_proposes_what_followed_it():
    d = SuffixDrafter()
    d.append([1, 2, 3, 4, 5, 6, 7, 9, 9, 1, 2, 3])
    assert d.propose(3) == [4, 5, 6]
    assert d.last_match == 3


def test_the_proposal_stops_at_max_k_and_at_the_history_end():
    d = SuffixDrafter()
    d.append([1, 2, 3, 4, 1, 2, 3])
    assert d.propose(2) == [4, 1]
    # the continuation may run into the current suffix itself (periodic text)
    assert d.propose(8) == [4, 1, 2, 3]


def test_no_proposal_without_a_trigram_repeat():
    d = SuffixDrafter()
    d.append([1, 2, 3, 4, 5, 6, 7, 8])
    assert d.propose(4) == []
    assert d.last_match == 0


def test_the_longest_match_wins_and_ties_keep_the_newest():
    d = SuffixDrafter()
    # "5 1 2 3" -> 10 (match 4), "1 2 3" -> 20 (match 3), then the suffix 5 1 2 3
    d.append([5, 1, 2, 3, 10, 0, 1, 2, 3, 20, 0, 5, 1, 2, 3])
    assert d.propose(1) == [10]
    assert d.last_match == 4
    d2 = SuffixDrafter()
    d2.append([1, 2, 3, 10, 7, 1, 2, 3, 20, 8, 1, 2, 3])
    assert d2.propose(1) == [20]


def test_appending_committed_tokens_extends_the_index():
    d = SuffixDrafter()
    d.append([7, 8, 9, 40, 41])
    assert d.propose(2) == []
    d.append([7, 8, 9])
    assert d.propose(2) == [40, 41]


def test_only_the_most_recent_ways_are_kept():
    d = SuffixDrafter()
    hist = []
    for marker in range(10, 10 + SuffixDrafter.WAYS + 2):
        hist += [1, 2, 3, marker]
    d.append(hist + [1, 2, 3])
    assert d.propose(1) == [10 + SuffixDrafter.WAYS + 1]


# -- the draft policy ------------------------------------------------------------


def test_no_proposal_keeps_the_mtp_window():
    assert DraftPolicy(4).choose(3, 0, 0) == (False, 3)


def test_a_long_match_takes_the_wider_window_before_any_measurement():
    use, t = DraftPolicy(4).choose(3, 3, 30)
    assert use and t == 4


def test_measured_costs_decide():
    p = DraftPolicy(4)
    # an MTP window of 3 commits 2.7 tokens in 26 ms; a 4-wide window costs 60 ms
    for _ in range(20):
        p.observe(False, 3, 2, 0, 26.0)
        p.observe(True, 4, 3, 30, 60.0)
    use, t = p.choose(3, 3, 30)
    assert not use and t == 3
    # the same 4-wide window at 31 ms pays
    q = DraftPolicy(4)
    for _ in range(20):
        q.observe(False, 3, 2, 0, 26.0)
        q.observe(True, 4, 3, 30, 31.0)
    assert q.choose(3, 3, 30) == (True, 4)


def test_rejected_lookups_lower_the_rate_until_the_policy_stops():
    p = DraftPolicy(4)
    for _ in range(20):
        p.observe(False, 3, 2, 0, 26.0)
    for _ in range(30):
        p.observe(True, 4, 0, 4, 31.0)
    assert p.lookup_rate(4) < 0.2
    assert p.choose(3, 3, 4)[0] is False


def test_an_unmeasured_size_is_probed_a_few_times_only():
    p = DraftPolicy(4)
    for _ in range(20):
        p.observe(False, 3, 2, 0, 1.0)  # an MTP window that looks unbeatable
    assert p.choose(3, 3, 30) == (True, 4)  # but a confident lookup gets tried
    for _ in range(3):
        p.observe(True, 4, 3, 30, 50.0)
    assert p.choose(3, 3, 30) == (False, 3)


def test_window_sizes_are_clamped_to_max_t():
    use, t = DraftPolicy(3).choose(3, 7, 30)
    assert t <= 3


# -- a lookup draft cannot change what is emitted ----------------------------------


def test_point_mass_drafts_commit_tokens_distributed_as_p():
    rng = np.random.default_rng(0)
    ids = np.array([5, 6, 7, 8], dtype=np.int64)
    p = np.array([0.5, 0.2, 0.2, 0.1])
    P = [(ids, p), (ids, p)]
    Q = spec_sample.point_rows([6])
    counts = np.zeros(4)
    n = 40_000
    for _ in range(n):
        accepted, nxt = spec_sample.accept([6], P, Q, rng)
        first = 6 if accepted else nxt
        counts[int(np.nonzero(ids == first)[0][0])] += 1
    assert np.allclose(counts / n, p, atol=0.01)


def test_a_point_mass_draft_is_accepted_with_probability_p_of_d():
    rng = np.random.default_rng(1)
    ids = np.array([1, 2], dtype=np.int64)
    P = [(ids, np.array([0.3, 0.7])), (ids, np.array([0.5, 0.5]))]
    n = 20_000
    acc = sum(spec_sample.accept([1], P, spec_sample.point_rows([1]), rng)[0] for _ in range(n))
    assert abs(acc / n - 0.3) < 0.015
