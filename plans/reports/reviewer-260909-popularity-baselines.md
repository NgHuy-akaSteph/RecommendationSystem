# Code Review — Popularity Baselines

**Scope:** Uncommitted changes in `src/baselines.py` and `tests/test_baselines_popularity.py` (person 1, popularity baseline).
**Reference contract:** `notes/person1-implementation-guide.md` §1 + §5 (evaluator signature).
**Verification method:** Read both files, ran the test file (`21 passed`), and executed independent formula checks in the project venv.

---

## Findings (ordered by severity)

### F1 — HIGH · `BayesianPopularityRecommender` default `m=None` is effectively a no-op on MovieLens 25M

`fit()` computes `prior_strength = votes.median()` when `m` is `None`. With the long-tail item distribution of ML-25M, the per-item vote count median is **≈ 1 vote**. IMDb-style shrinkage with `m=1` barely contracts low-volume items toward the global mean, so the "Bayesian" baseline degrades to "raw weighted mean ≈ item mean for popular items". This silently defeats the purpose of the model — by definition the Bayesian baseline should be more conservative than a plain mean when most items have < 10 ratings.

- **Evidence:** On a Zipf(α=2) synthetic catalog (10⁵ items), `median` vote count = 1, `25th` percentile = 1.
- **Recommendation:** Either (a) require `m` to be explicit (raise `ValueError` if `None`), (b) default to a documented heuristic such as the *global* mean of votes-per-item (i.e. `total_ratings / n_items`), or (c) document loudly in the docstring that the default is a poor choice for ML-25M and show how to pick `m`. Option (b) is the conventional IMDb/BR choice.
- **Source:** `src/baselines.py:78` (`prior_strength = float(self.m) if self.m is not None else float(votes.median())`).

### F2 — MEDIUM · `test_recommend_excludes_seen_items` masks a Wilson ranking regression risk

The test asserts `recommendations == [30]` for `seen={10, 20}` on `toy_df`. Today this passes, but only because Wilson produces identical scores for movieIds 20 and 30 on this fixture (`rate=0.667`, `votes=3`) and the merge-sort tie-break by `movieId` ascending happens to filter both. If anyone changes the fixture or the scoring formula even slightly, the test would need re-tightening. More importantly, this single test asserts *both* "excludes seen" and "exact ordering", but the test name only advertises the former. Either split into two tests or rename to advertise what's being checked.

- **Source:** `tests/test_baselines_popularity.py:46-50`.

### F3 — MEDIUM · Missing edge-case test coverage

The tests cover happy-path behaviour and validation, but miss several contract-level edge cases that the `evaluate_model` consumer in §5 will hit:

| Missing case | Why it matters |
|---|---|
| `recommend(n=0)` / `recommend(n=-1)` returns `[]` | §5.1 evaluator slices `predictions[:k]` so this is correct-by-construction, but the contract is undocumented and untested. |
| `recommend(n > len(ranked_items_))` returns the full list | Same reason; would catch a bug like off-by-one in slicing. |
| Refitting a fitted model replaces ranking cleanly | Currently implicit; one regression bug would leak stale state. |
| `seen` passed as `set()` (empty, not `None`) behaves the same as `seen=None` | `if not seen` covers both, but no test pins the contract. |
| NaN / non-numeric `rating` values in `train_df` | `groupby(...).mean()` silently drops NaNs → scores reflect fewer items than the row count. Baseline-tier issue, but the docstring says "MovieLens ratings" which won't contain NaN — only worth a one-line guard or comment. |
| Bayesian with default `m=None` produces a sensible shrink on a low-vote item | F1 cannot be fixed/tested without this assertion. |

- **Source:** `tests/test_baselines_popularity.py` (whole file).

### F4 — LOW · `WilsonLowerBoundRecommender` uses `.pow(0.5)` where `np.sqrt` is clearer

```python
margin = self.z * (
    positive_rate * (1.0 - positive_rate) / votes
    + z_squared / (4.0 * votes**2)
).pow(0.5)
```

The expression inside `pow(0.5)` is mathematically guaranteed non-negative, so `.pow(0.5)` is safe but slightly opaque to readers expecting `np.sqrt`. Purely a readability/maintenance nit. Not blocking.

- **Source:** `src/baselines.py:107-110`.

### F5 — LOW · `_set_ranking` computes the `sort_values` on a freshly-built DataFrame without preserving original column names

The intermediate `ranking` DataFrame is created only to extract `movieId` in order; `self.scores_` is then rebuilt from the *original* `scores` Series. Both pathways agree on item identities, so this is correct, but the indirection (build DataFrame → sort → re-iterate) is harder to audit than a direct `scores.sort_values(..., kind="mergesort")` followed by `index.tolist()` and `.to_dict()`. Pure style nit.

- **Source:** `src/baselines.py:18-31`.

---

## What was checked and is OK (no defects found)

- **Math:**
  - Wilson lower bound matches the textbook formula `(centre − margin) / denom` clipped to `[0, 1]`. Independently re-derived and confirmed against `wilson_lower(1.0, 10)`, `(1.0, 3)`, `(1.0, 2)` — all bit-identical with the implementation.
  - Bayesian formula `v/(v+m) · R_i + m/(v+m) · C` matches IMDb/BR. Numerical match verified for `m=2` on a 3-item fixture.
  - Popularity = `groupby.size()` (count of interactions, identical to `value_counts()`).
- **Interface:**
  - `fit(train_df) -> self` ✓ (verified `is model` in tests).
  - `recommend(user_id, n, seen) -> list[int]` ✓ — `list[tuple]` branch in §5.1 evaluator degrades correctly because baselines return `list[int]` (evaluator handles both).
  - Public attributes `popular_items` / `score_by_item` aliased to `ranked_items_` / `scores_` (identity, not copy) — confirmed via `id(...)` — so any consumer reading either form sees the same object.
- **Determinism / tie-break:** `sort_values(["score", "movieId"], ascending=[False, True], kind="mergesort")` is stable and deterministic. Re-fit on identical input produces identical output (test `test_fit_is_deterministic` passes for all three classes).
- **Input validation:** `ValueError` raised for missing required columns and empty `train_df` (both branches covered by `test_invalid_input_is_rejected`). `m <= 0` and `z <= 0` rejected at construction.
- **Pre-fit guard:** `RuntimeError("Call fit() first")` raised by all three recommenders (tested).
- **Test suite:** 21 / 21 pass; parametrisation covers all three classes consistently across guard, self-return, ID-type, seen-filter and determinism tests.
- **Numerical stability for ML-25M:**
  - `groupby.size()` returns `int64` — total of 25 M ratings is well below 2⁶³.
  - Wilson: `z² / votes` with `votes` up to ~10⁵ stays in float64 range; `(centre − margin) / denom` cannot overflow.
  - `clip(0, 1)` correctly floors any negative rounding artefacts.
  - `mergesort` on ~62 K movieIds is sub-second.

---

## Verdict

**Approve with required follow-ups (F1 must be fixed; F2–F3 should be addressed before merge).**

The popularity baselines are correctly implemented and the public surface matches the implementation guide and the downstream `evaluate_model` consumer. All 21 unit tests pass and the math has been independently re-derived. The only real defect is the default prior `m` in `BayesianPopularityRecommender` (F1) — under the ML-25M distribution it disables the very shrinkage the model exists to provide. Two further items (F2, F3) tighten contract guarantees but do not block delivery.

---

## Required changes before delivery

1. **F1 — fix Bayesian default `m`.** Pick one of:
   - Raise `ValueError` if `m` is `None`, or
   - Default to `total_ratings / n_items` (the conventional IMDb heuristic), or
   - Document the heuristic in the docstring and add a test that pins the behaviour.
2. **F2 — split or rename** `test_recommend_excludes_seen_items` so it clearly advertises both the exclusion and the ordering invariants, or assert only `set(recommendations) & {10, 20} == set()` and add a separate ordering test on a fixture where Wilson/Bayesian scores are distinct.
3. **F3 — add edge-case tests** for at least: `n=0`, `n > len(ranked_items_)`, refit-replaces-state, `seen=set()` vs `seen=None`.

Optional polish (not blocking): F4 swap `.pow(0.5)` → `np.sqrt`; F5 simplify `_set_ranking`.

---

**Status:** DONE_WITH_CONCERNS
**Summary:** Formulas, interface, determinism, validation and Wilson/Bayesian numerics are all correct (21/21 tests pass, math independently re-derived). One real defect remains: Bayesian default `m=votes.median()` collapses to ~1 on ML-25M and disables shrinkage. Two contract-tightening gaps in tests should also be addressed.
**Concerns/Blockers:** None blocking; F1 is a design defect that should be resolved before running the §5 evaluation, since otherwise the "Bayesian" row in the result table is essentially a duplicate of mean-rating.
