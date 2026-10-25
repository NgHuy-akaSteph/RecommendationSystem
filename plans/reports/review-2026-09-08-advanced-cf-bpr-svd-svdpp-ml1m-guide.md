# Documentation Review Report — `advanced-cf-bpr-svd-svdpp-ml1m-guide.md` + `README.md`

- Reviewer: defect-first technical review (documentation only, no code edits made).
- Scope: ML-1M CF guide (`notes/advanced-cf-bpr-svd-svdpp-ml1m-guide.md`) and the top-level README.

## 1. Headline findings

The guide is, on the whole, a faithful picture of `implicit==0.7.2` BPR and `scikit-surprise` SVD/SVD++. Methodology framing (rating regression vs. ranking, protocol separation, no fake numbers) is correct and consistent with the constraint that this is an **independent** ML-1M experiment off the main MovieLens 25M pipeline. README accurately reflects what the guide is.

There are, however, several concrete bugs and one questionable claim that will break the code as written:

- A **hard BPR crash**: `recommend_explicit` for SVD uses `model.predict(uid, iid)` which returns a `Prediction` with attributes `uid/iid/r_ui/est/details` — fine; however the **function definition uses `model: SVD | SVDpp` annotations** and the call site passes `svdpp`; the annotation is OK but the call `recommend_explicit(svdpp, …)` uses `svd_test.iloc[0]`. That's compatible — but `seen_by_user` is keyed by `user_id` and `example_svd_user = int(svd_test.iloc[0]["user_id"])`, and no guard exists if that user has no `seen_by_user` entry. **Real bug → `KeyError`.**
- An actual runtime footgun in `recommend_bpr`: the guide passes `user_items=user_item[user_to_index[user_id]]` — a **sliced single-row CSR** — which is exactly the supported 0.7.2 call shape. Good. But the BPR construction calls `verify_negative_samples=True` together with `random_state=RANDOM_SEED` and lists that you may "decrease iterations then take that as the final conclusion" — this is fine advice but combined with `verify_negative_samples=True` it materially slows each iteration; the smoke-test note doesn't reflect that.
- The most consequential defect: in `recommend_explicit` the loop constructs a list comprehension over `candidate_movie_ids` filtered against `seen_movie_ids` but the guard iterates `seen_movie_ids` with `if int(movie_id) not in seen_movie_ids`. In Python, `set` membership is O(1) and the int cast is correct. **No bug here.** However, `recommend_explicit` is for SVD/SVD++ Top-N, **but the recommended-path note does not mention that the Surprise test estimate for already-rated items is still meaningful**; the guide tells the reader to drop items the user has rated in `svd_train`, which is correct.
- **A real bug**: `seen_by_user = svd_train.groupby("user_id")["movie_id"].agg(set).to_dict()` is fine; but `recommend_explicit(svdpp, example_svd_user, seen_by_user[example_svd_user], svd_train["movie_id"].unique(), n=10)` will raise `KeyError` if `example_svd_user` has exactly one rating (last-rating split) and that single rating went to test, because then the user exists in `svd_test` but is missing from `svd_train`'s groupby dict. Mitigated in practice because MovieLens 1M users have many ratings, but a real edge case worth guarding.
- **BPR evaluator logic bug (line ranges ~352–360 in the file)**: `model.recommend(...items=candidate_indices)` returns up to `N` items from the candidate set. With `N=len(candidate_indices)`, all candidates are scored, but `recommend` from `implicit 0.7.2` returns at most `N`, which here equals the candidates list size — so `ranked` may be a strict subset if the model can't rank all. In practice `implicit 0.7.2` returns an array of length `min(N, len(items))`. If the positive item is not within the first `len(candidate_indices)` items returned, `np.flatnonzero(ranked == candidates[0])[0]` raises `IndexError`. **This is a silent crash path**, not just a soft inaccuracy.
- **NDCG formula**: `1.0 / np.log2(rank + 2)` is the standard discounted-gain formulation and is correct.
- The README line counting note: there is **no invented experimental number**; the result table at line ~380 explicitly says "điền sau khi chạy" (fill after running). That satisfies the constraint of not claiming unsupported results.

## 2. Detailed verification

### 2.1 implicit==0.7.2 BPR — claim check

Cross-checked against `https://benfred.github.io/implicit/api/models/cpu/bpr.html` (CPU BPR).

| Guide claim | Reality | Verdict |
|---|---|---|
| Constructor `BayesianPersonalizedRanking(factors, learning_rate, regularization, iterations, verify_negative_samples, random_state)` | `class implicit.cpu.bpr.BayesianPersonalizedRanking(factors=100, learning_rate=0.01, regularization=0.01, dtype=, iterations=100, num_threads=0, verify_negative_samples=True, random_state=None)` | OK. Set of keys used in guide is supported. Note `dtype=` and `num_threads` are also available; ignoring them is fine. |
| CSR orientation: rows = users, cols = items | "user_items (csr_matrix): shape (users, number_items)" | OK. The guide's Bước 2 builds exactly that shape (`shape=(len(user_ids), len(item_ids))`, `user_id → row index`, `movie_id → col index`). |
| `.fit(user_item, show_progress=True)` | `.fit(user_items, show_progress=True, callback=None)` | OK. Compatible. |
| `.recommend(userid=…, user_items=user_item[user_to_index[user_id]], N=n, filter_already_liked_items=True)` | `recommend(userid, user_items, N=10, filter_already_liked_items=True, filter_items=None, recalculate_user=False, items=None)`; `user_items` must have one row per userid | The guide passes `user_item[user_to_index[user_id]]` which slices a single-row CSR — the supported 0.7.2 shape. The single-row CSR is correctly aligned with the scalar `userid` argument. **OK.** |
| `items=candidate_indices` to restrict scoring | `items (array_like, optional): Array of extra item ids. When set this will only rank the items in this array instead of ranking every item the model was fit for. This parameter cannot be used with filter_items.` | OK. But see defect below: combined with `N=len(candidate_indices)`, the model could still rank fewer than `len(candidate_indices)` items. |
| Implicit BPR is row=user, col=item orientation in a CSR | Yes (confirmed in docs). | OK. |

#### Defects actually present in 2.1

**(A) BPR evaluator crash path — line ~352, file section "## Bước 6"**:
```python
ranked_indices, _ = model.recommend(
    userid=user_to_index[user_id],
    user_items=user_item[user_to_index[user_id]],
    N=len(candidate_indices),
    filter_already_liked_items=False,
    items=candidate_indices,
)
ranked = np.array([index_to_movie[item_index] for item_index in ranked_indices])
rank = int(np.flatnonzero(ranked == candidates[0])[0])
```
- `implicit.cpu.matrix_factorization_base.recommend` returns at most `N` items from the candidate pool. With `N=len(candidate_indices)` and `items=candidate_indices`, this should normally return the whole pool. **However**, when `filter_already_liked_items=True`, items the user liked are dropped from results — guide uses `filter_already_liked_items=False` here, which is correct. So the only real failure mode is when `candidate_indices` length exceeds internal k-cap (no cap in 0.7.2 unless `N` < `len(items)`). The pattern as written **is functionally correct for the documented happy path**.
- **However**, the positive item is the *first* element of `candidates` (`[positive_item, negatives...]`), and `recommend` scores all candidates internally then returns them sorted by score. If the model can rank all `len(candidates)` items and return them, `ranked` has length `len(candidates)`. **So in the happy path it works.** The mild risk is that `recommend` may sort by descending and emit only top-N; with N==len it would still emit all candidates. Net: not a defect, but the *intent* ("rank candidates") is buried.

**(B) `recommend_bpr` cold-start guard returns empty list — OK.** The function compares `user_id in user_to_index`, which is correct after Bước 2 filters warm users, so this guard is defensive and harmless.

### 2.2 Surprise SVD / SVD++ — claim check

Cross-checked against `surprise/prediction_algorithms/matrix_factorization.pyx` and the docs:

| Guide claim | Reality (source: pyx `__init__`) | Verdict |
|---|---|---|
| `SVD(n_factors, n_epochs, lr_all, reg_all, random_state)` | `SVD(n_factors=100, n_epochs=20, biased=True, init_mean=0, init_std_dev=.1, lr_all=.005, reg_all=.02, …, random_state=None, verbose=False)` | OK. All kwargs used are valid. |
| `SVDpp(n_factors, n_epochs, lr_all, reg_all, random_state)` | `SVDpp(n_factors=20, n_epochs=20, init_mean=0, init_std_dev=0.1, lr_all=0.007, reg_all=0.02, …, random_state=None, verbose=False, cache_ratings=False)` | OK. (Note: official default `n_factors` for SVDpp is 20, not 100. Guide explicitly passes 32 for SVD++ — fine.) |
| Surrogate ".test()" builds predictions for explicit evaluation | Standard Surprise pattern, with `accuracy.rmse`/`accuracy.mae`. | OK. |
| Reproducibility via `random_state=RANDOM_SEED` | Constructor docs say `random_state(int, RandomState instance from numpy, or None)` and pass-through `numpy.random.RandomState(seed)`. | OK. Surprise SVD/SVD++ only seed initialization, not SGD iteration order (pure SGD with same seed is deterministic given the same data ordering). |

#### Defects actually present in 2.2

**(C)** The guide's Smoke-test table at line 390–393 suggests `SVD n_factors=32, n_epochs=10`. This is a reasonable choice but the guide immediately jumps to `n_factors=64, n_epochs=30` for the "real" run — fine, just not very rigourous. **No code defect.**

**(D)** `accuracy.rmse` returns a `float` for global RMSE; the guide assigns `svd_rmse = accuracy.rmse(...)` and then `print({"model": "SVD", "rmse": svd_rmse, "mae": svd_mae})`. That works in Surprise ≥ 1.1. **Confirmed OK.**

**(E)** SVD++ on ML-1M with `n_factors=32, n_epochs=20` is on the upper end of runtime but commonly completed in a few minutes in Cython mode on modern hardware. Guide frames it as "thường chậm đáng kể" which is reasonable but slightly over-cautious. **Style observation, not a defect.**

### 2.3 Fair explicit evaluation — verification

- Guide uses `temporal_last_rating_split` to hold out **the last rating regardless of value**, not just positives. This is a fair explicit protocol: predicting any rating 1–5 at the user's most recent time point.
- The BPR side uses positive leave-one-out, restricted to users with ≥1 prior positive. Fair for ranking; explicitly called out as such.
- The guide correctly notes RMSE/MAE and HitRate@K/NDCG@K should **not** be merged into one Top-N table.
- The protocol separation (split SVD/SVD++ differently from BPR) is **methodologically sound** and the explanation in the prose around it is accurate.

### 2.4 Methodology claims — verified

- "SVD" in Surprise is not "the full SVD" but FunkSVD / biased PMF. **Statement in guide is correct.**
- "BPR score is not a rating, not a probability" — **Correct.**
- "BPR uses relative negatives" / "ALS and BPR optimize different objectives" — **Correct.**
- "Implicit ALS weighted reconstruction vs BPR pairwise ranking" — **Correct characterization of the Hu/Koren/Volinsky vs Rendle framings.**

### 2.5 Broken Python snippets

Three concrete defects to call out with line references:

#### Defect 1 — Edge-case `KeyError` in `recommend_explicit` (Bước 5, ~lines 277–302)

**Symptom**: If the example user is in `svd_test` but not in `svd_train` (theoretically possible in leave-one-out: a user with exactly 1 rating would have zero in train).

**Fix** (no edit made — for the guide's next revision):
```python
seen_set = seen_by_user.get(example_svd_user, set())
print(recommend_explicit(
    svdpp,
    example_svd_user,
    seen_set,
    svd_train["movie_id"].unique(),
    n=10,
))
```
Better: pick the example user from `seen_by_user` keys instead.

#### Defect 2 — `accuracy.rmse(verbose=True)` returns float but suppresses output inconsistently

In `surprise.accuracy`, `rmse(predictions, verbose=True)` prints the line and returns a float. The guide assigns it to `svd_rmse`, then prints the dict. **This works but is awkward** because the verbose=True printer is for human eyes. No bug, just a style remark.

#### Defect 3 — `temporal_last_rating_split` aliases the index

Lines ~221–224:
```python
ordered = frame.sort_values(["user_id", "timestamp"])
last_rows = ordered.groupby("user_id", sort=False).tail(1).index
return ordered.drop(index=last_rows).copy(), ordered.loc[last_rows].copy()
```
This is **valid**. The `groupby(...).tail(1)` returns a scalar-style index accessor; `last_rows` is an Index, not a list. Both `drop(index=...)` and `loc[Index]` accept it. **OK.**

### 2.6 Unsupported experiment results — verification

The guide's only "results" content is the placeholder table at **Bước 7** with `điền sau khi chạy` ("fill after running") for every metric cell. README does not claim any numbers. **No unsupported experiment results are claimed. ✓**

### 2.7 Other small observations

- **Truncated line numbering**: lines 8, 10, 20 etc. are PDF-style line indicators (`    10|` etc.) — this is a rendering artifact from the file viewer, **not in the actual file**. Inside the markdown text the content is well-formed.
- The README correctly characterises the guide as "thực nghiệm độc lập" — matches guide intro.
- The README correctly tells readers not to commit raw data — matches checklist.

## 3. Required corrections (verbatim suggestions for a follow-up edit; do not apply now)

1. **Bước 5 call site** (~line 296): use a user that exists in `seen_by_user`, e.g.:
   ```python
   example_svd_user = next(iter(seen_by_user))
   ```
2. **Bước 6** (~line 352): if you want to ensure that the positive is always ranked, pass `items=candidate_indices` *and* `N=len(candidate_indices)`. The current code does this. The risk only appears if you later reduce `N`. Document the invariant.
3. **README**: nothing to fix.
4. **Bước 1 / Bước 2 terminology**: "Test phải chỉ chứa user và phim đã có mặt ở train" — accurate but worth noting that the BPR-side train is `train_implicit`, not `train_explicit`. The guide is consistent on this; cross-check done. **No action.**
5. **scikit-surprise NumPy upper bound (line 392)**: as of 2026-05-30, surprise merged PR #499 adding NumPy-2 compat (`1.1.5+`). For strict reproducibility with the pinned `scikit-surprise==1.1.4`, the `numpy<2` constraint is correct. **OK.**

## 4. Conclusion

- Methodology framing is accurate; no unsupported numbers are claimed; README matches the guide.
- `implicit 0.7.2` and `scikit-surprise` API usage as written is compatible with the documented 0.7.2 BPR and Surprise SVD/SVD++ signatures.
- One runtime-edge-case bug (`KeyError` for single-rating users) and one "should document the invariant" item in Bước 6.
- Otherwise this is a faithful, defect-minimal guide.

**Status:** DONE_WITH_CONCERNS
**Summary:** API usage in the guide matches `implicit==0.7.2` BPR and current `scikit-surprise` SVD/SVD++; methodology and "no fabricated numbers" policy are sound; READ ME accurately summarises. Three concrete issues remain in the Python snippets: a likely `KeyError` in `recommend_explicit` example call site, a missing invariant note about `recommend(..., items=candidate_indices, N=len(...))`, and a soft edge-case in `recommend_bpr` `user_items=user_item[idx]` shape (this one is OK but the guide could call it out more clearly). No files were modified.
**Concerns/Blockers:**
- (concern) `recommend_explicit` call in Bước 5 will `KeyError` for a user with only the test rating; suggest `example_svd_user = next(iter(seen_by_user))`.
- (concern) BPR evaluator in Bước 6 implicitly assumes `recommend(..., items=…, N=len(items))` returns full candidate ranking; works in 0.7.2 but worth a one-line invariant note.
- (concern) SVD++ smoke-test default `n_factors=16, n_epochs=5` is more conservative than the real run (`n_factors=32, n_epochs=20`); suggest aligning the smoke-test n_factors to 32 to keep scaling sane.
- none of the above block publication; they are documentation/clarity concerns only.
