# SPEC: Sports Market Calibration

Status: accepted, 2026-11-26. Drafted 2026-11-15; the author accepted the proposals for every open decision (table at the end).

## Problem

Prediction markets now quote a price on every World Cup match. A price is often read as a probability. Is it a well-calibrated one, and is it better than what a simple statistical model gives from past results alone?

This project compares pre-match prediction-market prices with Elo and Poisson models on the 2026 FIFA World Cup, using proper scoring rules and reporting the uncertainty that a sample of about a hundred matches allows. It does not ask whether any of this is profitable.

## Approach

### 1. Data

| Data | Source | Status on 2026-11-15 |
| --- | --- | --- |
| International match results | `martj42/international_results` on GitHub | Licence CC0-1.0 per the GitHub API; last pushed 2026-10-31 |
| Match markets | Polymarket Gamma API, events with slugs like `fifwc-esp-arg-2026-07-19` | Reachable without a key. Each match event has three binary markets (home win, draw, away win) and a `gameStartTime` |
| Market price history | Polymarket CLOB `prices-history` | Reachable without a key. For the final, a request with explicit `startTs` and `endTs` returned one-minute prices up to kickoff. A request with `interval=max` returned nothing for settled markets, so explicit bounds are required |
| Bookmaker closing odds | Not chosen | No free source with 2026 World Cup closing odds was confirmed |

Coverage, counted on 2026-11-26: all 104 matches have a Polymarket match event in series `soccer-fifwc` (id 11433), each with the three binary markets. The smallest event volume is about $5.9 million, so no match is thin by volume; thinness inside the pre-match window is still checked per match.

Settlement, read from the market descriptions: every match market refers "only to the outcome within the first 90 minutes of regular play plus stoppage time", including knockout matches. The results dataset records the score after extra time, so for knockout matches it cannot tell a 90-minute draw from a win. The outcome used for scoring is therefore the market's own resolution, cross-checked against the results dataset wherever no extra time was played.

All downloaded data is cached locally with the request time. The repository stores the derived table of pre-match probabilities and results, if the terms of each source allow it; otherwise it stores the download script only.

### 2. Turning market prices into probabilities

- **Pre-match price.** The time-weighted average price over a fixed window that ends before kickoff (proposed: 120 to 60 minutes before). A single last trade is too noisy, and prices closer to kickoff can react to line-up news that the models do not have.
- **Three outcomes.** The three binary prices rarely sum to one. They are normalized by dividing by their sum, and the raw sum is kept as a diagnostic.
- **What a market settles on.** In knockout matches a market may settle on the 90-minute result or on who advances. The rule is read from each market's description, and matches are scored against the same definition the market used. Markets whose rule cannot be determined are excluded and listed.
- **Thin markets.** Matches with too few price updates in the window are flagged and reported separately.

### 3. Models

All models are fitted on matches strictly before the match being predicted.

| Model | Description |
| --- | --- |
| Elo | Ratings updated match by match, with a home-advantage term (zero at neutral venues), a margin-of-victory multiplier and a weight by match type. Win, draw and loss probabilities come from an ordered logit on the rating difference |
| Independent Poisson | Attack and defence strengths per team, fitted by maximum likelihood with exponential time decay |
| Dixon–Coles | The Poisson model with the low-score correction of Dixon and Coles (1997) |
| Climatological baseline | Historical frequencies of home win, draw and away win. The floor that any model must beat |

Hyperparameters (Elo K-factor, time-decay rate) are chosen on matches before 2026 and frozen before any 2026 World Cup match is scored.

### 4. Evaluation

- Multiclass Brier score, log loss and the ranked probability score, which respects the order win, draw, loss.
- Reliability diagrams with binned confidence intervals.
- The Murphy decomposition of the Brier score into reliability, resolution and uncertainty.
- Paired bootstrap over matches for the difference between any two forecasters, with a confidence interval.
- Favourite–longshot check: calibration by price bucket.

The headline result is a table of scores with confidence intervals. If the intervals overlap, the README says the sample cannot separate the forecasters.

## Non-goals

- No betting strategy, no profit and loss, no claim of an edge.
- No in-play prices.
- No other sports or competitions in v1.
- No player-level or line-up data.
- No trading and no authenticated endpoints.

## Acceptance criteria

- [ ] A test shows that no model input for a match is dated on or after its kickoff.
- [ ] On a synthetic season generated from known Poisson parameters, the fitted parameters recover the truth within a stated tolerance.
- [ ] Scoring functions are tested against hand-computed examples, and the Brier decomposition sums to the Brier score.
- [ ] The pre-match window rule is fixed in configuration and tested on a recorded price series.
- [ ] Every excluded match is listed with its reason.
- [ ] Every number in the README comes from one documented command and has a confidence interval.
- [ ] The README states the sample size next to each result and has a Limitations section.
- [ ] `uv run pytest` and ruff pass in CI, with no network access in tests (recorded responses only).

## Milestones

1. **Scaffold and data.** Count market coverage first. Results loader, market loader, pre-match price rule, settlement-rule reader.
2. **Models.** Elo, Poisson, Dixon–Coles, baseline, with parameter-recovery tests.
3. **Evaluation.** Scores, reliability, decomposition, bootstrap.
4. **Results and README.**

Each milestone is one branch and one PR.

## Dependencies (checked 2026-11-15)

- numpy, pandas, scipy at runtime; matplotlib for figures.
- `penaltyblog` 1.13.1 has Dixon–Coles and Elo implementations and could serve as a test oracle. Its licence field on PyPI is empty, so the licence must be read from its repository before it is added, even as a development dependency.
- Polymarket publishes a fee schedule for sports markets (the final's markets carry the field `sports_fees_v3`). Fees do not enter this project, which compares probabilities only.
- Rate limits and terms of use for the Polymarket endpoints were not read. They must be read before the bulk download in milestone 1.

## Cost

No paid APIs and no cloud compute.

## Limitations known in advance

- **Sample size.** One tournament is 104 matches. Differences between good forecasters are small, so the likely honest result is "not distinguishable".
- **One market.** Results describe Polymarket during one tournament, not prediction markets in general.
- **Information sets differ.** The market sees injuries, line-ups and news. The models see past scores only. A market win is expected; the size of the gap is the interesting part.
- **Prices are not pure probabilities.** Fees, spreads and capital costs push prices away from beliefs, most of all near 0 and 1.

## Open decisions

| ID | Decision | Proposal |
| --- | --- | --- |
| D1 | Scope | 2026 World Cup only for v1 |
| D2 | Names | Repository `sports-market-calibration`, import name `smcal` |
| D3 | Pre-match window | Time-weighted average, 120 to 60 minutes before kickoff |
| D4 | Bookmaker odds as a second benchmark | Leave out of v1 unless a free source with clear terms is found |
| D5 | If Polymarket coverage is too thin | Widen to other football matches with Polymarket markets, or switch the benchmark to bookmaker odds |
| D6 | Storing derived market data in the repository | Decide after reading Polymarket's terms. Read 2026-11-26 (terms effective 2026-08-11): they prohibit scraping and grant no research exception. The repository stores the download script and aggregate scores only, never price series. |
