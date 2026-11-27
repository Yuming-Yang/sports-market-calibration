# Decisions

Key choices, with the alternatives that were considered and dropped.

## Data

**Outcome is the market's 90-minute resolution, not the score in the results file.**
Every match market settles on the 90-minute result. The results file records the score after extra time. Five knockout matches have a full-time winner but a 90-minute draw. Scoring them by the full-time score would mark the market wrong on matches where it was right. The full-time score is still used as a check: it may differ from the market only by turning a draw into a win.

**Pre-match price: time-weighted average from 120 to 60 minutes before kickoff.**
Dropped: the last price before kickoff. It is the most informed price, but line-ups are announced about an hour before kickoff and the models never see them. Ending the window at 60 minutes keeps the comparison closer to equal information. In the data, prices rarely move inside this window (in at least half of the matches, one outcome's price did not change inside it), so the choice between a time-weighted average and the price at the window end matters little here.

**Normalization by the sum of the three prices.**
The three Yes prices sum to between 0.989 and 1.016. Dividing by the sum is the simplest way to get a distribution. Alternatives that move more mass away from longshots (power or Shin normalization) were not used, because with sums this close to one they change probabilities by well under one percentage point.

**No price series in the repository.**
Polymarket's terms (effective 2026-08-11) prohibit scraping and do not grant a research exception. The repository holds the download code and aggregate scores. `smcal data` rebuilds the per-match table from the public endpoints, with a delay between requests well under the documented rate limits.

## Models

**Elo link: a symmetric ordered logit with two parameters.**
Elo gives one rating difference per match. Turning it into three probabilities needs a draw model. The ordered logit treats home and away alike, so at a neutral venue swapping the labels mirrors the forecast. Not tried: a separate draw model as a function of |difference|, which needs more parameters.

**K table from the World Football Elo scheme, scaled by one tuned factor.**
The K table and goal-difference multiplier are standard. Only a global scale and the home advantage are tuned. Tuning each K separately would overfit the validation years.

**Goal models: ridge penalty on attack and defence.**
Over eight years there are about 290 national teams, many with few matches. The penalty shrinks them towards average and fixes the overall level, which the likelihood alone does not identify. Its strength is tuned with the time decay.

**Dixon-Coles rho fitted jointly, bounded to [-0.3, 0.3].**
tau can turn negative for very lopsided matches. The bound and a floor on tau keep the likelihood finite. For the World Cup forecasts the largest expected goal count was 3.8 and the fitted rho was about -0.06, so tau stays positive (it would need an expected goal count above 1/|rho|, about 17).

**Tuning: log loss on 2023-2025 competitive matches, then frozen.**
Friendlies are left out of validation because teams rotate players in them. The goal models are refitted at the start of each quarter, not each match, to keep tuning fast; the World Cup forecasts refit for each match date. The chosen values are in `config/frozen.json`. The ridge grid was widened once, before any World Cup match was scored, because the first run chose the smallest value.

**Training outcomes are full-time results.**
The results file includes extra time. For a small share of historical knockout matches the training outcome is therefore the post-extra-time result rather than the 90-minute one. This is a known bias that cannot be fixed with this dataset.

**Baseline: draw rate from the 1998-2022 World Cups, the rest split evenly.**
The draw rate in that data is slightly understated for the same extra-time reason.

## Evaluation

**Bootstrap over matches, paired.**
A match is the resampled unit, so a match's three outcome probabilities stay together. Comparisons are paired because two forecasters' scores on one match are strongly correlated.

**Three scores, and many comparisons.**
Brier, log loss and RPS, on three subsets and six pairs, make 54 paired intervals. Some will exclude zero by chance. The README leads with one comparison, the SPEC's question: the market against the strongest results-only model (Dixon-Coles), on the Brier score, over all matches. The others are descriptive.

**Favourite-longshot check: one exponent instead of bucket-by-bucket tests.**
With 312 forecast-outcome pairs, five price buckets have 18 to 112 forecasts each, and bucket-level intervals are too wide to say much. One parameter, alpha in q ~ p^alpha, uses all the data at once. The reliability table is still reported.

**Within-bin terms in the Brier decomposition.**
Without them, reliability minus resolution plus uncertainty does not equal the Brier score for continuous forecasts, and a reader checking the sum would find a gap.
