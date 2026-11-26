# Decisions

Key choices, with the alternatives that were considered and dropped.

## Data

**Outcome is the market's 90-minute resolution, not the score in the results file.**
Every match market settles on the 90-minute result. The results file records the score after extra time. Five knockout matches have a full-time winner but a 90-minute draw. Scoring them by the full-time score would mark the market wrong on matches where it was right. The full-time score is still used as a check: it may differ from the market only by turning a draw into a win.

**Pre-match price: time-weighted average from 120 to 60 minutes before kickoff.**
Dropped: the last price before kickoff. It is the most informed price, but line-ups are announced about an hour before kickoff and the models never see them. Ending the window at 60 minutes keeps the comparison closer to equal information. In the data, prices rarely move inside this window (median zero changes per outcome), so the choice between a time-weighted average and the price at the window end matters little here.

**Normalization by the sum of the three prices.**
The three Yes prices sum to between 0.989 and 1.016. Dividing by the sum is the simplest way to get a distribution. Alternatives that move more mass away from longshots (power or Shin normalization) were not used, because with sums this close to one they change probabilities by well under one percentage point.

**No price series in the repository.**
Polymarket's terms (effective 2026-08-11) prohibit scraping and do not grant a research exception. The repository holds the download code and aggregate scores. `smcal data` rebuilds the per-match table from the public endpoints, with a delay between requests well under the documented rate limits.
