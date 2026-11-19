# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/glodut815/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/glodut815/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecasting program that reads Exophase's recorded earn rate for every Grand Theft Auto VI trophy and predicts each trophy's rate one week after launch, as five quantiles, for the lowest pinball loss on the logit scale.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it works

A trophy's earn rate is a ratio — earners over tracked players — and both halves move: players keep earning trophies while Exophase keeps tracking new players who have earned almost nothing yet, so every refresh dilutes the rates. The program forecasts the two halves separately: it collapses the history to distinct page states, fits log earners and log tracked players against squashed time (ln(1 + hours since release)) by recency-weighted least squares, and reads both fits at the target. When the history is flat, it falls back to a refresh prior (one refresh before the target, tracked players up about sixfold, earners barely growing). The spread combines a random-walk floor, the fit's prediction error, and the lumpiness of refreshes. True constraints are enforced at the end: a later story chapter cannot exceed an earlier one, and the platinum cannot exceed 100% completion.

---

<sub>Entered on [make.fun](https://make.fun/?ref=s9fbxxcj&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
