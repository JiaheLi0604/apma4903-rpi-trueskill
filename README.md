# Is the RPI Fair? (APMA 4903, Fall 2026)

Code for our talk comparing TrueSkill ratings with the NCAA RPI on Division I women's volleyball (2022–2025).

## Data
Match and box-score data come from the `ncaavolleyballr` package by Jeffrey R. Stevens (MIT License), which collects results from stats.ncaa.org.
Source package: https://github.com/JeffreyRStevens/ncaavolleyballr

`data/` holds the 2025 team-match file used by the RPI scripts.
`work/D.pkl` is the processed match table (built in the notebook), included so the scripts below run without rebuilding it.

## Which model produced which result
Every TrueSkill number in the talk comes from the same message-passing algorithm (Nugget 3, `n3/mp.py`; season engine `work/ts_core.py`).
- **TrueSkill, full** (results slides, live run): the 6 vs. 6 player-level model with playing-time weights, parameters `work/P_tuned.pkl` (tuned on 2024 only, `work/tune1.py`). Results: `cmp/player_results.py`; live run: `n3/live.py`.
- **TrueSkill, same information** (results slides): the same algorithm with one rating per team (the 1 vs. 1 case), this season's wins and losses only, no home court: `cmp/run.py`, `cmp/tune_fair.py`.
- **RPI**: recomputed every day from this season's results, `cmp/run.py`; rankings `cmp/rank.py`.

## Files
| Folder | What it does | Talk section |
|---|---|---|
| `notebooks/Nugget_2_TrueSkill.ipynb` | Full TrueSkill pipeline: data, model, walk-forward predictions, parameter tuning (Section 6), checks | Model and tuning |
| `work/ts_core.py` | TrueSkill engine used by the notebook and the scripts | All |
| `n3/mp.py` | Message passing on the factor graph, the closed-form update, grid integration and Monte Carlo | Nugget 3 |
| `n3/check.py` | Three-way verification on four test matches | Nugget 3, verification table |
| `n3/toy.py` | The 2 vs. 2 toy match; edit the numbers at the top | Nugget 3, toy match |
| `n3/live.py` | UTEP vs. North Carolina, Dec 4, 2025 | Nugget 3, live run |
| `rpi/rpi_check.py` | Our RPI vs. the official ranking | Introduction |
| `cmp/run.py`, `cmp/tune_fair.py` | Walk-forward RPI and TrueSkill predictions for 2024–2025 | Results |
| `cmp/player_results.py` | Results with the player-level model: accuracy, log-loss, intervals, rankings, UTEP vs. North Carolina | Results |
| `cmp/eval2.py`, `cmp/rank.py` | Accuracy, log-loss, bootstrap intervals; rank comparison | Results |
| `lec/grid.py` | Full 540-combination tuning grid | Tuning check |

## Live demo in Colab
Open `notebooks/demo_colab.ipynb` in Colab (no sign-in needed to view; sign in to run):
https://colab.research.google.com/github/JiaheLi0604/apma4903-rpi-trueskill/blob/main/notebooks/demo_colab.ipynb

Cell 1 clones this repository; the toy-match cell can be edited and rerun.

## Run
Python 3 with numpy, pandas, scipy. From the repository root:

```
python n3/check.py      # verification table
python n3/live.py       # live run (needs work/ttt.py)
python rpi/rpi_check.py # our RPI vs. official
python cmp/player_results.py   # results with the player-level model (needs cmp/rank.py run first)
```

To rebuild the season predictions from scratch: `python cmp/run.py`, then `python cmp/tune_fair.py`, then `python cmp/rank.py`, then `python cmp/player_results.py`.

## AI use
Code was drafted and debugged with help from Claude; see the AI disclosure slide in the talk.
