# Is the RPI Fair? (APMA 4903, Fall 2026)

Code for our talk comparing TrueSkill ratings with the NCAA RPI on Division I women's volleyball (2022–2025).

## Data
Match and box-score data come from the `ncaavolleyballr` package by Jeffrey R. Stevens (MIT License), which collects results from stats.ncaa.org.
Source package: https://github.com/JeffreyRStevens/ncaavolleyballr

`data/` holds the 2025 team-match file used by the RPI scripts.
`work/D.pkl` is the processed match table (built in the notebook), included so the scripts below run without rebuilding it.

## Which model produced which result
- Results slides (accuracy, log-loss, rankings): team-level TrueSkill, `cmp/run.py` with `level='team'` and parameters `work/P_team.pkl`.
- Nugget 3 pseudocode and the live run: player-level TrueSkill with playing-time weights, `work/ts_core.py` with parameters `work/P_tuned.pkl`.

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
python cmp/eval2.py     # results (uses the saved predictions in cmp/)
```

To rebuild the season predictions from scratch: `python cmp/run.py`, then `python cmp/tune_fair.py`, then `python cmp/eval2.py`.

## AI use
Code was drafted and debugged with help from Claude; see the AI disclosure slide in the talk.
