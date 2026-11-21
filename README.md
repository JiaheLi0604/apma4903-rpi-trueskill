# Is the RPI Fair? (APMA 4903, Fall 2026)

Code for our talk comparing TrueSkill ratings with the NCAA RPI on Division I women's volleyball (2022–2025).

## Data
Match and box-score data come from the `ncaavolleyballr` package by Jeffrey R. Stevens (MIT License), which collects results from stats.ncaa.org.
Source package: https://github.com/JeffreyRStevens/ncaavolleyballr

`data/` holds the 2025 team-match file used by the RPI scripts.
`work/D.pkl` is the processed match table (built in the notebook), included so the scripts below run without rebuilding it.

## Which model produced which result
Every TrueSkill number in the talk comes from the same message-passing algorithm (Nugget 3, `n3/mp.py`; season engine `work/ts_core.py`).
- **TrueSkill, full** (results slides, live run): the 6 vs. 6 player-level model with playing-time weights. Final model (Nov 20): a team's expected lineup counts only players who have played this season (before a team's first match of a season, last season's players), and a new player starts at her team's current average rating instead of 25 (`run_walk_forward` defaults `lineup='active', new_prior='team'`; the previous version is `lineup='all', new_prior='global'`). Parameters `work/P_tuned.pkl` = (sigma0 4, tau 1, home 3, decay 0.8, rho 1), chosen by maximum likelihood: the code computes L(theta), the log-likelihood of the 2024 results (`run_walk_forward(...)["L"]`), and a coordinate search maximizes it, repeating rounds while a round raises L (`work/tune_converge.py`, Colab section 7; 3 rounds, 94 SeasonScore runs over 62 distinct settings). Ratings run on 2022–2024; only 2024 matches are scored, each predicted before it is used; 2025 is held out for testing. The likelihood is flat near the optimum: a finer grid (`lec/grid.py`) and L-BFGS-B (`work/grad_check.py`) find slightly better 2024 scores with different settings. Results: `cmp/player_results.py` (after `cmp/rank.py`); live run: `n3/live.py`. `work/tune1.py` is an earlier fixed two-pass version for the previous model.
- **TrueSkill, same information** (results slides): the same algorithm with one rating per team (the 1 vs. 1 case), this season's wins and losses only, no home court: `cmp/run.py`, `cmp/tune_fair.py`.
- **RPI**: recomputed every day from this season's results, `cmp/run.py`; rankings `cmp/rank.py`.

## Files
| Folder | What it does | Talk section |
|---|---|---|
| `notebooks/Nugget_2_TrueSkill.ipynb` | Earlier full pipeline notebook (previous model: all past players, new players at 25); kept for reference | History |
| `work/ts_core.py` | Season engine (Algorithm 2): walks every day forward and calls Algorithm 1 (`message_passing` from `n3/mp.py`) for every match | All |
| `n3/mp.py` | Message passing on the factor graph, the closed-form update, grid integration and Monte Carlo | Nugget 3 |
| `n3/check.py` | Three-way verification on four test matches | Nugget 3, verification table |
| `n3/toy.py` | The 2 vs. 2 toy match; edit the numbers at the top | Nugget 3, toy match |
| `n3/live.py` | UTEP vs. North Carolina, Dec 4, 2025 | Nugget 3, live run |
| `rpi/rpi_check.py` | Our RPI vs. the official ranking | Introduction |
| `cmp/run.py`, `cmp/tune_fair.py` | Walk-forward RPI and TrueSkill predictions for 2024–2025 | Results |
| `cmp/player_results.py` | Results with the player-level model: accuracy, log-loss, intervals, rankings, UTEP vs. North Carolina | Results |
| `cmp/eval2.py`, `cmp/rank.py` | Accuracy, log-loss, bootstrap intervals; rank comparison | Results |
| `work/tune_converge.py` | Maximum-likelihood tuning, coordinate search looped to convergence (Algorithm 3 on the slides); `--save` writes `P_tuned.pkl` | Nugget 3, pseudocode |
| `work/grad_check.py` | Gradient-method check (L-BFGS-B) of the tuning | Tuning check |
| `lec/grid.py` | Finer 540-combination grid around the chosen settings (run from `work/`) | Tuning check |

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
