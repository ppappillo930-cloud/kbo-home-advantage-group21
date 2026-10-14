# KBO Home Advantage — Group 21 (ACDT Project 1, Option 2: Interactive Webpage)

**Question:** Do KBO teams win more at home, and does a bigger crowd make that edge bigger?
**Answer:** Yes to the first (+4.5 win-rate points, p < 0.001). For the second we found no significant evidence (e.g. p = 0.40), which is not proof of zero.
The evidence points toward the ballpark: the edge survived near-empty stands in 2020 and is smaller for the two clubs that share Jamsil.

## Contents (all files sit in the repository root)
| File | What is inside |
|---|---|
| `index.html` | Interactive webpage, served by GitHub Pages: https://ppappillo930-cloud.github.io/kbo-home-advantage-group21/ |
| `Group21_Executive_Summary.pdf` | 1-page executive summary |
| `Group21_System_Maps_Report.pdf` | Three System Maps Comparison Report + Justification Document |
| `KBO_2010_2025_enriched.csv` | 152 team-seasons: home/away W-L, batting/pitching splits, attendance, stadium, travel, city, climate |
| `kbo_games_2021_2025.csv` | 3,494 decided games: game ID, date, teams, score, crowd, first-pitch weather, weekday, holiday, Jamsil derby flag |
| `kbo_team_splits_2014_2026.csv` | Cross-check dataset (MyKBO Stats home/away splits, 2026 provisional) |
| `00`–`05_*.py`, `stats_utils.py` | Python scripts; see the table below for what each one reproduces |
| `DATA_SOURCES.md` | Where each column comes from, measured vs estimated, and how it was checked |

## Reproduce
Python 3.10+, numpy, pandas, scipy, scikit-learn, matplotlib, reportlab, openpyxl. Run the scripts from the repository root.

| Script | What it reproduces | Status |
|---|---|---|
| `00_build_games.py kbo-crowd/data kbo_games_2021_2025.csv` | the 3,494-game file, from kbo-crowd at commit 1d28725 | Checked: identical output |
| `01_enrich_features.py` | adds stadium, travel, city and climate columns to the group's raw workbook | Needs `KBO_2010_2025_WITH_TRAVEL.xlsx`, which is **not** included; its output is the included `KBO_2010_2025_enriched.csv` |
| `02_analysis.py KBO_2010_2025_enriched.csv results` | every team-season number on the webpage: t-tests, Welch/F, OLS, chi-square, Double ML, random forest | Checked: all values match; Double ML and random-forest figures can differ in the 3rd decimal because random forests are random |
| `05_game_level.py kbo_games_2021_2025.csv KBO_2010_2025_enriched.csv` | every game-level number: 12 non-baseball features, Jamsil derby test with power, R² comparison | Checked: all values match the webpage |
| `03_build_page.py`, `04_exec_summary.py` | the **first** versions of the webpage and summary | Legacy. The current `index.html` and both PDFs are later hand-edited versions built from these results |

`stats_utils.py` holds the OLS (classical / HC1 / team-clustered SE), t-test and CI helpers.
22 hypothesis tests were run; the webpage's test table shows Holm-corrected p-values next to the raw ones.

## Team
Group 21: Ahn Sanghyun, Lee Junwoo, Lee Jaehyuk, Yun Younggyun, Cho Yongjae, Kim Taegyeong, Kim Sangmin.

## Data sources
KBO home/away records and splits (Yagoonara team pages, MyKBO Stats); KBO official attendance; game logs, crowds and
KMA ASOS first-pitch weather compiled by the open kbo-crowd project (github.com/KENNYSOFT/kbo-crowd);
Wikipedia / KMA stadium, census and climate data; Jeong, G. (2023), Sungkyunkwan University master's thesis.

Data are reused for an educational project with credit to their sources; copyright remains with KBO, Yagoonara, MyKBO Stats, the kbo-crowd authors, KMA and Wikipedia contributors.
