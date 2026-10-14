# Data sources and provenance

| File | Unit | Source | Measured or estimated | How it was checked |
|---|---|---|---|---|
| `data/KBO_2010_2025_enriched.csv` – W, L, T (home/away) | team-season, 2010–2025 | Yagoonara team pages (KBO records) | Recorded | 2014–2025 cross-checked against MyKBO Stats and Wikipedia season pages: 119/119 team-seasons match |
| same file – attendance | team-season | KBO historical attendance | Recorded | 2023–2025 matched Wikipedia / KBO year-end figures |
| same file – home/away batting and pitching splits | team-season | Yagoonara team pages (see `Metric_Source`, `Split_Source_URL`) | as labelled in file | — |
| same file – travel distance | team-season | VISUAL BASEBALL screenshots, transcribed | Transcribed | counting basis not documented by the source |
| same file – stadium, city, climate | team / season | Wikipedia, KMA 1991–2020 normals, 2020 census | Reference values | capacities are nominal |
| `data/kbo_games_2021_2025.csv` | game, 3,494 decided games | kbo-crowd (github.com/KENNYSOFT/kbo-crowd, commit 1d28725), built from KBO daily crowd records and the KBO schedule | Recorded | every row has `game_id`, date, both teams and final score; season totals of home wins match MyKBO Stats team splits (2022 SSG–Lotte gap resolved with this file) |
| same file – weather | game | KMA ASOS hourly observations at the city station mapped to each stadium (`kma_station`) | **Measured** at the first-pitch hour; rain = sum of 3 hourly readings from first pitch; missing readings left blank | — |
| same file – `occupancy_est` | game | crowd ÷ nominal capacity | **Estimated** | capped at 1 |
| same file – `covid_capped` | game | KBO / government rules | Our flag | all 2021 games; 2022 opened at 100% capacity (Newsis, 2022-03-30) |
| `data/kbo_team_splits_2014_2026.csv` | team-season | MyKBO Stats team splits | Recorded | used for cross-checks; 2026 provisional |

Rebuild the game file: `git clone https://github.com/KENNYSOFT/kbo-crowd && git -C kbo-crowd checkout 1d28725`
then `python3 analysis/00_build_games.py kbo-crowd/data data/kbo_games_2021_2025.csv`.
