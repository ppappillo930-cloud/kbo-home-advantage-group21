"""Step 0 - build the game-level dataset (2021-2025) from the open kbo-crowd project.

Source repository : https://github.com/KENNYSOFT/kbo-crowd  (commit 1d28725, 2026-10-06)
Files used        : data/dataset.csv   (game, crowd, first-pitch weather, pre-game standings)
                    data/schedule.csv  (start time, final score, rain-outs)
                    data/stadiums.csv  (home club, dome flag, KMA station, nominal capacity)
                    data/holidays.csv  (public holidays, maintained by hand)

Usage: git clone https://github.com/KENNYSOFT/kbo-crowd && git -C kbo-crowd checkout 1d28725
       python3 00_build_games.py kbo-crowd/data ../data/kbo_games_2021_2025.csv

Rules (also in DATA_SOURCES.md)
- Seasons 2021-2025 only; 2026 was still in progress when we collected.
- Games with a final score; ties removed (home_win needs a winner).
- Doubleheaders are matched to the schedule by (date, home, away, order of the game that day).
- Weather is MEASURED, not estimated: KMA ASOS hourly observations at the city station mapped to each
  stadium, taken at the first-pitch hour; rain_game = sum of the 3 hourly rain observations from first pitch.
  Missing observations stay blank (never filled with 0).
- occupancy is ESTIMATED: crowd / nominal stadium capacity (capacity_hint), capped at 1.
- covid_capped: every 2021 game (capacity limits all season). 2022 opened at 100% capacity on 2 April 2022
  (Newsis, 2022-03-30: only vocal cheering stayed banned), so no 2022 game is flagged.
"""
import sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
ds = pd.read_csv(f"{src}/dataset.csv")
sc = pd.read_csv(f"{src}/schedule.csv")
st = pd.read_csv(f"{src}/stadiums.csv")
hol = set(pd.read_csv(f"{src}/holidays.csv").date)

sc = sc[sc.home_score.notna()].copy()
for d in (ds, sc):
    d["k"] = d.groupby(["date", "home", "away"]).cumcount()
g = ds.merge(sc[["date", "home", "away", "k", "home_score", "away_score"]], on=["date", "home", "away", "k"], how="inner")
g = g.merge(st[["stadium", "capacity_hint", "asos_stn", "asos_name"]], on="stadium", how="left")
g = g[(g.season <= 2025) & (g.home_score != g.away_score)].copy()

EN = {"키움": "Kiwoom", "삼성": "Samsung", "SSG": "SSG", "롯데": "Lotte", "KT": "KT", "한화": "Hanwha",
      "LG": "LG", "두산": "Doosan", "NC": "NC", "KIA": "KIA"}
DOW = {"월": "Mon", "화": "Tue", "수": "Wed", "목": "Thu", "금": "Fri", "토": "Sat", "일": "Sun"}
g["game_id"] = g.date.str.replace("-", "") + "_" + g.home.map(EN) + "_" + g.away.map(EN) + "_" + (g.k + 1).astype(str)
out_df = pd.DataFrame({
    "game_id": g.game_id, "date": g.date, "season": g.season, "month": g.month, "weekday": g.dow.map(DOW),
    "start_time": g.start, "home_team": g.home.map(EN), "away_team": g.away.map(EN), "stadium_kr": g.stadium,
    "home_score": g.home_score.astype(int), "away_score": g.away_score.astype(int),
    "home_win": (g.home_score > g.away_score).astype(int),
    "crowd": g.crowd, "capacity_nominal": g.capacity_hint,
    "occupancy_est": (g.crowd / g.capacity_hint).clip(upper=1).round(3),
    "dome": g.is_dome,
    "kma_station_id": g.asos_stn, "kma_station": g.asos_name,
    "temp_c_first_pitch": g.temp, "rain_mm_3h_from_first_pitch": g.rain_game,
    "humidity_pct_first_pitch": g.humid, "wind_ms_first_pitch": g.wind, "cloud_tenths_first_pitch": g.cloud,
    "weekday_holiday": g.date.isin(hol).astype(int),
    "covid_capped": (g.season == 2021).astype(int),
    "jamsil_derby": (g.home.isin(["LG", "두산"]) & g.away.isin(["LG", "두산"])).astype(int),
})
out_df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{len(out_df)} games -> {out}")
