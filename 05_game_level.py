"""Step 5 - game-level checks (2021-2025): home win rate, crowd, 12 non-baseball features,
Jamsil derby test, and which X best predicts the team-season home edge.

Usage: python3 05_game_level.py ../data/kbo_games_2021_2025.csv ../data/KBO_2010_2025_enriched.csv
"""
import sys
import numpy as np, pandas as pd
from scipy import stats

games = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
teams = pd.read_csv(sys.argv[2])
y = games.home_win

# 1. home win rate (ties already removed)
n, p = len(games), y.mean()
z = (p - 0.5) / np.sqrt(0.25 / n)
print(f"Home win rate {p:.4f} (n={n}), one-sided p={1-stats.norm.cdf(z):.4f}")

# 2. non-baseball features: chi-square across groups
bins = {
    # occupancy from raw crowd / nominal capacity (not the rounded occupancy_est column), same as the webpage
    "occupancy": pd.cut((games.crowd / games.capacity_nominal).clip(0, 1), [0, .1, .3, .5, .7, .9, 1.0001], include_lowest=True, right=False),
    "temp_c": pd.cut(games.temp_c_first_pitch, [-1, 15, 20, 25, 30, 40]),
    "rain_mm": pd.cut(games.rain_mm_3h_from_first_pitch, [-0.1, 0, 1, 100]),
    "humidity": pd.cut(games.humidity_pct_first_pitch, [0, 40, 55, 70, 85, 101]),
    "wind_ms": pd.cut(games.wind_ms_first_pitch, [-0.1, 1.5, 2.5, 3.5, 20]),
    "cloud": pd.cut(games.cloud_tenths_first_pitch, [-0.1, 2, 5, 8, 10]),
    "weekday": games.weekday, "month": games.month, "start_hour": games.start_time.str[:2].astype(int),
    "weekday_holiday": games.weekday_holiday, "dome": games.dome, "season": games.season,
}
print("\nFeature            chi2    df   p")
for k, b in bins.items():
    ok = b.notna()
    chi, pv, df, _ = stats.chi2_contingency(pd.crosstab(b[ok], y[ok]))
    print(f"{k:16s} {chi:7.2f} {df:4d} {pv:7.4f}")

# 3. Jamsil derby vs LG/Doosan hosting other clubs at Jamsil
der = y[games.jamsil_derby == 1]
jm = y[(games.stadium_kr == "잠실") & (games.jamsil_derby == 0)]
p1, p2 = der.mean(), jm.mean()
pp = (der.sum() + jm.sum()) / (len(der) + len(jm))
zz = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / len(der) + 1 / len(jm)))
se = np.sqrt(p1 * (1 - p1) / len(der) + p2 * (1 - p2) / len(jm))
print(f"\nJamsil derby: {p1:.4f} (n={len(der)}) vs other Jamsil home games {p2:.4f} (n={len(jm)})")
print(f"  diff {p1-p2:+.4f}, 95% CI [{p1-p2-1.96*se:+.4f}, {p1-p2+1.96*se:+.4f}], z={zz:.2f}, p={2*(1-stats.norm.cdf(abs(zz))):.4f}")
need = (1.96 + 0.8416) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
print(f"  games per group for 80% power: {need:.0f} (~{need/16:.0f} seasons of derbies)")

# 4. which X predicts the team-season home edge (non-COVID seasons)
t = teams[teams.COVID_period == 0]
print("\nPredictor        R2      p")
for x in ["Attendance", "Occupancy", "Overall_WR", "dERA", "dOPS"]:
    r = stats.linregress(t[x], t.dWR)
    print(f"{x:14s} {r.rvalue**2:6.3f} {r.pvalue:8.4f}")
