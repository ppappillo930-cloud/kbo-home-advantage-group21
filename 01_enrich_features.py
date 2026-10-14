"""
Step 1 - Add NON-BASEBALL (context) features to the KBO 2010-2025 home/away dataset.

Feature groups (all season-team level):
  Stadium      : Stadium, Capacity, Occupancy, Dome, Stadium_Age, Years_In_Park, New_Park (first 3 seasons)
  Geography    : Travel_1000km (season total travel distance, 1,000 km; from the VISUAL BASEBALL figures already in
                 the group's WITH_TRAVEL workbook, column Travel_Distance_km),
                 Shared_Stadium (LG & Doosan share Jamsil), Capital_Region (Seoul/Incheon/Gyeonggi)
  City/Climate : City, City_Pop_M (2020 census), Summer_Temp (Jun-Aug mean, 1991-2020 normal),
                 Season_Rain_mm (Apr-Oct total), Summer_Humidity (Jul-Aug mean RH)
                 Dome teams keep the climate value of their city, but an "Exposed" flag shows open-air parks.
Sources are listed in SOURCES below and written to the output workbook.
"""
import math, sys
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "KBO_2010_2025_WITH_TRAVEL.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "KBO_2010_2025_enriched.xlsx"

df = pd.read_excel(SRC, sheet_name="KBO_2010_2025")

# --- Stadium eras: (team, first_season, last_season, stadium, city, capacity, opened, team_in_park_since, dome, lat, lon)
STADIUMS = [
    ("LG",      2010, 2025, "Jamsil Baseball Stadium",        "Seoul",    25000, 1982, 1982, 0, 37.5122, 127.0719),
    ("Doosan",  2010, 2025, "Jamsil Baseball Stadium",        "Seoul",    25000, 1982, 1985, 0, 37.5122, 127.0719),
    ("Kiwoom",  2010, 2015, "Mokdong Baseball Stadium",       "Seoul",    10500, 1989, 2008, 0, 37.5307, 126.8812),
    ("Kiwoom",  2016, 2025, "Gocheok Sky Dome",               "Seoul",    16744, 2015, 2016, 1, 37.4982, 126.8671),
    ("SSG",     2010, 2025, "Incheon Munhak (SSG Landers Field)", "Incheon", 23000, 2002, 2002, 0, 37.4370, 126.6932),
    ("KT",      2015, 2025, "Suwon KT Wiz Park",              "Suwon",    20000, 1989, 2015, 0, 37.2997, 127.0097),
    ("Samsung", 2010, 2015, "Daegu Baseball Stadium",         "Daegu",    10000, 1948, 1982, 0, 35.8810, 128.5866),
    ("Samsung", 2016, 2025, "Daegu Samsung Lions Park",       "Daegu",    24000, 2016, 2016, 0, 35.8411, 128.6815),
    ("KIA",     2010, 2013, "Gwangju Mudeung Stadium",        "Gwangju",  12500, 1965, 1982, 0, 35.1700, 126.8880),
    ("KIA",     2014, 2025, "Gwangju-Kia Champions Field",    "Gwangju",  20500, 2014, 2014, 0, 35.1682, 126.8890),
    ("Lotte",   2010, 2025, "Busan Sajik Baseball Stadium",   "Busan",    24500, 1985, 1985, 0, 35.1940, 129.0615),
    ("Hanwha",  2010, 2024, "Daejeon Hanbat Baseball Stadium","Daejeon",  13000, 1964, 1986, 0, 36.3171, 127.4291),
    ("Hanwha",  2025, 2025, "Daejeon Hanwha Life Ballpark",   "Daejeon",  20000, 2025, 2025, 0, 36.3165, 127.4275),
    ("NC",      2013, 2018, "Masan Baseball Stadium",         "Changwon", 11000, 1982, 2013, 0, 35.2225, 128.5822),
    ("NC",      2019, 2025, "Changwon NC Park",               "Changwon", 22112, 2019, 2019, 0, 35.2225, 128.5822),
]
# 1991-2020 climate normals (Wikipedia city climate tables, KMA data): Jun-Aug temp, Apr-Oct rain, Jul-Aug RH
CLIMATE = {  # city: (Apr..Oct mean temps, Apr..Oct precip, Apr..Oct RH)
    "Seoul":    ([12.6,18.2,22.7,25.3,26.1,21.7,15.0], [72.9,103.6,129.5,414.4,348.2,141.5,52.2], [54.8,59.7,65.7,76.2,73.5,66.4,61.8]),
    "Incheon":  ([11.5,16.8,21.3,24.4,25.6,21.5,15.3], [63.5,96.3,106.0,337.7,274.6,130.3,51.1], [64.5,69.8,75.9,83.8,80.1,72.9,66.7]),
    "Suwon":    ([12.0,17.6,22.2,25.3,26.0,21.4,14.6], [71.6,95.0,122.9,385.1,296.3,133.5,54.1], [62.1,66.1,71.4,79.9,77.6,73.2,69.8]),
    "Daejeon":  ([13.0,18.5,22.7,25.5,26.0,21.2,14.6], [81.6,91.8,167.3,306.9,299.8,152.5,59.3], [56.9,62.4,69.8,79.1,78.2,75.3,71.8]),
    "Daegu":    ([14.5,19.7,23.4,26.3,26.7,22.1,16.2], [70.6,77.9,129.2,223.9,245.3,142.4,50.1], [50.8,55.8,63.8,71.7,72.2,69.3,62.8]),
    "Gwangju":  ([13.4,18.7,22.7,25.9,26.5,22.2,16.1], [86.6,91.4,152.6,294.2,326.4,145.0,59.0], [60.2,64.5,72.0,79.8,78.0,73.6,67.6]),
    "Busan":    ([13.8,17.9,21.0,24.4,26.1,22.6,17.9], [140.9,155.9,188.4,326.8,266.5,160.6,79.6], [61.1,68.3,76.8,83.4,78.5,72.6,62.7]),
    "Changwon": ([13.8,18.5,21.8,25.3,26.5,22.6,17.3], [133.8,146.4,190.7,304.5,238.8,172.9,72.3], [59.3,65.7,74.6,81.0,77.1,71.4,62.5]),
}
POP_2020 = {"Seoul": 9586195, "Busan": 3349016, "Incheon": 2945454, "Daegu": 2410700, "Daejeon": 1488435,
            "Gwangju": 1477573, "Suwon": 1210150, "Changwon": 1029389}
CAPITAL = {"Seoul", "Incheon", "Suwon"}


def hav(a, b):
    (la1, lo1), (la2, lo2) = a, b
    R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def era(team, season):
    for t, s0, s1, *rest in STADIUMS:
        if t == team and s0 <= season <= s1:
            return rest
    raise KeyError((team, season))


rows = []
for _, r in df.iterrows():
    stad, city, cap, opened, since, dome, lat, lon = era(r.Team, r.Season)
    t, p, h = CLIMATE[city]
    rows.append(dict(
        Stadium=stad, City=city, Capacity=cap,
        Occupancy=r.Attendance / cap,
        Dome=dome, Stadium_Age=r.Season - opened,
        Years_In_Park=r.Season - since + 1, New_Park=int(r.Season - since + 1 <= 3),
        Travel_1000km=r.Travel_Distance_km / 1000,
        Shared_Stadium=int(r.Team in ("LG", "Doosan")),
        Capital_Region=int(city in CAPITAL),
        City_Pop_M=POP_2020[city] / 1e6,
        Summer_Temp=float(np.mean(t[2:5])), Season_Rain_mm=float(np.sum(p)),
        Summer_Humidity=float(np.mean(h[3:5])),
        Open_Air=1 - dome,
        Lat=lat, Lon=lon,
    ))
feat = pd.DataFrame(rows, index=df.index)
out = pd.concat([df, feat], axis=1)

# drop all-empty placeholder columns that were never populated (kept in the original file)
empty = [c for c in out.columns if out[c].isna().all()]
out = out.drop(columns=empty)

SOURCES = pd.DataFrame([
    ("Home/Away W-L, attendance, home/away batting & pitching splits", "Original group dataset (Yagoonara / KBO official / MyKBO Stats)", "see original Sources sheet"),
    ("Stadium names, capacity, opening year, dome", "Wikipedia - List of baseball stadiums in South Korea (capacities are current nominal values; historical re-seating not modelled)", "https://en.wikipedia.org/wiki/List_of_baseball_stadiums_in_South_Korea"),
    ("Old Daegu (10,000) and Mudeung (12,500) capacities", "Commonly reported era capacities; approximate", "-"),
    ("Travel_Distance_km -> Travel_1000km", "VISUAL BASEBALL team travel tables (visualbaseball.com), transcribed by the group from screenshots; counting basis not confirmed", "visualbaseball.com"),
    ("Stadium coordinates (Lat, Lon)", "Approximate stadium locations (not used in the analysis)", "-"),
    ("City population", "2020 Population and Housing Census (via Wikipedia 'List of cities in South Korea')", "https://en.wikipedia.org/wiki/List_of_cities_in_South_Korea"),
    ("Climate normals (temp, precipitation, humidity)", "KMA 1991-2020 normals via each city's Wikipedia climate table", "https://en.wikipedia.org/wiki/Seoul (and Busan, Incheon, Daegu, Daejeon, Gwangju, Suwon, Changwon)"),
], columns=["Field(s)", "Source", "URL"])

GUIDE = pd.DataFrame([
    ("Occupancy", "Attendance / Capacity (approximate; capacity is the era's nominal seat count)"),
    ("Dome", "1 = Gocheok Sky Dome (Kiwoom 2016+)"),
    ("Stadium_Age", "Season - stadium opening year"),
    ("Years_In_Park", "Seasons the team has played in this park (incl. current)"),
    ("New_Park", "1 if Years_In_Park <= 3 (honeymoon / unfamiliar park)"),
    ("Travel_1000km", "Season total road-travel distance in thousands of km (= Travel_Distance_km / 1000)"),
    ("Shared_Stadium", "1 = LG / Doosan (share Jamsil; their head-to-head 'away' games are at home)"),
    ("Capital_Region", "1 = Seoul, Incheon, Suwon"),
    ("City_Pop_M", "Home-city population, millions (2020 census)"),
    ("Summer_Temp", "Jun-Aug mean daily temperature, C (1991-2020 normal)"),
    ("Season_Rain_mm", "Apr-Oct total precipitation, mm (1991-2020 normal)"),
    ("Summer_Humidity", "Jul-Aug mean relative humidity, % (1991-2020 normal)"),
], columns=["Variable", "Meaning"])

with pd.ExcelWriter(OUT) as w:
    out.to_excel(w, sheet_name="KBO_2010_2025_enriched", index=False)
    GUIDE.to_excel(w, sheet_name="New_Variable_Guide", index=False)
    SOURCES.to_excel(w, sheet_name="New_Sources", index=False)
out.to_csv(OUT.replace(".xlsx", ".csv"), index=False)
print(out.shape)
print(out.groupby("Team")[["Capacity", "Occupancy", "Travel_1000km", "Summer_Temp", "City_Pop_M"]].mean().round(2))
print("max occupancy:", out.Occupancy.max().round(2), out.loc[out.Occupancy.idxmax(), ["Season", "Team", "Attendance"]].tolist())
