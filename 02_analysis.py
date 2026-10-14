"""
Step 2 - Full analysis for the KBO home-advantage project (Group 21).
Outputs ../results/results.json (feeds the interactive webpage) and prints a readable log.

  Map A  (statistical baseline) : paired t-test / CI on dWR, offence & pitching splits, attendance regressions
  Map B  (hidden / context)     : non-baseball features - stadium, geography, climate, city; COVID natural experiment
  Map C  (ML)                   : Double/Debiased ML (partialling-out, 5-fold cross-fitting, random forests)
                                  + random-forest predictability check with permutation importance
Run:  python3 02_analysis.py ../data_enriched.csv ../results
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import one_sample, welch, ols, mean_ci  # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else "../data_enriched.csv"
OUTD = sys.argv[2] if len(sys.argv) > 2 else "../results"
os.makedirs(OUTD, exist_ok=True)
d = pd.read_csv(DATA)
nc = d[d.COVID_period == 0].reset_index(drop=True)
R = {}


def fl(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def coef_rows(o):
    return [dict(term=i, coef=r.coef, se=r.se, t=r.t, p=r.p, lo=r.lo, hi=r.hi) for i, r in o["table"].iterrows()]


# ---------------------------------------------------------------- MAP A: Does home advantage exist?
tests = []

def add_test(id_, group, name, h0, method, res, stat_name, stat, p, est, lo, hi, unit, n, note=""):
    tests.append(dict(id=id_, group=group, name=name, h0=h0, method=method, stat_name=stat_name, stat=stat, p=p,
                      est=est, lo=lo, hi=hi, unit=unit, n=n, reject=bool(p < 0.05), note=note))

main = one_sample(d.dWR)
main_nc = one_sample(nc.dWR)
R["main"] = main
R["main_noncovid"] = main_nc
add_test("T1", "A", "Home win rate > away win rate (all seasons)", "mean dWR = 0", "Paired t-test (one-tailed)",
         main, "t", main["t"], main["p"], main["mean"], main["lo"], main["hi"], "WR points", main["n"],
         f"Wilcoxon p={main['wilcoxon_p']:.1e}; Shapiro p={main['shapiro_p']:.2f}; {main['pct_positive']*100:.0f}% of team-seasons positive")
add_test("T2", "A", "Home advantage, excluding COVID seasons", "mean dWR = 0", "Paired t-test (one-tailed)",
         main_nc, "t", main_nc["t"], main_nc["p"], main_nc["mean"], main_nc["lo"], main_nc["hi"], "WR points", main_nc["n"])

split_labels = {"dAVG": "Batting average (home - away)", "dOBP": "On-base % (home - away)",
                "dSLG": "Slugging % (home - away)", "dOPS": "OPS (home - away)",
                "dERA": "ERA (away - home; + = better at home)", "dWHIP": "WHIP (away - home; + = better at home)"}
R["splits"] = {}
for k, lab in split_labels.items():
    r = one_sample(d[k])
    R["splits"][k] = r
    add_test(f"S_{k}", "A", lab, f"mean {k} = 0", "One-sample t-test (one-tailed)", r, "t", r["t"], r["p"],
             r["mean"], r["lo"], r["hi"], k, r["n"], f"{r['pct_positive']*100:.0f}% positive")

# attendance (original H1 second half)
o_simple = ols(nc.dWR, nc[["Attendance_k"]], ["Attendance_k"])
o_simple_all = ols(d.dWR, d[["Attendance_k"]], ["Attendance_k"])
mA_vars = ["Attendance_k", "Overall_WR", "Season_Index"]
o_multi = ols(nc.dWR, nc[mA_vars], mA_vars, cluster=nc.Team)
w = nc.copy()
for c in ["dWR", "Attendance_k"]:
    w[c + "_w"] = w[c] - w.groupby("Team")[c].transform("mean")
o_within = ols(w.dWR_w, w[["Attendance_k_w"]], ["Attendance_k (within team)"], add_const=False, cluster=w.Team)
R["reg"] = {
    "simple": dict(n=o_simple["n"], r2=o_simple["r2"], rows=coef_rows(o_simple), se=o_simple["se_type"]),
    "simple_all": dict(n=o_simple_all["n"], r2=o_simple_all["r2"], rows=coef_rows(o_simple_all), se=o_simple_all["se_type"]),
    "multi": dict(n=o_multi["n"], r2=o_multi["r2"], adj_r2=o_multi["adj_r2"], rows=coef_rows(o_multi), se=o_multi["se_type"]),
    "within": dict(n=o_within["n"], r2=o_within["r2"], rows=coef_rows(o_within), se=o_within["se_type"]),
}
b = o_simple["table"].loc["Attendance_k"]
add_test("R1", "A", "Bigger crowds -> bigger home advantage (simple regression)", "beta(Attendance) = 0",
         "OLS dWR ~ Attendance (thousands), non-COVID", None, "t", b.t, b.p, b.coef, b.lo, b.hi, "per 1,000 fans",
         o_simple["n"], f"R2 = {o_simple['r2']:.3f}")
b = o_multi["table"].loc["Attendance_k"]
add_test("R2", "A", "Crowd effect after controlling team quality & season", "beta(Attendance) = 0",
         "Multiple OLS, team-clustered SE", None, "t", b.t, b.p, b.coef, b.lo, b.hi, "per 1,000 fans",
         o_multi["n"], f"R2 = {o_multi['r2']:.3f}")
b = o_within["table"].iloc[0]
add_test("R3", "A", "Same team, more fans than its usual -> bigger edge? (within-team)", "beta(within) = 0",
         "Team-demeaned OLS, clustered SE", None, "t", b.t, b.p, b.coef, b.lo, b.hi, "per 1,000 fans",
         o_within["n"])

hi_ = nc[nc.Occupancy > 0.7].dWR
lo_ = nc[nc.Occupancy <= 0.7].dWR
occ = welch(hi_, lo_)
R["occupancy_split"] = occ
add_test("W1", "A", "High (>70%) vs low occupancy parks (Week 3 plan)", "mean dWR equal",
         "Welch t-test + F-test", occ, "t", occ["t"], occ["p"], occ["diff"], occ["lo"], occ["hi"], "WR points",
         occ["n_a"] + occ["n_b"], f"F = {occ['F']:.2f} (p={occ['F_p']:.2f}); n_high={occ['n_a']}, n_low={occ['n_b']}")

# mediators: how is the edge produced?
o_med = ols(d.dWR, d[["dOPS", "dERA"]], ["dOPS", "dERA"])
R["mediators"] = dict(n=o_med["n"], r2=o_med["r2"], rows=coef_rows(o_med))

# ---------------------------------------------------------------- MAP B: COVID natural experiment + context features
cov20 = d[d.Season == 2020].dWR
r20 = one_sample(cov20)
R["covid2020"] = r20
add_test("C1", "B", "Empty stadiums (2020): is there still a home edge?", "mean dWR(2020) = 0",
         "One-sample t-test (one-tailed)", r20, "t", r20["t"], r20["p"], r20["mean"], r20["lo"], r20["hi"],
         "WR points", r20["n"], f"Avg attendance {d[d.Season==2020].Attendance.mean():.0f}; {r20['pct_positive']*100:.0f}% of teams positive")
cw = welch(cov20, nc.dWR)
R["covid_vs_normal"] = cw
add_test("C2", "B", "Empty-stadium 2020 vs normal seasons", "mean dWR equal", "Welch t-test", cw, "t", cw["t"],
         cw["p"], cw["diff"], cw["lo"], cw["hi"], "WR points", cw["n_a"] + cw["n_b"],
         "If fans caused the edge, 2020 should be LOWER - it is not")

shared = welch(d[d.Shared_Stadium == 1].dWR, d[d.Shared_Stadium == 0].dWR)
R["shared"] = shared
add_test("B1", "B", "Shared stadium (LG & Doosan at Jamsil) vs everyone else", "mean dWR equal", "Welch t-test",
         shared, "t", shared["t"], shared["p"], shared["diff"], shared["lo"], shared["hi"], "WR points",
         shared["n_a"] + shared["n_b"], "Their head-to-head 'away' games are played in their own park")

dome = welch(d[d.Dome == 1].dWR, d[d.Dome == 0].dWR)
add_test("B2", "B", "Dome (Gocheok, Kiwoom 2016+) vs open-air", "mean dWR equal", "Welch t-test", dome, "t",
         dome["t"], dome["p"], dome["diff"], dome["lo"], dome["hi"], "WR points", dome["n_a"] + dome["n_b"],
         "Only one dome team - cannot separate dome from Kiwoom")
newp = welch(d[d.New_Park == 1].dWR, d[d.New_Park == 0].dWR)
add_test("B3", "B", "First 3 seasons in a new park vs established", "mean dWR equal", "Welch t-test", newp, "t",
         newp["t"], newp["p"], newp["diff"], newp["lo"], newp["hi"], "WR points", newp["n_a"] + newp["n_b"])
cap = welch(d[d.Capital_Region == 1].dWR, d[d.Capital_Region == 0].dWR)
add_test("B4", "B", "Capital-region teams (short trips) vs regional teams", "mean dWR equal", "Welch t-test", cap,
         "t", cap["t"], cap["p"], cap["diff"], cap["lo"], cap["hi"], "WR points", cap["n_a"] + cap["n_b"])

FEATS = {  # name: (label, domain, unit)
    "Attendance_k": ("Avg attendance (k)", "Crowd", "per 1k fans"),
    "Occupancy": ("Occupancy rate", "Crowd", "per 100%"),
    "Travel_1000km": ("Season travel distance (1,000 km)", "Geography", "per 1,000 km"),
    "Shared_Stadium": ("Shared stadium", "Geography", "yes vs no"),
    "Capital_Region": ("Capital region", "Geography", "yes vs no"),
    "Dome": ("Dome", "Stadium", "yes vs no"),
    "New_Park": ("New park (<=3 yrs)", "Stadium", "yes vs no"),
    "Stadium_Age": ("Stadium age", "Stadium", "per year"),
    "Capacity": ("Capacity (seats)", "Stadium", "per seat"),
    "City_Pop_M": ("City population (M)", "City", "per million"),
    "Summer_Temp": ("Summer temperature", "Climate", "per deg C"),
    "Season_Rain_mm": ("Season rainfall", "Climate", "per mm"),
    "Summer_Humidity": ("Summer humidity", "Climate", "per % RH"),
    "Overall_WR": ("Team quality (overall WR)", "Baseball control", "per 1.0 WR"),
    "Season_Index": ("Season trend", "Time", "per season"),
    "COVID_period": ("COVID seasons", "Time", "yes vs no"),
}
screen = []
for f, (lab, dom, unit) in FEATS.items():
    o = ols(d.dWR, d[[f]], [f], cluster=d.Team)
    t = o["table"].loc[f]
    sd = d[f].std()
    r, rp = stats.pearsonr(d[f], d.dWR)
    screen.append(dict(feature=f, label=lab, domain=dom, unit=unit, coef=t.coef, lo=t.lo, hi=t.hi, p=t.p,
                       r=r, r_p=rp, r2=o["r2"], std_effect=t.coef * sd, std_lo=t.lo * sd, std_hi=t.hi * sd,
                       non_baseball=dom in ("Geography", "Stadium", "City", "Climate", "Crowd")))
R["screen"] = screen

mB_vars = ["Attendance_k", "Overall_WR", "Season_Index", "COVID_period", "Shared_Stadium", "Travel_1000km",
           "New_Park", "Stadium_Age", "Summer_Temp", "City_Pop_M"]
o_mB = ols(d.dWR, d[mB_vars], mB_vars, cluster=d.Team)
R["reg"]["mapB"] = dict(n=o_mB["n"], r2=o_mB["r2"], adj_r2=o_mB["adj_r2"], rows=coef_rows(o_mB), se=o_mB["se_type"])

# noise ceiling: how much team-to-team variation in dWR would pure binomial luck create?
nh = d.Home_W + d.Home_L
na = d.Away_W + d.Away_L
p = d.Overall_WR
bvar = p * (1 - p) * (1 / nh + 1 / na)
obs_var = d.dWR.var(ddof=1)
chi = (((d.dWR - d.dWR.mean()) ** 2) / bvar).sum()
R["noise"] = dict(obs_var=obs_var, obs_sd=np.sqrt(obs_var), luck_var=bvar.mean(), luck_sd=np.sqrt(bvar.mean()),
                  ratio=obs_var / bvar.mean(), chi2=chi, df=len(d) - 1, p_excess=stats.chi2.sf(chi, len(d) - 1))
add_test("N1", "B", "Do teams truly differ in home edge, beyond luck?", "var(dWR) = binomial luck variance",
         "Chi-square test for variance (one-tailed, excess)", None, "chi2", chi, R["noise"]["p_excess"],
         obs_var / bvar.mean(), None, None, "obs/luck variance ratio", len(d),
         f"Observed SD {np.sqrt(obs_var):.3f} vs luck-only SD {np.sqrt(bvar.mean()):.3f}")

# ---------------------------------------------------------------- MAP C: Double ML + RF predictability
W_ALL = ["Overall_WR", "Season_Index", "League_Team_Count", "COVID_period", "Shared_Stadium", "Capital_Region",
         "Dome", "New_Park", "Stadium_Age", "Travel_1000km", "City_Pop_M", "Summer_Temp", "Season_Rain_mm",
         "Summer_Humidity", "Capacity"]


def dml_plr(df, T, W, Y="dWR", reps=20, folds=5):
    th, ses = [], []
    for rep in range(reps):
        kf = KFold(folds, shuffle=True, random_state=rep)
        yt, tt = np.zeros(len(df)), np.zeros(len(df))
        for tr, te in kf.split(df):
            g = RandomForestRegressor(300, min_samples_leaf=5, random_state=rep).fit(df[W].iloc[tr], df[Y].iloc[tr])
            m = RandomForestRegressor(300, min_samples_leaf=5, random_state=rep).fit(df[W].iloc[tr], df[T].iloc[tr])
            yt[te] = df[Y].values[te] - g.predict(df[W].iloc[te])
            tt[te] = df[T].values[te] - m.predict(df[W].iloc[te])
        theta = (tt @ yt) / (tt @ tt)
        e = yt - theta * tt
        se = np.sqrt(np.mean(tt ** 2 * e ** 2)) / np.mean(tt ** 2) / np.sqrt(len(df))
        th.append(theta)
        ses.append(se)
    th, ses = np.array(th), np.array(ses)
    med = np.median(th)
    se = np.sqrt(np.median(ses ** 2 + (th - med) ** 2))  # DoubleML's median aggregation
    return dict(theta=med, se=se, lo=med - 1.96 * se, hi=med + 1.96 * se, p=2 * stats.norm.sf(abs(med / se)),
                n=len(df), reps=reps, folds=folds, last_resid_T=tt.tolist(), last_resid_Y=yt.tolist())


dml = {}
dml["att_all"] = dml_plr(d, "Attendance_k", [w for w in W_ALL])
dml["att_noncovid"] = dml_plr(nc, "Attendance_k", [w for w in W_ALL if w != "COVID_period"])
dml["occ_all"] = dml_plr(d, "Occupancy", W_ALL)
dml["travel"] = dml_plr(d, "Travel_1000km", [w for w in W_ALL if w not in ("Travel_1000km", "Capital_Region")] + ["Attendance_k"])
R["dml"] = dml
for k, lab in [("att_all", "Crowd effect (Double ML, all seasons)"), ("att_noncovid", "Crowd effect (Double ML, non-COVID)")]:
    x = dml[k]
    add_test(f"D_{k}", "C", lab, "theta(Attendance) = 0", "DML partially-linear, RF nuisance, 5-fold x 20",
             None, "z", x["theta"] / x["se"], x["p"], x["theta"], x["lo"], x["hi"], "per 1,000 fans", x["n"])

# RF predictability: can ANY combination of features predict a team's home edge?
X_rf = d[W_ALL + ["Attendance_k", "Occupancy"]]
rf = RandomForestRegressor(500, min_samples_leaf=5, random_state=0)
cv_pred = cross_val_predict(rf, X_rf, d.dWR, cv=KFold(5, shuffle=True, random_state=0))
cv_r2 = 1 - ((d.dWR - cv_pred) ** 2).sum() / ((d.dWR - d.dWR.mean()) ** 2).sum()
rf.fit(X_rf, d.dWR)
pi = permutation_importance(rf, X_rf, d.dWR, n_repeats=30, random_state=0)
R["rf"] = dict(cv_r2=cv_r2, in_sample_r2=rf.score(X_rf, d.dWR),
               importance=sorted([dict(feature=c, mean=m, sd=s) for c, m, s in
                                  zip(X_rf.columns, pi.importances_mean, pi.importances_std)],
                                 key=lambda r: -r["mean"]))

# ---------------------------------------------------------------- chart data
season = []
for s, g in d.groupby("Season"):
    c = mean_ci(g.dWR)
    season.append(dict(season=int(s), n=c["n"], mean=c["mean"], lo=c["lo"], hi=c["hi"],
                       att=g.Attendance.mean(), covid=int(g.COVID_period.max()),
                       home_wr=g.Home_WR.mean(), away_wr=g.Away_WR.mean()))
team = []
for t, g in d.groupby("Team"):
    c = mean_ci(g.dWR)
    team.append(dict(team=t, n=c["n"], mean=c["mean"], lo=c["lo"], hi=c["hi"], att=g.Attendance.mean(),
                     travel=g.Travel_Distance_km.mean(), shared=int(g.Shared_Stadium.max()), city=g.City.iloc[0]))
R["season"] = season
R["team"] = team
keep = ["Season", "Team", "Home_W", "Home_L", "Away_W", "Away_L", "Home_WR", "Away_WR", "dWR", "Overall_WR",
        "Attendance", "Occupancy", "COVID_period", "Stadium", "City", "Capacity", "Dome", "New_Park",
        "Stadium_Age", "Travel_Distance_km", "Shared_Stadium", "Capital_Region", "City_Pop_M", "Summer_Temp",
        "Season_Rain_mm", "Summer_Humidity", "dAVG", "dOBP", "dSLG", "dOPS", "dERA", "dWHIP"]
R["rows"] = d[keep].to_dict(orient="records")
R["tests"] = tests

with open(os.path.join(OUTD, "results.json"), "w") as f:
    json.dump(clean(R), f)

# ---------------------------------------------------------------- log
pd.set_option("display.width", 200)
print(pd.DataFrame(tests)[["id", "name", "est", "lo", "hi", "stat", "p", "reject"]].round(4).to_string())
print("\nMap A multiple:\n", o_multi["table"].round(4), o_multi["r2"])
print("\nMap B multiple:\n", o_mB["table"].round(4), o_mB["r2"])
print("\nnoise:", {k: round(v, 4) for k, v in R["noise"].items()})
print("\nDML:", {k: {kk: round(v[kk], 5) for kk in ("theta", "se", "lo", "hi", "p")} for k, v in dml.items()})
print("\nRF cv R2:", round(cv_r2, 3), "in-sample", round(R["rf"]["in_sample_r2"], 3))
print(pd.DataFrame(R["rf"]["importance"]).round(5).head(8))
print(pd.DataFrame(screen)[["feature", "coef", "lo", "hi", "p", "r"]].round(4))
