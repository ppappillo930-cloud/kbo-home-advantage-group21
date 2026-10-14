"""Step 4 - one-page executive summary PDF (reportlab + matplotlib)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
_F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", _F + "DejaVuSansCondensed.ttf"))
pdfmetrics.registerFont(TTFont("DVB", _F + "DejaVuSansCondensed-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVM", _F + "DejaVuSansMono.ttf"))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DVB"); addMapping("DV", 0, 1, "DV"); addMapping("DV", 1, 1, "DVB")

res, out = sys.argv[1], sys.argv[2]
R = json.load(open(res))
HOME, AWAY, INK, GREY, WARN = "#c23a2b", "#2f5d8a", "#17212b", "#4a5560", "#b7791f"

# chart
S = R["season"]
fig, ax = plt.subplots(figsize=(7.4, 2.35), dpi=200)
ax2 = ax.twinx()
ax2.bar([s["season"] for s in S], [s["att"] for s in S], color=[WARN if s["covid"] else "#dfe4dc" for s in S], width=0.7, zorder=0)
ax2.set_ylabel("Fans / game", fontsize=7, color=GREY); ax2.tick_params(labelsize=6, colors=GREY)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
x = [s["season"] for s in S]; m = [s["mean"] * 100 for s in S]
ax.errorbar(x, m, yerr=[[(s["mean"] - s["lo"]) * 100 for s in S], [(s["hi"] - s["mean"]) * 100 for s in S]],
            fmt="o-", color=HOME, ms=3.5, lw=1.4, elinewidth=0.8, capsize=2)
ax.axhline(0, color=GREY, lw=0.6); ax.axhline(R["main"]["mean"] * 100, color=HOME, lw=0.6, ls="--")
ax.set_ylabel("Home − away WR (pts)", fontsize=7); ax.tick_params(labelsize=6)
ax.set_xticks(x); ax.set_xticklabels([str(v)[2:] for v in x])
ax.text(2020.9, max(m) - 1, "empty\nstands", ha="center", fontsize=6, color=WARN)
for sp in ["top"]: ax.spines[sp].set_visible(False); ax2.spines[sp].set_visible(False)
plt.tight_layout(); fig.savefig("/tmp/claude-0/exec_chart.png"); plt.close(fig)

ss = {
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=15.5, leading=18, textColor=colors.HexColor(INK)),
    "eb": ParagraphStyle("eb", fontName="DVM", fontSize=7, leading=9, textColor=colors.HexColor(GREY)),
    "b": ParagraphStyle("b", fontName="DV", fontSize=8.3, leading=10.8, textColor=colors.HexColor(INK)),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=9.5, leading=12, textColor=colors.HexColor(HOME), spaceBefore=3),
    "c": ParagraphStyle("c", fontName="DV", fontSize=7.4, leading=9.2, textColor=colors.HexColor(INK)),
}
P = lambda t, s="b": Paragraph(t, ss[s])
m, c20, sh, rs = R["main"], R["covid2020"], R["shared"], R["reg"]["simple"]["rows"][1]
dm = R["dml"]["att_all"]
pct = lambda v: f"{v*100:+.1f}"
fp = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"

story = [
    P("ACDT PROJECT 1 · GROUP 21 · EXECUTIVE SUMMARY · KBO 2010–2025 (152 TEAM-SEASONS)", "eb"), Spacer(1, 3),
    P("Home teams win more. The fans aren't why.", "h1"), Spacer(1, 4),
    P("<b>Conventional wisdom.</b> KBO teams play better at home, and bigger crowds make the home edge bigger. "
      "<b>H₀:</b> μ(ΔWR) = 0, where ΔWR = home win rate − away win rate per team-season. "
      "<b>H₁:</b> μ(ΔWR) &gt; 0 and the edge grows with attendance."),
    P("Key findings", "h2"),
]
rows = [["Question", "Result", "95% CI", "p", "Verdict"],
        ["Is there a home edge?", f"{pct(m['mean'])} pts", f"{pct(m['lo'])} to {pct(m['hi'])}", fp(m["p"]), "Reject H₀"],
        ["Better hitting at home? (OPS)", f"{R['splits']['dOPS']['mean']:+.3f}", f"{R['splits']['dOPS']['lo']:+.3f} to {R['splits']['dOPS']['hi']:+.3f}", fp(R['splits']['dOPS']['p']), "Yes"],
        ["Better pitching at home? (ERA)", f"{-R['splits']['dERA']['mean']:+.2f} ERA", f"{-R['splits']['dERA']['hi']:+.2f} to {-R['splits']['dERA']['lo']:+.2f}", fp(R['splits']['dERA']['p']), "Yes"],
        ["+1,000 fans → bigger edge? (OLS)", f"{rs['coef']*100:+.2f} pts", f"{rs['lo']*100:+.2f} to {rs['hi']*100:+.2f}", fp(rs["p"]), "No"],
        ["Same, Double ML (15 confounders)", f"{dm['theta']*100:+.2f} pts", f"{dm['lo']*100:+.2f} to {dm['hi']*100:+.2f}", fp(dm["p"]), "No"],
        ["Edge with empty stands (2020)?", f"{pct(c20['mean'])} pts", f"{pct(c20['lo'])} to {pct(c20['hi'])}", fp(c20["p"]), "Yes, intact"],
        ["Shared stadium (LG, Doosan) vs others", f"{pct(sh['diff'])} pts", f"{pct(sh['lo'])} to {pct(sh['hi'])}", fp(sh["p"]), "Smaller edge"],
        ["Travel distance / climate / city size", "≈ 0", "all include 0", "> 0.1", "No effect"],
        ]
t = Table(rows, colWidths=[62*mm, 26*mm, 34*mm, 15*mm, 25*mm])
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "DVB", 7.3), ("FONT", (0, 1), (-1, -1), "DV", 7.6),
    ("FONT", (1, 1), (3, -1), "DVM", 7.6),
    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor(INK)), ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#d5dbd2")),
    ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ("TEXTCOLOR", (4, 1), (4, -1), colors.HexColor(HOME)),
]))
story += [t, Spacer(1, 4), Image("/tmp/claude-0/exec_chart.png", width=180*mm, height=57*mm),
          P("League-average ΔWR by season with 95% CI (red) and average home attendance (bars; orange = COVID seasons).", "c"),
          P("What the data shows vs. conventional wisdom", "h2"),
          P(f"The home edge is real: about {m['mean']*100:.1f} win-rate points, about three more wins in 72 home games than the team's road rate would give, and {m['pct_positive']*100:.0f}% of team-seasons show it. "
            "Both batting and pitching improve at home, and together they explain "
            f"{R['mediators']['r2']*100:.0f}% of ΔWR. The crowd half of the belief fails every test: simple, controlled, within-team, occupancy split and Double ML. "
            "The 2020 natural experiment is decisive: with a few hundred fans per game, every team was still better at home."),
          P("Hidden (non-baseball) features", "h2"),
          P("After TA feedback we added stadium (capacity, dome, age, new park), geography (distance to other parks, shared stadium, capital region), city population and 1991–2020 climate normals. "
            f"The shared stadium is the one that matters: LG and Doosan play their head-to-head \"road\" games at Jamsil, and their edge is {abs(sh['diff'])*100:.1f} points smaller. "
            "Without this variable, crowd size even looks harmful, because the two biggest-crowd teams share a park. "
            f"Team-to-team differences are otherwise no larger than coin-flip luck (observed SD {R['noise']['obs_sd']:.3f} vs luck-only {R['noise']['luck_sd']:.3f}; χ² p = {R['noise']['p_excess']:.2f}), "
            f"and a random forest using all 17 features cannot predict ΔWR (cross-validated R² = {R['rf']['cv_r2']:.2f})."),
          P("Implications", "h2"),
          P("(1) Justify attendance marketing on revenue, not wins. (2) Treat home field as a near-constant 4–5 point bonus for every club; \"fortress\" reputations are mostly noise. "
            "(3) LG and Doosan carry a structural handicap at home while they share Jamsil. "
            "<b>Limits:</b> season aggregates, n = 10 clubs for team-level features, approximate capacities, climate normals not seasonal weather. "
            "<b>Next:</b> game-level logs, LG–Doosan head-to-heads, daily weather and rest days."),
          Spacer(1, 3),
          P("Data: Yagoonara/KBO home-away records & splits, KBO official attendance, Wikipedia/KMA stadium, census and climate data. Methods: paired & Welch t-tests, F and χ² variance tests, "
            "OLS with team-clustered SE, Double ML (random forests, 5-fold × 20). Code, enriched dataset and interactive webpage supplied with this submission.", "c")]

doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=14*mm, rightMargin=14*mm, topMargin=11*mm, bottomMargin=9*mm,
                        title="KBO Home Advantage - Executive Summary (Group 21)")
doc.build(story)
print("ok", out)
