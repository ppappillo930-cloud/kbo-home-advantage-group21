"""Step 3 - inject results.json and the code/data files into the webpage template -> index.html
usage: python3 03_build_page.py results.json webpage_template.html index.html [data.csv] [code_dir]"""
import json, os, sys
res, tpl, out = sys.argv[1:4]
csv = sys.argv[4] if len(sys.argv) > 4 else None
code_dir = sys.argv[5] if len(sys.argv) > 5 else os.path.dirname(os.path.abspath(__file__))
R = json.load(open(res))
for v in R["dml"].values():
    v.pop("last_resid_T", None); v.pop("last_resid_Y", None)
DESC = {
    "README.md": "Run order and requirements.",
    "01_enrich_features.py": "Adds the non-baseball features (stadium, travel, city, climate) to the group dataset.",
    "02_analysis.py": "All tests behind Maps A, B and C: t-tests, Welch/F, OLS with clustered SE, chi-square variance test, Double ML, random forest.",
    "stats_utils.py": "Helper functions: mean CI, one-sample and Welch tests, OLS with classical / HC1 / cluster-robust SE.",
    "03_build_page.py": "Builds this webpage from the template, the results and these files.",
    "04_exec_summary.py": "Builds the one-page executive summary PDF.",
}
files = []
if csv:
    files.append(dict(name=os.path.basename(csv), kind="csv", text=open(csv, encoding="utf-8").read(),
                      desc="Analysis dataset: 152 team-seasons. Original home/away records, attendance and splits (Yagoonara / KBO / MyKBO Stats), "
                           "season travel distance (VISUAL BASEBALL), plus stadium, city and climate features added for this project."))
for n in ["README.md", "01_enrich_features.py", "02_analysis.py", "stats_utils.py", "03_build_page.py", "04_exec_summary.py"]:
    fp = os.path.join(code_dir, n)
    if os.path.exists(fp):
        files.append(dict(name=n, kind="md" if n.endswith(".md") else "py", text=open(fp, encoding="utf-8").read(), desc=DESC[n]))
html = open(tpl, encoding="utf-8").read()
html = html.replace("/*__DATA__*/null", json.dumps(R, separators=(",", ":")))
html = html.replace("/*__FILES__*/null", json.dumps(files, ensure_ascii=False).replace("</", "<\\/"))
open(out, "w", encoding="utf-8").write(html)
print(out, len(html) // 1024, "KB", len(files), "files")
