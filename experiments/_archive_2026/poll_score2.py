import subprocess, time, csv, io
COMP="birdclef-2026"
for _ in range(50):  # ~100 min
    p=subprocess.run(["uv","run","kaggle","competitions","submissions","-c",COMP,"--csv"],capture_output=True,text=True)
    rows=[r for r in csv.reader(io.StringIO("\n".join(l for l in p.stdout.splitlines() if not l.lstrip().startswith("Warning")))) if r and r[0]=="submission.csv"]
    top=rows[:2]
    info=[(r[3].split('.')[-1], r[4]) for r in top]
    print(time.strftime("%H:%M:%S"), info, flush=True)
    if len(top)==2 and all(r[4].strip() for r in top):
        print("BOTH_SCORED", [r[4] for r in top]); break
    if any("ERROR" in r[3] for r in top):
        print("AN_ERROR", info); break
    time.sleep(120)
else:
    print("TIMEOUT_100MIN")
