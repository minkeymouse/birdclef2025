import subprocess, time, csv, io
COMP="birdclef-2026"
while True:
    p=subprocess.run(["uv","run","kaggle","competitions","submissions","-c",COMP,"--csv"],capture_output=True,text=True)
    lines=[l for l in p.stdout.splitlines() if not l.lstrip().startswith("Warning")]
    rows=[r for r in csv.reader(io.StringIO("\n".join(lines))) if r and r[0]=="submission.csv"]
    top=rows[:2]
    scored=[(r[3],r[4]) for r in top]
    print(time.strftime("%H:%M:%S"), scored, flush=True)
    if len(top)==2 and all(r[4].strip() or "COMPLETE" in r[3] for r in top):
        print("BOTH_SCORED", scored); break
    time.sleep(120)
