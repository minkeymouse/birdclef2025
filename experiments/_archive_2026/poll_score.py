import subprocess, time, csv, io
COMP="birdclef-2026"
for _ in range(50):  # up to ~100 min
    p=subprocess.run(["uv","run","kaggle","competitions","submissions","-c",COMP,"--csv"],capture_output=True,text=True)
    rows=[r for r in csv.reader(io.StringIO("\n".join(l for l in p.stdout.splitlines() if not l.lstrip().startswith("Warning")))) if r and r[0]=="submission.csv"]
    top=rows[0]
    print(time.strftime("%H:%M:%S"),"status=",top[3].split(".")[-1],"public=",repr(top[4]),flush=True)
    if top[4].strip(): print("SCORE_READY",top[4]); break
    if "ERROR" in top[3]: print("ERRORED"); break
    time.sleep(120)
else:
    print("NO_SCORE_AFTER_100MIN")
