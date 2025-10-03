import json, sys
from pathlib import Path
import pandas as pd
import typer
from rich import print
from .bundle import save_bundle, load_bundle

app = typer.Typer(add_completion=False)

@app.command("train")
def cli_train():
    import meddx.train as tr
    tr.main()
    from joblib import load
    model = load("artifacts/model.joblib")
    save_bundle(model, threshold=0.5, extra={"stage":"baseline"})
    print("[green]train complete; bundle saved[/green]")

@app.command("eval")
def cli_eval():
    import meddx.evaluate as ev
    ev.main()
    print("[green]eval complete[/green]")

@app.command("predict")
def cli_predict(json_path: str = typer.Option(None, "--json"), stdin: bool = typer.Option(False, "--stdin")):
    if json_path:
        payload = json.loads(json_path)
    elif stdin:
        payload = json.loads(sys.stdin.read())
    else:
        raise typer.BadParameter("provide --json or --stdin")
    model, thr, meta = load_bundle()
    df = pd.DataFrame([payload])
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(df)[:,1][0])
    else:
        p = float(model.predict(df)[0])
    y = int(p >= thr)
    out = {"prob": p, "pred": y, "threshold": thr, "meta": meta}
    print(json.dumps(out))
