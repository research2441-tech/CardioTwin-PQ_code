"""Framingham Heart Study teaching cohort (anonymised release) arranged as a longitudinal, multi-institution
problem: education strata act as institutions, exams 1-3 as visits, incident CVD as the outcome.
Time is expressed in 4-year units so the twin's 12 x 0.25 intervals span 12 years."""
import numpy as np
import pyreadr
from common import DATA, RUNS

UNIT = 4.0 * 365.25   # days per time unit
GROUPS = dict(vit=["SYSBP", "DIABP", "HEARTRTE", "BMI"], metab=["TOTCHOL", "GLUCOSE"], lipid=["HDLC", "LDLC"],
              hist=["PREVHYP", "PREVAP", "CIGPDAY", "BPMEDS"])


def load():
    df = list(pyreadr.read_r(f"{DATA}/framingham.rda").values())[0]
    df = df.sort_values(["RANDID", "PERIOD"]).reset_index(drop=True)
    base = df[df.PERIOD == 1].set_index("RANDID")
    # exclude prevalent CVD at baseline (CHD or stroke), as in standard incident-CVD analyses
    keep = base.index[(base.PREVCHD == 0) & (base.PREVSTRK == 0)]
    df = df[df.RANDID.isin(keep)].copy()
    df["educ"] = df["educ"].fillna(1)
    # visits before the CVD event only (landmark predictions for event-free participants)
    df = df[df.TIME < df.TIMECVD].copy()
    df["T"] = (df.TIMECVD - df.TIME) / UNIT
    df["E"] = df.CVD.astype(int)
    df["t"] = df.TIME / UNIT
    return df


def arrays(df):
    n = len(df)
    V = dict(pid=df.RANDID.values.astype(int), site=(df.educ.values.astype(int) - 1), vidx=(df.PERIOD.values - 1).astype(int),
             t=df.t.values, cal=df.t.values, T=df["T"].values, E=df.E.values, Ttr=df["T"].values, Etr=df.E.values,
             age=df.AGE.values.astype(float), male=(df.SEX.values == 1).astype(int), smoke=df.CURSMOKE.values.astype(int),
             diab=df.DIABETES.values.astype(int), pmi=df.PREVMI.values.astype(int),
             htn=np.nan_to_num(df.BPMEDS.values).astype(int), statin=np.zeros(n, int))
    feats = {g: df[c].values.astype(float) for g, c in GROUPS.items()}
    return V, feats


if __name__ == "__main__":
    df = load()
    V, F = arrays(df)
    print(len(df), len(np.unique(V["pid"])), np.bincount(V["site"]), V["E"].mean())
    for g, x in F.items():
        print(g, np.isnan(x).mean(0).round(2))
