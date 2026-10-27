"""Grayscale schematic figures: overall architecture, twin/conformal mechanism, security round protocol."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"})
from common import FIGS as OUT


class Canvas:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.boxes = {}

    def box(self, name, x, y, w, h, text, fill="white", lw=0.9, fs=7.2, ls="-", bold=False, round_=True):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06" if round_ else "square,pad=0",
                           fc=fill, ec="black", lw=lw, ls=ls, zorder=2)
        self.ax.add_patch(p)
        if text:
            self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
                         fontweight="bold" if bold else "normal", linespacing=1.25)
        self.boxes[name] = (x, y, w, h)

    def frame(self, x, y, w, h, label, fs=7.5):
        self.ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="black", lw=0.8, ls=(0, (4, 2)), zorder=1))
        self.ax.text(x + 0.08, y + h - 0.08, label, ha="left", va="top", fontsize=fs, fontweight="bold", zorder=3)

    def anchor(self, name, side, frac=0.5):
        x, y, w, h = self.boxes[name]
        return {"l": (x, y + h * frac), "r": (x + w, y + h * frac), "t": (x + w * frac, y + h),
                "b": (x + w * frac, y)}[side]

    def arrow(self, p1, p2, dashed=False, text=None, toff=(0, 0.07), fs=6.5, lw=0.9, conn="arc3", ha="center"):
        a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8, lw=lw, color="black",
                            ls=(0, (3, 2)) if dashed else "-", connectionstyle=conn, zorder=4, shrinkA=0, shrinkB=0)
        self.ax.add_patch(a)
        if text:
            mx, my = (p1[0] + p2[0]) / 2 + toff[0], (p1[1] + p2[1]) / 2 + toff[1]
            self.ax.text(mx, my, text, ha=ha, va="center", fontsize=fs, zorder=5,
                         bbox=dict(fc="white", ec="none", pad=0.6))

    def poly(self, pts, dashed=False, text=None, tpos=None, fs=6.5):
        """orthogonal polyline with arrow head on last segment"""
        for a, b in zip(pts[:-2], pts[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=0.9, ls=(0, (3, 2)) if dashed else "-", zorder=4)
        self.arrow(pts[-2], pts[-1], dashed=dashed)
        if text:
            self.ax.text(tpos[0], tpos[1], text, ha="center", va="center", fontsize=fs, zorder=5,
                         bbox=dict(fc="white", ec="none", pad=0.6))

    def save(self, name):
        self.fig.savefig(f"{OUT}/{name}.png", dpi=600)
        plt.close(self.fig)


def fig_architecture():
    C = Canvas(7.16, 4.62)
    G1, G2 = "0.93", "0.82"
    C.frame(0.08, 0.08, 4.90, 4.46, r"Institution $s$ (raw records stay on site)")
    xA, wA, xB, wB, xC, wC, xD, wD = 0.20, 0.92, 1.40, 0.98, 2.68, 0.98, 3.95, 0.90
    ys = [3.62, 3.14, 2.66, 2.18]
    names = ["ECG", "PPG", "Vital-sign series", "Laboratory panel"]
    for k, (y, n) in enumerate(zip(ys, names)):
        C.box(f"in{k}", xA, y, wA, 0.36, n, fs=6.6)
    C.box("ehr", xA, 1.40, wA, 0.36, "EHR covariates", fs=6.6)
    C.box("vault", xA, 0.30, wA, 0.70, "Record vault\nXChaCha20-\nPoly1305", fill=G2, fs=6.3)
    C.arrow(C.anchor("vault", "t"), C.anchor("ehr", "b"))
    C.box("enc", xB, 2.10, wB, 1.88, "Self-supervised\nmultimodal\nencoder $f_{\\phi}$\n\nmasked temporal\nreconstruction\n+\ncross-modal\nInfoNCE", fill=G1, fs=6.4)
    for k in range(4):
        C.arrow(C.anchor(f"in{k}", "r"), (xB, ys[k] + 0.18))
    C.box("adp", xC, 3.20, wC, 0.78, "Acquisition\nadapter $\\psi_s$\n$(\\mathbf{D}_s,\\boldsymbol{\\alpha}_s)$", fs=6.4)
    C.arrow((xB + wB, 3.59), (xC, 3.59), text="$\\mathbf{e}^{m}_{i,t}$", toff=(0, 0.11))
    C.box("twin", xC, 1.86, wC, 1.04, "State-space twin\nVan Loan predict\n+ information\nupdate", fill=G1, fs=6.4)
    C.arrow(C.anchor("adp", "b"), C.anchor("twin", "t"))
    C.poly([C.anchor("ehr", "r"), (3.02, 1.58), (3.02, 1.86)])
    C.box("haz", xD, 2.45, wD, 0.70, "Moment-matched\nhazard\n$\\hat{R}_{i,t}(h)$", fs=6.3)
    C.box("fc", xD, 1.62, 0.60, 0.58, "Next-visit\nforecast", fs=6.1)
    C.arrow((xC + wC, 2.70), (xD, 2.70))
    C.arrow((xC + wC, 1.98), (xD, 1.98))
    C.box("drift", xC, 0.30, wC, 0.80, "Drift monitor\nNIS $\\chi_{i,t}$\nCUSUM alarm", fs=6.4)
    C.arrow((3.40, 1.86), (3.40, 1.10), text="innovations", toff=(0.0, 0.0), fs=6.0)
    C.poly([C.anchor("drift", "l", 0.5), (2.52, 0.70), (2.52, 3.36), (xC, 3.36)], dashed=True,
           text="label-free\nrefit", tpos=(2.24, 1.02))
    C.box("conf", xD, 0.30, wD, 0.92, "Site $\\times$ label\nconformal sets,\ndelayed ACI,\ngated deferral", fill=G1, fs=6.1)
    C.poly([C.anchor("haz", "b", 0.86), (xD + 0.86 * wD, 1.22)])
    C.arrow((xC + wC, 0.58), (xD, 0.58), text="$\\zeta_{i,t}$", toff=(0, 0.10), fs=6.2)
    C.box("loss", xD, 3.56, wD, 0.66, "$\\mathcal{J}_s=\\mathcal{L}_{\\rm surv}$\n$+\\lambda_{\\rm inn}\\mathcal{L}_{\\rm inn}$\n$+\\lambda_{y}\\mathcal{L}_{\\rm traj}$", fs=6.3)
    C.arrow(C.anchor("haz", "t", 0.5), C.anchor("loss", "b", 0.5))
    # reverse-mode gradient (dashed): loss -> adapter and loss -> twin
    C.arrow((xD, 3.86), (xC + wC, 3.86), dashed=True)
    C.poly([(xD, 3.66), (3.82, 3.66), (3.82, 2.82), (xC + wC, 2.82)], dashed=True)
    C.ax.text(3.80, 4.08, "$\\nabla\\mathcal{J}_s$", ha="center", va="center", fontsize=6.2)
    C.frame(5.48, 0.08, 1.60, 4.46, "Aggregation server")
    C.box("sagg", 5.60, 2.80, 1.36, 1.32, "", fill=G1)
    C.ax.text(6.28, 3.74, "Secure aggregation\nML-KEM pairwise masks\nML-DSA verification", ha="center", va="center", fontsize=5.9, zorder=3, linespacing=1.25)
    C.ax.text(6.28, 3.08, "$\\boldsymbol{\\Theta}^{(r+1)}=\\sum_s\\omega_s\\boldsymbol{\\Theta}_s$", ha="center", va="center", fontsize=6.2, zorder=3)
    C.box("shamir", 5.92, 1.66, 1.04, 0.74, "Threshold\nrecovery\nShamir $t$-of-$n$", fs=6.1)
    C.box("ledger", 5.60, 0.30, 1.36, 0.96, "Append-only\naudit ledger\nSHA3-256 chain\n+ ML-DSA-65", fill=G2, fs=6.1)
    C.arrow(C.anchor("shamir", "t"), (6.44, 2.80))
    C.arrow((5.74, 2.80), (5.74, 1.26), text="log", toff=(0.0, 0.0), fs=5.8)
    C.arrow((xD + wD, 3.96), (5.60, 3.96), text="$\\tilde{\\mathbf{y}}_s,\\sigma_s$", toff=(-0.01, 0.11), fs=6.0)
    C.arrow((5.60, 3.24), (4.98, 3.24), text="$\\boldsymbol{\\Theta}^{(r+1)},\\sigma$", toff=(-0.06, 0.12), fs=5.8)
    C.save("fig1_architecture")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_architecture()


def fig_protocol():
    W, H = 3.5, 5.28
    C = Canvas(W, H)
    xs = {"s": 0.52, "sp": 1.38, "srv": 2.30, "led": 3.08}
    heads = {"s": "Institution $s$", "sp": "Institution $s'$", "srv": "Server", "led": "Ledger"}
    top = 4.94
    for k, x in xs.items():
        C.box("h" + k, x - 0.34 if k in ("s", "sp") else x - 0.30, top, 0.68 if k in ("s", "sp") else 0.60, 0.26, heads[k], fill="0.88", fs=5.8)
        C.ax.plot([x, x], [0.08, top], color="black", lw=0.6, ls=(0, (2, 2)), zorder=1)
    y = [4.72]
    bb = dict(fc="white", ec="none", pad=0.5)

    def msg(a, b, text, dy=0.30, dashed=False, fs=5.6):
        yy = y[0]
        C.arrow((xs[a], yy), (xs[b], yy), dashed=dashed)
        xm = (xs[a] + xs[b]) / 2
        C.ax.text(xm, yy + 0.085, text, ha="center", va="bottom", fontsize=fs, zorder=6, bbox=bb)
        y[0] -= dy

    def note(x0, x1, text, h=0.20, dy=0.30, fs=5.5):
        yy = y[0] - 0.02
        C.box("n%.2f" % yy, x0, yy - h / 2, x1 - x0, h, text, fill="white", fs=fs)
        y[0] -= dy

    def phase(text):
        yy = y[0]
        C.ax.add_patch(Rectangle((0.04, yy - 0.08), W - 0.08, 0.16, fc="0.90", ec="none", zorder=3))
        C.ax.text(0.10, yy, text, ha="left", va="center", fontsize=5.8, fontweight="bold", zorder=4)
        y[0] -= 0.27

    phase("1  Pairwise key establishment (every round)")
    msg("sp", "s", "$\\mathrm{ek}_{s'}$ (1184 B)", dy=0.29)
    msg("s", "sp", "$\\mathbf{ct}_{s,s'}$ (1088 B)", dy=0.24)
    note(0.18, 1.79, "both derive $k_{s,s'}=\\mathrm{HKDF}(K_{s,s'},\\,r\\|s\\|s')$", dy=0.29)
    phase("2  Share distribution and local training")
    msg("s", "sp", "Shamir shares (AEAD)", dy=0.24)
    note(0.18, 0.92, "minimise $\\mathcal{J}_s$", dy=0.29)
    phase("3  Masked, signed upload")
    msg("s", "srv", "$\\tilde{\\mathbf{y}}_s$, ML-DSA-65 $\\sigma_s$", dy=0.30)
    msg("srv", "led", "$(\\eta_k,\\sigma_k)$", dy=0.26, fs=5.4)
    phase("4  Dropout recovery ($s'$ silent)")
    msg("srv", "s", "request shares of $k_{s',\\cdot}$", dy=0.30, dashed=True)
    msg("s", "srv", "$t$ shares", dy=0.26, dashed=True)
    phase("5  Aggregation and broadcast")
    note(1.88, 2.72, "unmask, average", dy=0.32)
    msg("srv", "s", "signed $\\boldsymbol{\\Theta}^{(r+1)}$", dy=0.30)
    msg("srv", "led", "$(\\eta_{k+1},\\sigma_{k+1})$", dy=0.22, fs=5.4)
    C.save("fig3_protocol")


if __name__ == "__main__":
    fig_protocol()
