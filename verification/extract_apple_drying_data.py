"""Digitise the apple-slice drying curves of an open-access paper directly from the vector graphics of its PDF.

Source: O. Dajbych, A. Kabutey, Č. Mizera, A. Sedláček, D. Herak, "Effect of Diffusion Path Lengths on Effective
Moisture Diffusivity and Activation Energy of Red Delicious Apple Slices Under Convective Drying", Processes 14(12)
(2026) 2015, https://doi.org/10.3390/pr14122015 (Red Delicious apple slices, convective hot-air oven, 40-80 degC,
duplicate tests I and II, 10 h).
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
The licence of this repository (MIT) does not apply to the data file.
Changes: the moisture-ratio curves of Fig. 2e,f were digitised from the vector graphics of the published PDF
(this script) and resampled at 10-min intervals by linear interpolation; values are not the authors' original data.
Half-thicknesses (Table 1) are transcribed in the scripts. No figure or table of the paper is reproduced.

Figure 2e,f show the moisture ratio
MR = (M_t - M_e)/(M_0 - M_e) against drying time for the five temperatures, with M defined relative to the final
mass (M_e taken as 0). The curves are polylines in the PDF; their coordinates are mapped to data units with the
axis anchors (MR = 1 at t = 0, MR = 0 at the end of the run, 0 and 600 min on the time axis).

Usage: python extract_apple_drying_data.py path/to/processes-14-02015.pdf
Output: ../data/external/apple_drying_curves_Processes2026.csv  (test, temperature_C, time_min, moisture_ratio)
Half-thickness values and the initial dry-basis moisture are transcribed from Table 1 and Figure 2c,d of the paper.
Scope: digitised published data; they carry the paper's own definitions and measurement error.
"""
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "external" / "apple_drying_curves_Processes2026.csv"
COLOUR_TO_T = {(0.0, 0.0, 1.0): 40, (1.0, 0.0, 0.0): 50, (0.0, 0.5, 0.0): 60, (1.0, 0.0, 1.0): 70, (0.0, 0.0, 0.0): 80}
# Table 1: initial (L_i), final (L_f) and average (L_avg) half-thickness in m, per test and temperature
THICKNESS = {("I", 40): (0.00375, 0.00273, 0.00324), ("I", 50): (0.00375, 0.00220, 0.00297), ("I", 60): (0.00415, 0.00228, 0.00322),
             ("I", 70): (0.00407, 0.00242, 0.00324), ("I", 80): (0.00410, 0.00276, 0.00343),
             ("II", 40): (0.00385, 0.00250, 0.00317), ("II", 50): (0.00393, 0.00247, 0.00320), ("II", 60): (0.00402, 0.00247, 0.00325),
             ("II", 70): (0.00412, 0.00228, 0.00320), ("II", 80): (0.00417, 0.00247, 0.00332)}


def polyline(drawing):
    pts = []
    for it in drawing["items"]:
        if it[0] == "l":
            if not pts:
                pts.append((it[1].x, it[1].y))
            pts.append((it[2].x, it[2].y))
    return np.array(pts)


def main(pdf):
    page = fitz.open(pdf)[6]                                   # page 7, Figure 2
    groups = {"I": [], "II": []}
    for d in page.get_drawings():
        if len(d["items"]) > 300 and d["rect"].y0 > 460 and d["rect"].y1 < 612:       # panels e and f (MR, axis 0-1)
            test = "I" if d["rect"].x0 < 300 else "II"
            groups[test].append(d)
    rows = []
    for test, ds in groups.items():
        assert len(ds) == 5, (test, len(ds))
        x0 = min(d["rect"].x0 for d in ds); x600 = max(d["rect"].x1 for d in ds)
        y_top = min(d["rect"].y0 for d in ds); y_bot = max(d["rect"].y1 for d in ds)
        for d in ds:
            T = COLOUR_TO_T[tuple(round(c, 1) for c in d["color"])]
            p = polyline(d)
            t = (p[:, 0] - x0) / (x600 - x0) * 600.0
            mr = (y_bot - p[:, 1]) / (y_bot - y_top)
            order = np.argsort(t)
            t, mr = t[order], mr[order]
            for tt in np.arange(0.0, t.max() + 1e-9, 10.0):
                rows.append((test, T, round(float(tt), 1), round(float(np.interp(tt, t, mr)), 4)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "temperature_C", "time_min", "moisture_ratio"])
        w.writerows(rows)
    print(len(rows), "points written to", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
