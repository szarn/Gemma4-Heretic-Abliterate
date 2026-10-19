# SPDX-License-Identifier: AGPL-3.0-or-later

"""Draw the README's animated figures from the study journal and the residual-geometry summary.

    python scripts/make_figures.py

Standard library only. Writes a light and a dark SVG of each figure to docs/figures/; the README
picks one per GitHub theme with <picture>. The animations are CSS keyframes (plus SMIL for the
curve morph), so they play inside GitHub's <img> sandbox. Each SVG's un-animated state is the
final frame, which is what viewers with reduced motion see.
"""

import json
import math
import random
from pathlib import Path

CHECKPOINT = Path("checkpoints/google--gemma-4-12B-it.jsonl")
GEOMETRY = Path("outputs/analysis/residual_geometry.json")  # scripts/residual_geometry.py
OUT_DIR = Path("docs/figures")
SELECTED_TRIAL = 143
N_STARTUP_TRIALS = 60  # Heretic's default: random sampling before TPE takes over.

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"

# GitHub's page surfaces and ink; blue/orange validated for CVD separation on both surfaces.
THEMES = {
    "light": {
        "surface": "#ffffff",
        "border": "#d1d9e0",
        "ink": "#1f2328",
        "ink2": "#59636e",
        "muted": "#818b98",
        "grid": "#eef1f4",
        "axis": "#c8d1da",
        "harmless": "#2a78d6",
        "harmful": "#eb6834",
        "random": "#a0a9b3",
        "o_proj": "#59636e",
        "down_proj": "#b3bcc6",
    },
    "dark": {
        "surface": "#0d1117",
        "border": "#3d444d",
        "ink": "#f0f6fc",
        "ink2": "#9198a1",
        "muted": "#7d8590",
        "grid": "#1b2129",
        "axis": "#3d444d",
        "harmless": "#3987e5",
        "harmful": "#d95926",
        "random": "#6e7781",
        "o_proj": "#9198a1",
        "down_proj": "#4f5862",
    },
}


def f(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def keyframes(name: str, period: float, frames: list[tuple[float, str]]) -> str:
    """@keyframes with stops given in seconds of a `period`-second loop."""
    stops = "".join(f"{f(100 * t / period)}%{{{css}}}" for t, css in frames)
    return f"@keyframes {name}{{{stops}}}"


def svg(theme: dict, width: int, height: int, title: str, desc: str, css: str, body: str) -> str:
    t = theme
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="title desc">
<title id="title">{title}</title>
<desc id="desc">{desc}</desc>
<style>
text{{font-family:{FONT};font-size:14px;fill:{t["ink"]}}}
.ink2{{fill:{t["ink2"]}}}.muted{{fill:{t["muted"]}}}.small{{font-size:12px}}.b{{font-weight:600}}
.end{{text-anchor:end}}.mid{{text-anchor:middle}}
.halo{{paint-order:stroke;stroke:{t["surface"]};stroke-width:4px;stroke-linejoin:round}}
{css}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important}}}}
</style>
<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="8" fill="{t["surface"]}" stroke="{t["border"]}"/>
{body}
</svg>
"""


def arrowhead(id_: str, color: str) -> str:
    return (
        f'<marker id="{id_}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,1L9,5L0,9z" fill="{color}"/></marker>'
    )


def fig_direction(t: dict) -> str:
    """Schematic: difference-of-means direction, orthogonalized, then projected out."""
    period = 14.0
    scale, ox, oy = 66, 46, 368  # px per unit; the origin of the 2-D slice

    def px(p):
        return ox + p[0] * scale, oy - p[1] * scale

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    rng = random.Random(7)
    harmless = [(rng.gauss(3.0, 0.38), rng.gauss(1.15, 0.30)) for _ in range(36)]
    harmful = [(rng.gauss(3.7, 0.40), rng.gauss(3.25, 0.32)) for _ in range(36)]
    mu_good = (sum(p[0] for p in harmless) / 36, sum(p[1] for p in harmless) / 36)
    mu_bad = (sum(p[0] for p in harmful) / 36, sum(p[1] for p in harmful) / 36)

    # Heretic: r = normalize(mu_bad - mu_good), minus its component along mu_good.
    g_len = math.hypot(*mu_good)
    g_hat = (mu_good[0] / g_len, mu_good[1] / g_len)
    diff = (mu_bad[0] - mu_good[0], mu_bad[1] - mu_good[1])
    along = dot(diff, g_hat)
    r = (diff[0] - along * g_hat[0], diff[1] - along * g_hat[1])
    r_len = math.hypot(*r)
    r_hat = (r[0] / r_len, r[1] / r_len)

    gx, gy = px(mu_good)
    bx, by = px(mu_bad)
    tip = px((mu_good[0] + r[0], mu_good[1] + r[1]))

    def screen_angle(v):
        return math.degrees(math.atan2(-v[1], v[0]))

    rot = screen_angle(diff) - screen_angle(r)
    stretch = math.hypot(*diff) / r_len

    css = [
        f".dot{{transform:translate(var(--dx),var(--dy));animation:dot {period}s infinite ease-in-out}}",
        keyframes(
            "dot",
            period,
            [
                (0, "opacity:0;transform:translate(0,0)"),
                (1.0, "opacity:1"),
                (7.0, "transform:translate(0,0)"),
                (9.0, "transform:translate(var(--dx),var(--dy))"),
                (13.0, "opacity:1;transform:translate(var(--dx),var(--dy))"),
                (14.0, "opacity:0;transform:translate(var(--dx),var(--dy))"),
            ],
        ),
        f".ghost{{opacity:.22;animation:ghost {period}s infinite}}",
        keyframes("ghost", period, [(0, "opacity:0"), (7.0, "opacity:0"), (7.6, "opacity:.22"), (13.0, "opacity:.22"), (14.0, "opacity:0")]),
        f".means{{animation:means {period}s infinite}}",
        keyframes("means", period, [(0, "opacity:0"), (1.4, "opacity:0"), (1.9, "opacity:1"), (13.0, "opacity:1"), (14.0, "opacity:0")]),
        f".diff{{opacity:.35;transform-origin:{f(gx)}px {f(gy)}px;animation:diff {period}s infinite ease-in-out}}",
        keyframes(
            "diff",
            period,
            [
                (0, "opacity:0;transform:scale(0)"),
                (2.0, "opacity:1;transform:scale(0)"),
                (3.0, "transform:scale(1)"),
                (4.6, "opacity:1"),
                (5.2, "opacity:.35"),
                (13.0, "opacity:.35;transform:scale(1)"),
                (14.0, "opacity:0;transform:scale(1)"),
            ],
        ),
        f".difflabel{{opacity:0;animation:difflabel {period}s infinite}}",
        keyframes("difflabel", period, [(0, "opacity:0"), (2.6, "opacity:0"), (3.0, "opacity:1"), (4.4, "opacity:1"), (4.8, "opacity:0")]),
        f".r{{transform-origin:{f(gx)}px {f(gy)}px;animation:r {period}s infinite ease-in-out}}",
        keyframes(
            "r",
            period,
            [
                (0, f"opacity:0;transform:rotate({f(rot)}deg) scale({stretch:.3f})"),
                (4.2, f"opacity:0;transform:rotate({f(rot)}deg) scale({stretch:.3f})"),
                (4.6, f"opacity:1;transform:rotate({f(rot)}deg) scale({stretch:.3f})"),
                (5.9, "transform:rotate(0deg) scale(1)"),
                (13.0, "opacity:1;transform:rotate(0deg) scale(1)"),
                (14.0, "opacity:0;transform:rotate(0deg) scale(1)"),
            ],
        ),
        f".plane{{animation:plane {period}s infinite}}",
        keyframes("plane", period, [(0, "opacity:0"), (4.2, "opacity:0"), (4.6, "opacity:1"), (13.0, "opacity:1"), (14.0, "opacity:0")]),
        f".perp{{animation:perp {period}s infinite}}",
        keyframes("perp", period, [(0, "opacity:0"), (5.9, "opacity:0"), (6.3, "opacity:1"), (13.0, "opacity:1"), (14.0, "opacity:0")]),
    ]
    step_frames = [
        [(0, "opacity:1"), (4.2, "opacity:1"), (4.5, "opacity:.3"), (13.7, "opacity:.3"), (14.0, "opacity:1")],
        [(0, "opacity:.3"), (3.9, "opacity:.3"), (4.2, "opacity:1"), (6.8, "opacity:1"), (7.1, "opacity:.3"), (14.0, "opacity:.3")],
        [(0, "opacity:.3"), (6.5, "opacity:.3"), (6.8, "opacity:1"), (13.0, "opacity:1"), (13.7, "opacity:.3"), (14.0, "opacity:.3")],
    ]
    for i, frames in enumerate(step_frames, 1):
        css += [f".s{i}{{animation:s{i} {period}s infinite}}", keyframes(f"s{i}", period, frames)]

    body = [
        "<defs>",
        arrowhead("ah-axis", t["axis"]),
        arrowhead("ah-ink2", t["ink2"]),
        arrowhead("ah-ink", t["ink"]),
        "</defs>",
        f'<text x="{ox}" y="34" class="small muted">Schematic: a 2-D slice of the 3,840-dimensional residual stream</text>',
        # Axes of the slice.
        f'<path d="M{ox},{oy}H{f(ox + 5.7 * scale)}" stroke="{t["axis"]}" stroke-width="1.5" fill="none" marker-end="url(#ah-axis)"/>',
        f'<path d="M{ox},{oy}V{f(oy - 4.5 * scale)}" stroke="{t["axis"]}" stroke-width="1.5" fill="none" marker-end="url(#ah-axis)"/>',
        f'<text x="{ox - 8}" y="{oy + 4}" class="small muted end">0</text>',
    ]

    # The plane orthogonal to r_hat passes through the origin and mu_good (r_hat is orthogonal to mu_good).
    reach = min(5.6 / g_hat[0], 4.4 / g_hat[1])
    ex, ey = px((g_hat[0] * reach, g_hat[1] * reach))
    body += [
        '<g class="plane">',
        f'<path d="M{ox},{oy}L{f(ex)},{f(ey)}" stroke="{t["muted"]}" stroke-width="1.5" stroke-dasharray="5 5" fill="none"/>',
        f'<text x="{f(ex)}" y="{f(ey - 10)}" class="small ink2 end">plane ⊥ r̂</text>',
        "</g>",
    ]

    for cls in ("ghost", "dot"):
        body.append(f'<g class="{cls}s">')
        for pts, color in ((harmless, t["harmless"]), (harmful, t["harmful"])):
            for p in pts:
                x, y = px(p)
                k = dot(p, r_hat)
                style = ""
                if cls == "dot":
                    style = f' style="--dx:{-k * r_hat[0] * scale:.1f}px;--dy:{k * r_hat[1] * scale:.1f}px"'
                body.append(
                    f'<circle class="{cls}" cx="{f(x)}" cy="{f(y)}" r="4.5" fill="{color}" '
                    f'stroke="{t["surface"]}" stroke-width="1.5"{style}/>'
                )
        body.append("</g>")

    # Difference of means, then the orthogonalized direction r_hat rotating out of it.
    shorten = 9 / math.hypot(bx - gx, by - gy)
    body += [
        f'<path class="diff" d="M{f(gx)},{f(gy)}L{f(bx - (bx - gx) * shorten)},{f(by - (by - gy) * shorten)}" '
        f'stroke="{t["ink2"]}" stroke-width="1.75" fill="none" marker-end="url(#ah-ink2)"/>',
        f'<text class="difflabel small ink2 end halo" x="{f((gx + bx) / 2 - 12)}" y="{f((gy + by) / 2)}">μ harmful − μ harmless</text>',
        f'<g class="r"><path d="M{f(gx)},{f(gy)}L{f(tip[0])},{f(tip[1])}" stroke="{t["ink"]}" stroke-width="2.5" '
        f'fill="none" marker-end="url(#ah-ink)"/></g>',
        f'<text x="{f(tip[0] - 10)}" y="{f(tip[1] + 4)}" class="perp b end halo" style="font-size:16px">r̂</text>',
    ]
    a = 0.17
    corner = [
        px((mu_good[0] + a * g_hat[0], mu_good[1] + a * g_hat[1])),
        px((mu_good[0] + a * (g_hat[0] + r_hat[0]), mu_good[1] + a * (g_hat[1] + r_hat[1]))),
        px((mu_good[0] + a * r_hat[0], mu_good[1] + a * r_hat[1])),
    ]
    body.append(
        f'<path class="perp" d="M{f(corner[0][0])},{f(corner[0][1])}L{f(corner[1][0])},{f(corner[1][1])}'
        f'L{f(corner[2][0])},{f(corner[2][1])}" stroke="{t["ink"]}" stroke-width="1.25" fill="none"/>'
    )
    body += [
        '<g class="means">',
        f'<circle cx="{f(gx)}" cy="{f(gy)}" r="7" fill="{t["harmless"]}" stroke="{t["ink"]}" stroke-width="2"/>',
        f'<circle cx="{f(bx)}" cy="{f(by)}" r="7" fill="{t["harmful"]}" stroke="{t["ink"]}" stroke-width="2"/>',
        f'<text x="{f(gx + 13)}" y="{f(gy + 18)}" class="small b halo">μ harmless</text>',
        f'<text x="{f(bx + 13)}" y="{f(by - 9)}" class="small b halo">μ harmful</text>',
        "</g>",
    ]

    steps = [
        (
            "Find the refusal direction",
            [
                "Average the residual stream over 400 harmful",
                "and 400 harmless prompts, layer by layer. The",
                "difference of the two means points at refusal.",
            ],
        ),
        (
            "Spare the harmless direction",
            [
                "Drop its component along the harmless mean",
                "(r̂ ⊥ μ harmless), so ordinary activations",
                "are left where they are.",
            ],
        ),
        (
            "Project it out of the weights",
            [
                "W ← W − λ r̂ r̂ᵀ W for each attn.o_proj and",
                "mlp.down_proj. Those layers stop writing along",
                "r̂, so harmful activations lose that component.",
            ],
        ),
    ]
    px0 = 486  # step panel
    for i, (head, lines) in enumerate(steps, 1):
        y = 82 + (i - 1) * 104
        body.append(f'<g class="s{i}">')
        body.append(f'<circle cx="{px0}" cy="{y - 5}" r="11" fill="none" stroke="{t["ink"]}" stroke-width="1.5"/>')
        body.append(f'<text x="{px0}" y="{y}" class="b mid small">{i}</text>')
        body.append(f'<text x="{px0 + 21}" y="{y}" class="b" style="font-size:15px">{head}</text>')
        for j, line in enumerate(lines):
            body.append(f'<text x="{px0 + 21}" y="{y + 24 + j * 20}" class="ink2">{line}</text>')
        body.append("</g>")

    return svg(
        t,
        880,
        400,
        "Abliteration in the residual stream",
        "Animated schematic. Harmless and harmful prompt activations form two clusters. The refusal "
        "direction r-hat is the difference of their means, made orthogonal to the harmless mean. "
        "Projecting r-hat out of the weights moves harmful activations onto the plane orthogonal to "
        "r-hat, while harmless activations barely move.",
        "\n".join(css),
        "\n".join(body),
    )


def load_trials() -> list[dict]:
    """Completed trials' user attributes from Optuna's journal, in trial order."""
    attrs: dict[int, dict] = {}
    for line in CHECKPOINT.read_text().splitlines():
        op = json.loads(line)
        if op["op_code"] == 8:  # SET_TRIAL_USER_ATTR, one attribute per entry
            attrs.setdefault(op["trial_id"], {}).update(op["user_attr"])
    return sorted(attrs.values(), key=lambda a: a["index"])


def pareto_front(trials: list[dict]) -> list[dict]:
    """Non-dominated trials by (refusals, KL), as heretic/main.py lists them."""
    front, best_kl = [], math.inf
    for trial in sorted(trials, key=lambda a: (a["refusals"], a["kl_divergence"])):
        if trial["kl_divergence"] < best_kl:
            best_kl = trial["kl_divergence"]
            front.append(trial)
    return front


def fig_search(t: dict, trials: list[dict]) -> str:
    """The 200 trials appearing in order, then the Pareto front and the chosen trial."""
    period = 16.0
    left, right, top, bottom = 72, 852, 56, 372
    log_min, log_max = -3.25, -0.25
    y_min = 20  # no trial got below 35 refusals

    def px(kl, refusals):
        x = left + (math.log10(kl) - log_min) / (log_max - log_min) * (right - left)
        return x, bottom - (refusals - y_min) / (100 - y_min) * (bottom - top)

    front = pareto_front(trials)
    front_ids = {a["index"] for a in front}
    first_front, front_done = 8.7, 10.3

    css = [
        ".dot{transform-box:fill-box;transform-origin:center}",
        f".front{{stroke-dasharray:1;stroke-dashoffset:0;animation:front {period}s infinite ease-in-out}}",
        keyframes(
            "front",
            period,
            [(0, "stroke-dashoffset:1"), (first_front, "stroke-dashoffset:1"), (front_done, "stroke-dashoffset:0;opacity:1"), (15.0, "opacity:1"), (16.0, "opacity:0;stroke-dashoffset:0")],
        ),
        f".pick{{transform-box:fill-box;transform-origin:center;animation:pick {period}s infinite ease-out}}",
        keyframes(
            "pick",
            period,
            [(0, "opacity:0;transform:scale(2.5)"), (front_done, "opacity:0;transform:scale(2.5)"), (front_done + 0.5, "opacity:1;transform:scale(1)"), (15.0, "opacity:1;transform:scale(1)"), (16.0, "opacity:0;transform:scale(1)")],
        ),
        f".picklabel{{animation:picklabel {period}s infinite}}",
        keyframes("picklabel", period, [(0, "opacity:0"), (front_done + 0.3, "opacity:0"), (front_done + 0.8, "opacity:1"), (15.0, "opacity:1"), (16.0, "opacity:0")]),
    ]
    phases = [
        ("Trials 1–60: random exploration", 0, 0.4 + N_STARTUP_TRIALS * 0.04),
        ("Trials 61–200: TPE-guided search", 0.4 + N_STARTUP_TRIALS * 0.04, first_front),
        (f"{len(front)} trials on the Pareto front", first_front, 15.0),
    ]
    phase_frames = [
        [(0, "opacity:1"), (phases[0][2], "opacity:0"), (period, "opacity:0")],
        [(0, "opacity:0"), (phases[1][1], "opacity:1"), (phases[1][2], "opacity:0"), (period, "opacity:0")],
        [(0, "opacity:0"), (phases[2][1], "opacity:1"), (15.5, "opacity:0"), (period, "opacity:0")],
    ]
    for i, frames in enumerate(phase_frames):
        css += [f".ph{i}{{opacity:{1 if i == 2 else 0};animation:ph{i} {period}s infinite step-end}}", keyframes(f"ph{i}", period, frames)]

    body = [f'<text x="{right}" y="34" class="b end ph{i}">{label}</text>' for i, (label, _, _) in enumerate(phases)]
    # Legend.
    lx = left
    for label, color in (("random (60)", t["random"]), ("TPE-guided (140)", t["harmless"]), ("Pareto front", t["harmful"])):
        body.append(f'<circle cx="{lx + 5}" cy="29" r="5" fill="{color}"/>')
        body.append(f'<text x="{lx + 15}" y="34" class="ink2">{label}</text>')
        lx += 15 + len(label) * 7.4 + 22

    # Grid and axes.
    for refusals in (20, 40, 60, 80, 100):
        _, y = px(1, refusals)
        body.append(f'<path d="M{left},{f(y)}H{right}" stroke="{t["grid"] if refusals > y_min else t["axis"]}" stroke-width="1"/>')
        body.append(f'<text x="{left - 8}" y="{f(y + 4)}" class="small muted end">{refusals}</text>')
    for kl, label in ((0.001, "0.001"), (0.01, "0.01"), (0.1, "0.1")):
        x, _ = px(kl, 0)
        body.append(f'<path d="M{f(x)},{top}V{bottom}" stroke="{t["grid"]}" stroke-width="1"/>')
        body.append(f'<text x="{f(x)}" y="{bottom + 20}" class="small muted mid">{label}</text>')
    _, y99 = px(1, 99)
    body.append(f'<path d="M{left},{f(y99)}H{right}" stroke="{t["muted"]}" stroke-width="1" stroke-dasharray="4 4"/>')
    body.append(f'<text x="{left + 8}" y="{f(y99 + 18)}" class="small muted">original model: 99</text>')
    body.append(f'<text x="{(left + right) / 2}" y="{bottom + 44}" class="small ink2 mid">KL divergence from the original model (first token, harmless prompts; log scale)</text>')
    body.append(f'<text transform="translate(26,{(top + bottom) / 2}) rotate(-90)" class="small ink2 mid">Refusals out of 100 harmful prompts</text>')

    # Trials, each popping in at its turn; Pareto trials turn orange once the front is drawn.
    for trial in trials:
        k = trial["index"]
        start = 0.4 + (k - 1) * 0.04
        base = t["random"] if k <= N_STARTUP_TRIALS else t["harmless"]
        on_front = k in front_ids
        fill = t["harmful"] if on_front else base
        frames = [
            (0, "opacity:0;transform:scale(0)" + (f";fill:{base}" if on_front else "")),
            (start, "opacity:0;transform:scale(0)"),
            (start + 0.15, "opacity:1;transform:scale(1.6)"),
            (start + 0.4, "transform:scale(1)"),
        ]
        if on_front:
            frames += [(first_front, f"fill:{base}"), (first_front + 0.6, f"fill:{t['harmful']}")]
        frames += [(15.0, "opacity:1"), (16.0, "opacity:0;transform:scale(1)" + (f";fill:{t['harmful']}" if on_front else ""))]
        css.append(keyframes(f"t{k}", period, frames))
        x, y = px(trial["kl_divergence"], trial["refusals"])
        body.append(
            f'<circle class="dot" cx="{f(x)}" cy="{f(y)}" r="4.5" fill="{fill}" stroke="{t["surface"]}" '
            f'stroke-width="1.5" style="animation:t{k} {period}s infinite ease-out"/>'
        )

    # Pareto staircase, drawn after the trials, then the chosen trial.
    points = [px(a["kl_divergence"], a["refusals"]) for a in sorted(front, key=lambda a: a["kl_divergence"])]
    d = f"M{f(points[0][0])},{f(points[0][1])}" + "".join(f"H{f(x)}V{f(y)}" for x, y in points[1:])
    body.append(f'<path class="front" d="{d}" pathLength="1" stroke="{t["harmful"]}" stroke-width="2" fill="none"/>')
    pick = next(a for a in trials if a["index"] == SELECTED_TRIAL)
    x, y = px(pick["kl_divergence"], pick["refusals"])
    body.append(f'<circle class="pick" cx="{f(x)}" cy="{f(y)}" r="11" fill="none" stroke="{t["ink"]}" stroke-width="2"/>')
    body.append(
        f'<g class="picklabel"><path d="M{f(x + 8)},{f(y + 8)}L{f(x + 40)},{f(y + 40)}" stroke="{t["ink2"]}" stroke-width="1"/>'
        f'<text x="{f(x + 44)}" y="{f(y + 46)}" class="b">Trial {SELECTED_TRIAL} (saved)</text>'
        f'<text x="{f(x + 44)}" y="{f(y + 64)}" class="ink2">{pick["refusals"]} refusals, KL {pick["kl_divergence"]:.3f}</text></g>'
    )

    return svg(
        t,
        880,
        430,
        "Heretic's parameter search",
        f"Animated scatter of all {len(trials)} trials, refusals against KL divergence on a log scale, "
        f"appearing in order: {N_STARTUP_TRIALS} random trials, then TPE-guided ones. The Pareto front of "
        f"{len(front)} trials is drawn, and trial {SELECTED_TRIAL} ({pick['refusals']} refusals, KL "
        f"{pick['kl_divergence']:.4f}) is highlighted as the saved model.",
        "\n".join(css),
        "\n".join(body),
    )


def ablation_weights(params: dict, n_layers: int) -> list[float]:
    """Per-layer weight lambda_l for one component, as heretic/model.py, abliterate() computes it."""
    weights = []
    for layer in range(n_layers):
        distance = abs(layer - params["max_weight_position"])
        if distance > params["min_weight_distance"]:
            weights.append(0.0)
        else:
            weights.append(
                params["max_weight"]
                + distance / params["min_weight_distance"] * (params["min_weight"] - params["max_weight"])
            )
    return weights


def fig_layers(t: dict, geometry: dict, trial: dict) -> str:
    """Per-layer alignment with r_hat, morphing from the original model to the abliterated one."""
    period = 15.0
    n = geometry["n_layers"]
    left, right, top, bottom = 72, 852, 64, 300
    c_min, c_max = -0.08, 0.52
    strip_base, strip_px = 380, 34  # kernel strip baseline and px per unit of lambda

    def x(i):
        return left + i / n * (right - left)

    def y(c):
        return bottom - (c - c_min) / (c_max - c_min) * (bottom - top)

    def line(values):
        return "M" + "L".join(f"{f(x(i))},{f(y(v))}" for i, v in enumerate(values))

    def band(lo, hi):
        upper = "L".join(f"{f(x(i))},{f(y(v))}" for i, v in enumerate(hi))
        lower = "L".join(f"{f(x(i))},{f(y(v))}" for i, v in reversed(list(enumerate(lo))))
        return f"M{upper}L{lower}Z"

    cos = geometry["cosine_to_r"]
    morph_start, morph_end = 4.2, 6.2

    def morph(before: str, after: str) -> str:
        return (
            f'<animate attributeName="d" dur="{period}s" repeatCount="indefinite" '
            f'values="{before};{before};{after};{after}" '
            f'keyTimes="0;{morph_start / period:.4f};{morph_end / period:.4f};1" calcMode="spline" '
            'keySplines="0 0 1 1;0.45 0 0.2 1;0 0 1 1"/>'
        )

    fade = [(13.8, "opacity:1"), (15.0, "opacity:0")]
    css = [
        # SMIL ignores prefers-reduced-motion, so the morphing paths have static stand-ins.
        ".still{display:none}@media (prefers-reduced-motion:reduce){.still{display:inline}.morph{display:none}}",
        f".curve{{stroke-dasharray:1;stroke-dashoffset:0;animation:curve {period}s infinite ease-in-out}}",
        keyframes("curve", period, [(0, "stroke-dashoffset:1"), (0.3, "stroke-dashoffset:1"), (2.2, "stroke-dashoffset:0;opacity:1")] + fade),
        f".band{{animation:band {period}s infinite}}",
        keyframes("band", period, [(0, "opacity:0"), (1.6, "opacity:0"), (2.4, "opacity:1")] + fade),
        f".rmark{{animation:rmark {period}s infinite}}",
        keyframes("rmark", period, [(0, "opacity:0"), (2.2, "opacity:0"), (2.6, "opacity:1")] + fade),
        f".kernel{{transform-origin:0 {strip_base}px;animation:kernel {period}s infinite ease-in-out}}",
        keyframes("kernel", period, [(0, "transform:scaleY(0)"), (2.8, "transform:scaleY(0)"), (4.0, "transform:scaleY(1)")] + fade),
        f".ghost{{animation:ghost {period}s infinite}}",
        keyframes("ghost", period, [(0, "opacity:0"), (morph_start, "opacity:0"), (morph_start + 0.4, "opacity:1")] + fade),
        f".note{{animation:note {period}s infinite}}",
        keyframes("note", period, [(0, "opacity:0"), (morph_end + 0.3, "opacity:0"), (morph_end + 0.8, "opacity:1")] + fade),
        f".before{{opacity:0;animation:before {period}s infinite}}",
        keyframes("before", period, [(0, "opacity:1"), (morph_start, "opacity:1"), (morph_start + 0.4, "opacity:0"), (14.6, "opacity:0"), (15.0, "opacity:1")]),
        f".after{{animation:after {period}s infinite}}",
        keyframes("after", period, [(0, "opacity:0"), (morph_start, "opacity:0"), (morph_start + 0.4, "opacity:1"), (14.6, "opacity:1"), (15.0, "opacity:0")]),
    ]

    body = ["<defs>", arrowhead("ah-ink2", t["ink2"]), "</defs>"]

    # Legend and the before/after chip.
    lx = left
    for label, color in (("harmful prompts", t["harmful"]), ("harmless prompts", t["harmless"])):
        body.append(f'<path d="M{lx},29H{lx + 18}" stroke="{color}" stroke-width="2.5"/>')
        body.append(f'<text x="{lx + 24}" y="34" class="ink2">{label}</text>')
        lx += 24 + len(label) * 7.4 + 22
    body.append(f'<text x="{left}" y="54" class="small muted">median and interquartile range over 100 held-out prompts each</text>')
    for cls, label in (("before", "Original model"), ("after", f"Abliterated (trial {SELECTED_TRIAL})")):
        w = len(label) * 7.6 + 24
        body.append(
            f'<g class="{cls}"><rect x="{f(right - w)}" y="15" width="{f(w)}" height="26" rx="13" fill="none" '
            f'stroke="{t["ink"]}" stroke-width="1.25"/><text x="{f(right - w / 2)}" y="33" class="b mid">{label}</text></g>'
        )

    # Grid and axes.
    for c in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        body.append(f'<path d="M{left},{f(y(c))}H{right}" stroke="{t["axis"] if c == 0 else t["grid"]}" stroke-width="1"/>')
        body.append(f'<text x="{left - 8}" y="{f(y(c) + 4)}" class="small muted end">{f(c) if c else "0"}</text>')
    for i in range(0, n + 1, 8):
        body.append(f'<text x="{f(x(i))}" y="{strip_base + 20}" class="small muted mid">{i}</text>')
    body.append(f'<text x="{(left + right) / 2}" y="{strip_base + 42}" class="small ink2 mid">Residual stream after layer n (0 = embeddings)</text>')
    body.append(f'<text transform="translate(26,{(top + bottom) / 2}) rotate(-90)" class="small ink2 mid">cos(residual, r̂)</text>')

    # Where r_hat comes from: interpolated between layers at the trial's direction index.
    rx = x(geometry["direction_index"] + 1)
    body.append(
        f'<g class="rmark"><path d="M{f(rx)},{top - 6}V{bottom}" stroke="{t["muted"]}" stroke-width="1.25" stroke-dasharray="2 4"/>'
        f'<text x="{f(rx)}" y="{top - 12}" class="small ink2 mid">r̂ measured here</text></g>'
    )

    # Original (ghost) vs. current curves; the current ones morph from original to abliterated.
    for name in ("harmless", "harmful"):
        before, after = cos["original"][name], cos["heretic"][name]
        b_band, a_band = band(before["q25"], before["q75"]), band(after["q25"], after["q75"])
        attrs = f'd="{a_band}" fill="{t[name]}" fill-opacity="0.16"'
        body.append(f'<path class="band morph" {attrs}>{morph(b_band, a_band)}</path><path class="still" {attrs}/>')
    body.append(
        f'<path class="ghost" d="{line(cos["original"]["harmful"]["median"])}" stroke="{t["harmful"]}" stroke-width="1.5" '
        'stroke-dasharray="4 4" fill="none" opacity="0.8"/>'
    )
    for name in ("harmless", "harmful"):
        b_line, a_line = line(cos["original"][name]["median"]), line(cos["heretic"][name]["median"])
        attrs = f'd="{a_line}" stroke="{t[name]}" stroke-width="2.25" fill="none" stroke-linejoin="round"'
        body.append(f'<path class="curve morph" pathLength="1" {attrs}>{morph(b_line, a_line)}</path><path class="still" {attrs}/>')

    peak = max(range(n + 1), key=lambda i: cos["original"]["harmful"]["median"][i])
    p_before = cos["original"]["harmful"]["median"][peak]
    p_after = cos["heretic"]["harmful"]["median"][peak]
    body.append(
        f'<g class="note"><path d="M{f(x(peak) - 14)},{f(y(p_before) + 4)}V{f(y(p_after) - 10)}" stroke="{t["ink2"]}" '
        f'stroke-width="1.5" marker-end="url(#ah-ink2)"/>'
        f'<text x="{f(x(peak) - 22)}" y="{f(y(0.42))}" class="b end">refusal signal</text>'
        f'<text x="{f(x(peak) - 22)}" y="{f(y(0.42) + 18)}" class="b end">projected out</text></g>'
    )

    # Kernel strip: lambda_l per decoder layer l (0-based in Heretic), drawn at its output, layer l + 1.
    o_proj = ablation_weights(trial["parameters"]["attn.o_proj"], n)
    down_proj = ablation_weights(trial["parameters"]["mlp.down_proj"], n)
    first = min(i for i in range(n) if o_proj[i] or down_proj[i])
    body.append(f'<text x="{left}" y="{strip_base - 34}" class="small ink2">Abliteration weight λ per layer</text>')
    lx = left
    for label, color in (("attn.o_proj", t["o_proj"]), ("mlp.down_proj", t["down_proj"])):
        body.append(f'<rect x="{lx}" y="{strip_base - 20}" width="9" height="9" rx="2" fill="{color}"/>')
        body.append(f'<text x="{lx + 14}" y="{strip_base - 11}" class="small muted">{label}</text>')
        lx += 14 + len(label) * 6.6 + 16
    body.append(f'<text x="{f(x(first + 1) - 12)}" y="{strip_base - 4}" class="small muted end">layers {first + 1}–{n} only</text>')
    body.append('<g class="kernel">')
    for layer in range(n):
        cx = x(layer + 1)
        for dx, w, color in ((-6, o_proj[layer], t["o_proj"]), (1, down_proj[layer], t["down_proj"])):
            if w:
                h = w * strip_px
                body.append(f'<rect x="{f(cx + dx)}" y="{f(strip_base - h)}" width="5" height="{f(h)}" rx="1.5" fill="{color}"/>')
    body.append("</g>")
    body.append(f'<path d="M{left},{strip_base}H{right}" stroke="{t["axis"]}" stroke-width="1"/>')
    body.append(f'<path d="M{f(x(first + 1) - 8)},{f(strip_base - strip_px)}H{right}" stroke="{t["muted"]}" stroke-width="1" stroke-dasharray="2 3"/>')
    body.append(f'<text x="{f(x(first + 1) - 12)}" y="{f(strip_base - strip_px + 4)}" class="small muted end">λ = 1</text>')

    return svg(
        t,
        880,
        432,
        "Refusal signal per layer, before and after abliteration",
        f"Animated line chart over the {n} layers of gemma-4-12B-it: the cosine between each prompt's "
        "residual stream and the refusal direction r-hat, median and interquartile range for 100 harmful "
        "and 100 harmless held-out prompts. In the original model the harmful curve peaks near 0.46 "
        "around layer 30. After abliteration it falls to about 0.05 there and tracks the harmless curve, "
        f"which barely changes. A strip below shows trial {SELECTED_TRIAL}'s ablation weights, applied "
        f"to layers {first + 1} to {n}.",
        "\n".join(css),
        "\n".join(body),
    )


def main():
    trials = load_trials()
    selected = next(a for a in trials if a["index"] == SELECTED_TRIAL)
    geometry = json.loads(GEOMETRY.read_text()) if GEOMETRY.exists() else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme_name, theme in THEMES.items():
        figures = {"direction": fig_direction(theme), "search": fig_search(theme, trials)}
        if geometry is not None:
            figures["layers"] = fig_layers(theme, geometry, selected)
        for name, content in figures.items():
            path = OUT_DIR / f"{name}-{theme_name}.svg"
            path.write_text(content)
            print(f"Wrote {path}")
    if geometry is None:
        print(f"Skipped the layers figure: {GEOMETRY} not found (see scripts/residual_geometry.py)")


if __name__ == "__main__":
    main()
