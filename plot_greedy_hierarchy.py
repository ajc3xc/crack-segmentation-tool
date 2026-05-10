"""
Greedy Branch Construction Algorithm — Priority Hierarchy Diagram
Thesis figure: shows the 4-level candidate selection hierarchy.
Output: CURRENT_PROJECT_CODE/greedy_hierarchy.png
"""
import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), 'greedy_hierarchy.png')

# ── Layout constants ─────────────────────────────────────────────────────────
FIG_W, FIG_H = 9.5, 7.2
BOX_W, BOX_H = 7.8, 0.88
LEFT        = (FIG_W - BOX_W) / 2
GAP         = 0.28        # vertical gap between boxes
ARROW_H     = GAP * 0.72  # arrow fills most of gap
TOP_Y       = FIG_H - 0.55

DPI = 200

# ── Priority levels ───────────────────────────────────────────────────────────
levels = [
    {
        "rank":    "Priority 1",
        "title":   "Maximum Length (Seed & Tiebreak)",
        "body":    (
            "Seed: select S* = argmax L(Sᵢ) from unused pool.  "
            "Tiebreak: among equally-valid candidates, prefer the longer segment."
        ),
        "color":   "#1a6fad",   # deep blue
        "icon":    "①",
    },
    {
        "rank":    "Priority 2",
        "title":   "Prospective Loop Protection",
        "body":    (
            "Before appending a candidate, verify that its far endpoint is not already "
            "an interior key of the current branch.  Candidates that would close a cycle "
            "are excluded; the segment is isolated as a solo branch."
        ),
        "color":   "#2e8b57",   # sea green
        "icon":    "②",
    },
    {
        "rank":    "Priority 3",
        "title":   "Anti-Doubling-Back Filter  (θ > 150°)",
        "body":    (
            "At each junction, compute the turn angle θ between the incoming branch "
            "direction and each candidate's outgoing direction.  If any candidate at "
            "that junction satisfies θ ≤ 150°, all candidates with θ > 150° are "
            "excluded from that selection pass (near-U-turn suppression)."
        ),
        "color":   "#c17d11",   # amber
        "icon":    "③",
    },
    {
        "rank":    "Priority 4  ·  Full Override",
        "title":   "Image-Edge Proximity Preference  (d_edge < 5 px)",
        "body":    (
            "After angle filtering, if any surviving candidate's far endpoint lies "
            "within 5 px of the image boundary, that candidate is selected unconditionally "
            "— length and turn angle are both overridden.  This anchors crack branches "
            "that reach the frame edge, preventing interior stubs from stealing priority."
        ),
        "color":   "#8b1a1a",   # deep red
        "icon":    "④",
    },
]

# ── Draw ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor='white')
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')

# Title
ax.text(FIG_W / 2, TOP_Y + 0.22,
        "Greedy Branch Construction — Candidate Selection Hierarchy",
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='#111111')

y = TOP_Y
for i, lv in enumerate(levels):
    col = lv['color']

    # ── Box ──────────────────────────────────────────────────────────────────
    box = mpatches.FancyBboxPatch(
        (LEFT, y - BOX_H), BOX_W, BOX_H,
        boxstyle="round,pad=0.04",
        linewidth=2.2, edgecolor=col,
        facecolor=col + '18',   # very light tint
        zorder=2,
    )
    ax.add_patch(box)

    # Left accent bar
    bar = mpatches.FancyBboxPatch(
        (LEFT, y - BOX_H), 0.22, BOX_H,
        boxstyle="round,pad=0.0",
        linewidth=0, facecolor=col,
        zorder=3,
    )
    ax.add_patch(bar)

    # Icon
    ax.text(LEFT + 0.11, y - BOX_H / 2, lv['icon'],
            ha='center', va='center',
            fontsize=16, fontweight='bold', color='white', zorder=4)

    # Rank tag
    ax.text(LEFT + 0.38, y - 0.14,
            lv['rank'],
            ha='left', va='top',
            fontsize=8.5, color=col, fontweight='bold',
            fontstyle='italic', zorder=4)

    # Title
    ax.text(LEFT + 0.38, y - 0.30,
            lv['title'],
            ha='left', va='top',
            fontsize=10.5, fontweight='bold', color='#111111', zorder=4)

    # Body
    ax.text(LEFT + 0.38, y - 0.50,
            lv['body'],
            ha='left', va='top',
            fontsize=8.2, color='#333333',
            wrap=True, zorder=4,
            linespacing=1.35,
            )

    # ── Arrow to next box ────────────────────────────────────────────────────
    if i < len(levels) - 1:
        ax_bot  = y - BOX_H
        ax_next = ax_bot - GAP
        arr = FancyArrowPatch(
            (FIG_W / 2, ax_bot - 0.03),
            (FIG_W / 2, ax_bot - ARROW_H),
            arrowstyle='-|>',
            mutation_scale=14,
            linewidth=2.0,
            color='#555555',
            zorder=5,
        )
        ax.add_patch(arr)

        ax.text(FIG_W / 2 + 0.22, ax_bot - ARROW_H / 2,
                "else ↓",
                ha='left', va='center',
                fontsize=7.5, color='#666666', style='italic')

    y -= BOX_H + GAP

# Footer note
ax.text(FIG_W / 2, 0.12,
        "Priorities evaluated in order per junction per iteration.  "
        "Priority 4 (edge proximity) fully overrides Priority 1–3 when triggered.",
        ha='center', va='bottom', fontsize=7.5, color='#555555', style='italic')

plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()
print('saved', OUT)
