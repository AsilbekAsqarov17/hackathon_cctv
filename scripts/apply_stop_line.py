"""One-off: apply the measured stop line and signal directions to the scene file.

Kept as a script rather than a hand edit so the provenance of every coordinate
in the scene file is reproducible: the segment printed by
``scripts/fit_stop_line_sweep.py`` is pasted in verbatim, together with the note
explaining how it was derived.
"""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path("configs/scenes/data_video1.json")

SEGMENT = [[1824, 804], [1574, 1334]]

NOTE = [
    "b_eastbound_stop is the ONLY signal-controlled approach at this camera.",
    "",
    "DERIVED, NOT AUTHORED BY EYE. scripts/fit_stop_line_sweep.py places it:",
    "",
    "1. The head that governs this line is found by declared approach direction",
    "   (SceneContext.signal_for_line), not by image position, because a signal",
    "   head is mounted high and projects above the road point it controls.",
    "2. The crossing that head faces is the one traffic queues for. On this",
    "   camera that is crossing[1] (centre 1040,276 at 1080p, 161px from the",
    "   head) and NOT crossing[0] (207px away). An earlier calibration anchored",
    "   to crossing[0] and therefore placed the line inside the junction, where",
    "   it was crossed by ordinary through traffic.",
    "3. The line is then swept upstream from that crossing and scored against",
    "   the queue itself: the vehicles that HELD the queue (those that stopped",
    "   upstream of the crossing, not the ones blocked inside it) are the",
    "   calibration target. Setback 80px puts them at -19px and -17px, i.e.",
    "   their noses park right at the line; every other setback is worse.",
    "4. The segment is extended perpendicular to travel until it leaves the",
    "   governed carriageway on both sides, so it spans the whole approach.",
    "   scripts/check_line_span.py asserts this and runs in the test suite.",
    "",
    "The old segment covered only the top ~20% of the carriageway, so crossings",
    "on the rest of it were invisible and red_light could not fire at all.",
    "",
    "Crossing detection uses the vehicle FRONT bumper (the bbox edge furthest",
    "along `direction`), matching the official definition that the event starts",
    "when the front of the vehicle crosses the line. `direction` is the",
    "measured eastbound travel direction, which is also what queue suppression",
    "needs to decide which side is the approach."
]


def main() -> int:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    data["stop_lines"][0]["segment"] = SEGMENT
    data["stop_lines"][0]["direction"] = [1.0, 0.47]
    for light in data["traffic_lights"]:
        if light["id"] == "signal_eastbound":
            light["direction"] = [1.0, 0.47]
        elif light["id"] == "signal_left":
            light["direction"] = [-1.0, -0.30]
    data["_stop_line_note"] = NOTE
    data["_signal_note"] = [
        "Each signal ROI declares the approach direction it faces. A head is",
        "matched to a stop line by comparing declared directions, NOT by image",
        "position: a head sits several metres up, so its ROI projects well above",
        "the road point it governs and can look 'upstream' in the image. See",
        "SceneContext.signal_for_line.",
        "",
        "signal_eastbound governs the only signal-controlled approach here.",
        "signal_left faces the opposite way and is never used to judge that",
        "line; it would be used only if a westbound stop line were calibrated.",
        "",
        "There is no readable signal on the east leg: those heads face away from",
        "this camera, so that region deliberately has no stop line."
    ]
    PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("stop line:", SEGMENT, data["stop_lines"][0]["direction"])
    for light in data["traffic_lights"]:
        print(" ", light["id"], light["roi"], light["direction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
