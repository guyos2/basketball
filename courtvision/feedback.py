"""Turn decisions into coaching feedback (Hebrew)."""
from __future__ import annotations

from .court import ZONE_LABELS_HE
from .decision import Decision, PassOption, ShotOption, contest_label


def _clock(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _shot(s: ShotOption) -> str:
    return (f"{ZONE_LABELS_HE[s.zone]}, {contest_label(s.defender_distance)} "
            f"(מגן במרחק {s.defender_distance:.1f} מ'), {s.expected_points:.2f} נק' צפויות")


def _pass(p: PassOption) -> str:
    return (f"מסירה ל-#{p.receiver} ({p.distance:.1f} מ', {p.success:.0%} שתגיע) ← אחרי שההגנה תסגור: "
            f"{_shot(p.shot)}; שווה {p.expected_points:.2f} נק'")


def decision_text(d: Decision) -> str:
    best_pass = d.passes[0] if d.passes else None
    lines = []
    if d.event_kind == "shot":
        lines.append(f"[{_clock(d.time)}] #{d.player} זרק: {_shot(d.shot)}.")
    else:
        to = f" ל-#{d.target}" if d.target is not None else ""
        lines.append(f"[{_clock(d.time)}] #{d.player} מסר{to}. אפשרות הזריקה שלו: {_shot(d.shot)}.")
        chosen = next((p for p in d.passes if p.receiver == d.target), None)
        if chosen:
            lines.append(f"  המסירה שבוצעה: {_pass(chosen)}.")
    if best_pass and (d.event_kind == "shot" or best_pass.receiver != d.target):
        lines.append(f"  האפשרות הטובה ביותר למסירה: {_pass(best_pass)}.")

    if d.verdict == "good":
        lines.append("  ✔ החלטה טובה.")
    elif d.verdict == "should_pass":
        lines.append(f"  ✘ עדיף היה למסור ל-#{d.recommended_target} "
                     f"(+{d.best_points - d.shot.expected_points:.2f} נק' צפויות).")
    elif d.verdict == "should_shoot":
        lines.append(f"  ✘ עדיף היה לזרוק – הייתה זריקה טובה יותר מכל מסירה "
                     f"(+{d.shot.expected_points - (d.actual_points or best_pass.expected_points):.2f}).")
    elif d.verdict == "better_pass":
        lines.append(f"  ~ מסירה נכונה, אבל ל-#{best_pass.receiver} הייתה עדיפה "
                     f"(+{best_pass.expected_points - d.actual_points:.2f}).")
    return "\n".join(lines)


def report_markdown(decisions: list[Decision], duration: float) -> str:
    good = sum(d.verdict == "good" for d in decisions)
    lost = sum(d.regret or 0.0 for d in decisions if d.verdict != "good")
    out = [
        "# ניתוח החלטות: זריקה מול מסירה",
        "",
        f"אורך הקטע: {duration:.1f} שניות. נותחו {len(decisions)} החלטות, {good} מהן טובות.",
    ]
    if lost:
        out.append(f"נקודות צפויות שהלכו לאיבוד בהחלטות שאפשר לשפר: {lost:.2f}.")
    out += ["", "```"] + ["\n".join([decision_text(d), ""]) for d in decisions] + ["```", ""]
    out += [
        "## איך לקרוא את המספרים",
        "- **נק' צפויות**: כמה נקודות הזריקה שווה בממוצע, לפי האזור במגרש והמרחק מהמגן הקרוב.",
        "- **מסירה**: שווי הזריקה של המקבל אחרי שההגנה מספיקה לסגור עליו בזמן המסירה, "
        "כפול הסיכוי שהמסירה לא תיחטף.",
        "- המספרים מבוססים על ממוצעי ליגה ומיועדים להשוואה בין אפשרויות באותו רגע, לא לחיזוי אחוזים של שחקן מסוים.",
        "",
    ]
    return "\n".join(out)
