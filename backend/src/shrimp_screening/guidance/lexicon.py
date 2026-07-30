"""A scanner for claims this project is not entitled to make.

The naive version of this check -- "reject any text containing the word
`healthy`" -- fails immediately, because the single most important sentence in the
product is *"This does not mean the shrimp is healthy or disease-free."* A rule
that forbids the disclaimer along with the claim would be worse than no rule: it
would push authors toward vaguer language precisely where precision matters.

So the scanner distinguishes two kinds of prohibition.

**Contextual claims** (health assertions, diagnosis, confirmation, cure). Flagged
only when the *sentence containing them* carries no negation or hedge. "The shrimp
is healthy" is a violation; "cannot tell you whether the shrimp is healthy" is not.
Sentence-scoped rather than window-scoped, because a sentence is the unit a reader
actually parses, and it is explainable to a non-programmer.

**Absolute prohibitions** (drug names, dose units and quantities). Never
acceptable in any framing. There is no legitimate reason for this system to emit
"oxytetracycline" or "5 mg/L", including inside a warning -- naming a compound and
a number is how a reader ends up with a dose regardless of the surrounding words.
The correct phrasing is "do not apply medication without professional advice",
which contains no compound and no number.

This module is deliberately importable by tests, the guidance loader and any
future authoring tool, so one definition governs all of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Cues that can negate a claim when they directly govern it. Broad hedges such
#: as "may", "might", "only", "ask" and "seek" are deliberately absent: they do
#: not make an unsupported positive claim safe.
_NEGATION_CUES = re.compile(
    r"\b(?:not|never|cannot|can't|no\s+(?:result|evidence|indication|sign|screen|image|photograph)|nor|neither|without|un(?:able|confirmed)|"
    r"does\s+not|do\s+not|did\s+not|is\s+not|are\s+not|was\s+not|were\s+not|"
    r"rather\s+than|instead\s+of|refuse[sd]?|declin\w+|avoid|prohibit\w*|"
    r"forbid\w*|must\s+not|should\s+not)\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(
    r"(?:[,;:]|\b(?:and|but|yet|because|while|although|however)\b)", re.IGNORECASE
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")


@dataclass(frozen=True, slots=True)
class ProhibitedClaim:
    """One rule and the reason it exists."""

    code: str
    pattern: re.Pattern[str]
    rationale: str
    #: When True, a negation in the same sentence does not excuse the match.
    absolute: bool = False


@dataclass(frozen=True, slots=True)
class Violation:
    """A specific prohibited claim found in a specific place."""

    code: str
    matched_text: str
    sentence: str
    rationale: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.matched_text!r} in: {self.sentence.strip()!r}"


#: Compounds and preparations a screening tool must never name.
_DRUG_NAMES = (
    "oxytetracycline",
    "tetracycline",
    "enrofloxacin",
    "ciprofloxacin",
    "florfenicol",
    "chloramphenicol",
    "sulfamethoxazole",
    "trimethoprim",
    "nitrofuran\\w*",
    "malachite green",
    "formalin",
    "formaldehyde",
    "copper sulphate",
    "copper sulfate",
    "potassium permanganate",
    "benzalkonium",
    "iodophor",
    "antibiotic\\w*",
    "antimicrobial\\w*",
    "chemotherapeutant\\w*",
)

#: Units in which a dose would be expressed.
_DOSE_UNITS = (
    "mg/l",
    "mg/kg",
    "g/kg",
    "g/l",
    "ml/l",
    "ppm",
    "ppt",
    "iu/kg",
    "mg per litre",
    "mg per liter",
    "parts per million",
)

PROHIBITED_CLAIMS: tuple[ProhibitedClaim, ...] = (
    ProhibitedClaim(
        code="HEALTH_ASSERTION",
        pattern=re.compile(
            r"\b(?:is|are|looks?|appears?|seems?)\s+(?:\w+\s+){0,2}"
            r"(?:healthy|disease[- ]free|infection[- ]free|safe\s+to\s+(?:eat|sell|harvest))\b",
            re.IGNORECASE,
        ),
        rationale=(
            "An image screen cannot establish that an animal is healthy; the absence of "
            "two visible appearances is not the absence of disease."
        ),
    ),
    ProhibitedClaim(
        code="DIAGNOSIS",
        pattern=re.compile(r"\b(?:diagnos(?:e|es|ed|is|tic|tics)|diagnosing)\b", re.IGNORECASE),
        rationale="This system screens for appearances; it does not diagnose.",
    ),
    ProhibitedClaim(
        code="PATHOGEN_CONFIRMATION",
        pattern=re.compile(
            r"\b(?:confirm(?:s|ed|ing|ation)?|prove[sdn]?|verif(?:y|ies|ied))\s+"
            r"(?:\w+\s+){0,3}"
            r"(?:wssv|white\s+spot|infection|infected|disease|pathogen|virus|outbreak)\b",
            re.IGNORECASE,
        ),
        rationale=(
            "Confirmation of a pathogen requires molecular or histological testing. A "
            "photograph cannot supply it."
        ),
    ),
    ProhibitedClaim(
        code="CURE_OR_TREATMENT",
        pattern=re.compile(
            r"\b(?:cure[sd]?|curing|heal\s+the|eradicate[sd]?|"
            r"treat(?:s|ed|ing|ment)?\s+(?:the\s+)?(?:shrimp|pond|infection|disease|stock))\b",
            re.IGNORECASE,
        ),
        rationale="Prescribing or implying a remedy is outside the scope and competence "
        "of this system.",
    ),
    ProhibitedClaim(
        code="DRUG_NAMED",
        pattern=re.compile(r"\b(?:" + "|".join(_DRUG_NAMES) + r")\b", re.IGNORECASE),
        rationale="Naming a compound is a step toward a dose, in any framing.",
        absolute=True,
    ),
    ProhibitedClaim(
        code="DOSE_QUANTITY",
        pattern=re.compile(
            r"\b\d+(?:[.,]\d+)?\s*(?:" + "|".join(re.escape(u) for u in _DOSE_UNITS) + r")\b",
            re.IGNORECASE,
        ),
        rationale="A quantity plus a unit is a dose, regardless of the surrounding words.",
        absolute=True,
    ),
    ProhibitedClaim(
        code="EMS_AHPND_CLAIM",
        pattern=re.compile(
            r"\b(?:detect(?:s|ed|ion)?|screen(?:s|ed|ing)?|identif(?:y|ies|ied))\s+"
            r"(?:\w+\s+){0,3}(?:ems|ahpnd|acute\s+hepatopancreatic)\b",
            re.IGNORECASE,
        ),
        rationale="No available dataset supports EMS/AHPND, so it is out of scope.",
    ),
)


def _sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _directly_negated(sentence: str, claim_start: int) -> bool:
    """Whether the nearest cue directly governs this claim, not another clause."""
    prefix = sentence[max(0, claim_start - 120) : claim_start]
    cues = list(_NEGATION_CUES.finditer(prefix))
    if not cues:
        return False
    between = prefix[cues[-1].end() :]
    return len(between.split()) <= 12 and _CLAUSE_BREAK.search(between) is None


def scan(text: str) -> list[Violation]:
    """Return every prohibited claim in ``text``.

    Contextual rules are excused by a negation or hedge in the same sentence;
    absolute rules are not excused by anything.
    """
    violations: list[Violation] = []
    for sentence in _sentences(text):
        for rule in PROHIBITED_CLAIMS:
            for match in rule.pattern.finditer(sentence):
                if not rule.absolute and _directly_negated(sentence, match.start()):
                    continue
                violations.append(
                    Violation(
                        code=rule.code,
                        matched_text=match.group(0),
                        sentence=sentence,
                        rationale=rule.rationale,
                    )
                )
    return violations


def assert_clean(text: str, *, where: str) -> None:
    """Raise :class:`ValueError` listing every violation in ``text``."""
    violations = scan(text)
    if violations:
        listed = "\n  ".join(str(item) for item in violations)
        raise ValueError(f"prohibited claims in {where}:\n  {listed}")
