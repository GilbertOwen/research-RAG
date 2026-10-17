"""Check the [S#] citations in an answer against the sources that were actually given."""

import re

CITATION = re.compile(r"\[(S\d+(?:\s*[,;]\s*S?\d+)*)\]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def labels_in(text: str) -> list[int]:
    return [int(n) for group in CITATION.findall(text) for n in re.findall(r"\d+", group)]


def split_sentences(answer: str) -> list[str]:
    """Sentences, with a citation written after the full stop ("... recall. [S1]") kept on its sentence."""
    sentences = []
    for piece in SENTENCE_END.split(answer.strip()):
        leading = re.match(r"^((?:\[[^\]]*\]\s*)+)(.*)$", piece)
        if leading and sentences and CITATION.search(leading.group(1)):
            sentences[-1] += " " + leading.group(1).strip()
            piece = leading.group(2)
        if piece.strip():
            sentences.append(piece.strip())
    return sentences


def check_citations(answer: str, n_sources: int, no_evidence_reply: str) -> dict:
    if answer.strip().upper().startswith(no_evidence_reply.upper()):
        return {"abstained": True, "cited": [], "invalid": [], "uncited_sentences": [], "ok": True}
    used = labels_in(answer)
    invalid = sorted({n for n in used if not 1 <= n <= n_sources})
    uncited = [s for s in split_sentences(answer) if not CITATION.search(s)]
    return {
        "abstained": False,
        "cited": sorted({n for n in used if 1 <= n <= n_sources}),
        "invalid": invalid,
        "uncited_sentences": uncited,
        "ok": not invalid and not uncited and bool(used),
    }
