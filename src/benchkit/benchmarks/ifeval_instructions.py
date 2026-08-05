"""Verifiable instruction checkers used by the IFEval benchmark.

Each checker mirrors the reference implementation shipped with the IFEval
paper (google-research/instruction_following_eval). The reference code leans on
NLTK for tokenization and langdetect for `language:response_language`; both are
reimplemented here with small dependency-free equivalents so BenchKit stays
installable without extra model or corpus downloads.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable

Checker = Callable[[str, dict], bool]

_COMPARISONS = {
    "less than": lambda value, target: value < target,
    "at least": lambda value, target: value >= target,
}

_CONSTRAINED_RESPONSES = (
    "my answer is yes.",
    "my answer is no.",
    "my answer is maybe.",
)

_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "fig",
    "no",
    "inc",
    "ltd",
    "co",
    "approx",
    "u.s",
    "u.k",
}


def _compare(value: int, relation: str, target: int) -> bool:
    compare = _COMPARISONS.get(str(relation).strip().lower())
    if compare is None:
        raise ValueError(f"unsupported relation: {relation!r}")
    return compare(value, target)


# Tokenization ---------------------------------------------------------


def count_words(text: str) -> int:
    """Count word tokens the way the reference tokenizer (`\\w+`) does."""
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a small abbreviation-aware splitter."""
    sentences: list[str] = []
    current: list[str] = []

    for chunk in re.split(r"(?<=[.!?])[\"')\]]*\s+", text.strip()):
        if not chunk:
            continue
        current.append(chunk)
        tail = chunk.rstrip("\"')]").rstrip()
        last_word = re.split(r"[\s]", tail)[-1].rstrip(".").lower()
        ends_sentence = tail.endswith((".", "!", "?"))
        if ends_sentence and last_word not in _ABBREVIATIONS:
            sentences.append(" ".join(current))
            current = []

    if current:
        sentences.append(" ".join(current))

    return [sentence for sentence in sentences if sentence.strip()]


def count_sentences(text: str) -> int:
    return len(split_sentences(text))


# Language detection ---------------------------------------------------

_SCRIPT_RANGES: list[tuple[int, int, str]] = [
    (0x0590, 0x05FF, "he"),
    (0x0600, 0x06FF, "arabic"),
    (0x0750, 0x077F, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
    (0x0D80, 0x0DFF, "si"),
    (0x0E00, 0x0E7F, "th"),
    (0x0400, 0x04FF, "cyrillic"),
    (0x0370, 0x03FF, "el"),
    (0x1100, 0x11FF, "ko"),
    (0x3130, 0x318F, "ko"),
    (0xAC00, 0xD7AF, "ko"),
    (0x3040, 0x30FF, "ja"),
    (0x4E00, 0x9FFF, "zh"),
]

# fmt: off
_LATIN_STOPWORDS: dict[str, set[str]] = {
    "en": {
        "the", "and", "that", "this", "with", "for", "you", "are", "was",
        "have", "not", "but", "they", "from", "your", "will", "can", "there",
        "what", "which", "about", "would", "should", "here",
    },
    "de": {
        "der", "die", "das", "und", "ist", "nicht", "ich", "ein", "eine",
        "mit", "für", "sich", "auch", "auf", "dass", "sie", "wir", "aber",
    },
    "it": {
        "il", "la", "di", "che", "non", "per", "una", "sono", "con", "del",
        "gli", "come", "più", "anche", "questo", "essere",
    },
    "pt": {
        "de", "que", "não", "para", "uma", "com", "os", "as", "do", "da",
        "mais", "por", "como", "está", "você", "são",
    },
    "es": {
        "el", "los", "las", "por", "con", "para", "una", "pero", "más",
        "está", "son", "este", "muy",
    },
    "fr": {
        "le", "les", "des", "une", "est", "pour", "dans", "pas", "avec",
        "sur", "vous", "nous", "plus", "cette",
    },
    "fi": {
        "ja", "on", "ei", "että", "se", "olla", "joka", "kuin", "myös",
        "mutta", "kun", "hän", "tai", "ovat",
    },
    "sw": {
        "na", "ya", "kwa", "ni", "wa", "katika", "za", "la", "hii", "kuwa",
        "yake", "hata", "wale", "ambayo",
    },
    "nl": {"het", "een", "van", "en", "is", "niet", "dat", "op", "met", "zijn"},
}
# fmt: on

_VIETNAMESE_MARKS = set("ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")

_DEVANAGARI_HINTS = {
    "mr": {"आहे", "आणि", "मध्ये", "यांनी", "त्या", "करण्यात", "नाही", "होते"},
    "ne": {"छ", "हो", "गर्न", "भएको", "लागि", "पनि", "गरेको", "थियो"},
    "hi": {"है", "हैं", "और", "के", "में", "का", "की", "को", "यह", "नहीं"},
}

_ARABIC_HINTS = {
    "ur": {"ہے", "کے", "میں", "اور", "کی", "کا", "نہیں", "ہیں"},
    "fa": {"است", "را", "این", "که", "می", "برای", "با", "های"},
    "ar": {"في", "من", "على", "التي", "هذا", "أن", "إلى", "مع"},
}

_CYRILLIC_HINTS = {
    "bg": {"на", "се", "да", "който", "като", "със", "това", "са"},
    "ru": {"это", "что", "как", "для", "они", "был", "она", "или"},
    "uk": {"це", "що", "як", "для", "він", "вона", "або", "той"},
}


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)


def _best_hint(text: str, hints: dict[str, set[str]], default: str) -> str:
    words = set(_words(text))
    scores = {code: len(words & markers) for code, markers in hints.items()}
    best = max(scores, key=lambda code: scores[code])
    return best if scores[best] else default


def _dominant_script(text: str) -> str | None:
    counts: dict[str, int] = {}
    latin = 0

    for char in text:
        if not char.isalpha():
            continue
        code = ord(char)
        if code < 0x0250 or 0x1E00 <= code <= 0x1EFF:
            latin += 1
            continue
        for start, end, script in _SCRIPT_RANGES:
            if start <= code <= end:
                counts[script] = counts.get(script, 0) + 1
                break

    if counts:
        script = max(counts, key=lambda key: counts[key])
        if counts[script] >= max(latin, 1):
            return script
    return "latin" if latin else None


def detect_language(text: str) -> str | None:
    """Best-effort ISO 639-1 language code for a response.

    Non-Latin scripts are identified by Unicode block and then narrowed with
    stopword hints where a block is shared (Devanagari, Arabic, Cyrillic).
    Latin-script languages are scored purely on stopword overlap.
    """
    text = text.strip()
    if not text:
        return None

    script = _dominant_script(text)
    if script is None:
        return None
    if script == "devanagari":
        return _best_hint(text, _DEVANAGARI_HINTS, "hi")
    if script == "arabic":
        return _best_hint(text, _ARABIC_HINTS, "ar")
    if script == "cyrillic":
        return _best_hint(text, _CYRILLIC_HINTS, "ru")
    if script != "latin":
        return script

    lowered = unicodedata.normalize("NFC", text.lower())
    if any(char in _VIETNAMESE_MARKS for char in lowered):
        return "vi"

    words = set(_words(text))
    scores = {code: len(words & markers) for code, markers in _LATIN_STOPWORDS.items()}
    best = max(scores, key=lambda code: scores[code])
    return best if scores[best] else "en"


# Instruction checkers -------------------------------------------------


def _keywords_existence(response: str, kwargs: dict) -> bool:
    lowered = response.lower()
    return all(keyword.lower() in lowered for keyword in kwargs["keywords"])


def _keywords_frequency(response: str, kwargs: dict) -> bool:
    found = len(re.findall(re.escape(kwargs["keyword"]), response, flags=re.IGNORECASE))
    return _compare(found, kwargs["relation"], int(kwargs["frequency"]))


def _keywords_forbidden(response: str, kwargs: dict) -> bool:
    return not any(
        re.search(rf"\b{re.escape(word)}\b", response, flags=re.IGNORECASE)
        for word in kwargs["forbidden_words"]
    )


def _letter_frequency(response: str, kwargs: dict) -> bool:
    letter = kwargs["letter"].lower()
    found = response.lower().count(letter)
    return _compare(found, kwargs["let_relation"], int(kwargs["let_frequency"]))


def _response_language(response: str, kwargs: dict) -> bool:
    return detect_language(response) == kwargs["language"]


def _number_sentences(response: str, kwargs: dict) -> bool:
    return _compare(
        count_sentences(response), kwargs["relation"], int(kwargs["num_sentences"])
    )


def _number_words(response: str, kwargs: dict) -> bool:
    return _compare(count_words(response), kwargs["relation"], int(kwargs["num_words"]))


def _number_paragraphs(response: str, kwargs: dict) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    num_paragraphs = int(kwargs["num_paragraphs"])

    for index, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            continue
        # A leading or trailing empty chunk just means the divider sat at the
        # very start or end of the response; anything else is a real miss.
        if index in (0, len(paragraphs) - 1):
            num_paragraphs += 1
        else:
            return False

    return len(paragraphs) == num_paragraphs


def _nth_paragraph_first_word(response: str, kwargs: dict) -> bool:
    num_paragraphs = int(kwargs["num_paragraphs"])
    nth = int(kwargs["nth_paragraph"])
    first_word = kwargs["first_word"].lower()

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\n", response)
        if paragraph.strip()
    ]
    if len(paragraphs) != num_paragraphs or nth > len(paragraphs) or nth <= 0:
        return False

    paragraph = paragraphs[nth - 1].strip()
    words = re.findall(r"[\w']+", paragraph)
    return bool(words) and words[0].lower() == first_word


def _number_placeholders(response: str, kwargs: dict) -> bool:
    placeholders = re.findall(r"\[.*?\]", response)
    return len(placeholders) >= int(kwargs["num_placeholders"])


def _postscript(response: str, kwargs: dict) -> bool:
    marker = kwargs["postscript_marker"].strip().lower()
    if marker == "p.p.s":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "p.s.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(marker) + r".*$"
    return bool(re.search(pattern, response.lower(), flags=re.MULTILINE))


def _number_bullets(response: str, kwargs: dict) -> bool:
    bullets = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    bullets += re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    return len(bullets) == int(kwargs["num_bullets"])


def _constrained_response(response: str, _kwargs: dict) -> bool:
    lowered = response.strip().lower()
    return any(option in lowered for option in _CONSTRAINED_RESPONSES)


def _number_highlighted_sections(response: str, kwargs: dict) -> bool:
    count = 0
    for match in re.findall(r"\*[^\n\*]*\*", response):
        if match.strip("*").strip():
            count += 1
    for match in re.findall(r"\*\*[^\n\*]*\*\*", response):
        if match.strip("*").strip():
            count += 1
    return count >= int(kwargs["num_highlights"])


def _multiple_sections(response: str, kwargs: dict) -> bool:
    splitter = re.escape(kwargs["section_spliter"])
    sections = re.split(rf"\s?{splitter}\s?\d+\s?", response)
    return len(sections) - 1 >= int(kwargs["num_sections"])


def _json_format(response: str, _kwargs: dict) -> bool:
    text = response.strip()
    text = re.sub(r"^```(?:json|JSON)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        json.loads(text)
    except (ValueError, TypeError):
        return False
    return True


def _title(response: str, _kwargs: dict) -> bool:
    return any(
        match.strip("<>").strip() for match in re.findall(r"<<[^\n]+>>", response)
    )


def _two_responses(response: str, _kwargs: dict) -> bool:
    parts = response.split("******")
    if len(parts) != 2:
        return False
    return all(part.strip() for part in parts)


def _repeat_prompt(response: str, kwargs: dict) -> bool:
    prompt = kwargs["prompt_to_repeat"].strip().lower()
    return response.strip().lower().startswith(prompt)


def _end_checker(response: str, kwargs: dict) -> bool:
    return response.strip().lower().endswith(kwargs["end_phrase"].strip().lower())


def _quotation(response: str, _kwargs: dict) -> bool:
    text = response.strip()
    return len(text) > 1 and text.startswith('"') and text.endswith('"')


def _is_english(response: str) -> bool:
    # The reference checkers pair the case rule with a language check and treat
    # an undetectable response as a pass.
    detected = detect_language(response)
    return detected in (None, "en")


def _english_capital(response: str, _kwargs: dict) -> bool:
    return response.isupper() and _is_english(response)


def _english_lowercase(response: str, _kwargs: dict) -> bool:
    return response.islower() and _is_english(response)


def _capital_word_frequency(response: str, kwargs: dict) -> bool:
    # Whitespace tokens with the outer punctuation trimmed, so abbreviations
    # such as "P.P.S" count once rather than once per letter.
    words = [token.strip(".,;:!?\"'()[]{}") for token in response.split()]
    capitals = [
        word
        for word in words
        if word and word.isupper() and any(char.isalpha() for char in word)
    ]
    return _compare(
        len(capitals), kwargs["capital_relation"], int(kwargs["capital_frequency"])
    )


def _no_comma(response: str, _kwargs: dict) -> bool:
    return "," not in response and "，" not in response


CHECKERS: dict[str, Checker] = {
    "change_case:capital_word_frequency": _capital_word_frequency,
    "change_case:english_capital": _english_capital,
    "change_case:english_lowercase": _english_lowercase,
    "combination:repeat_prompt": _repeat_prompt,
    "combination:two_responses": _two_responses,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:json_format": _json_format,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:number_bullet_lists": _number_bullets,
    "detectable_format:number_highlighted_sections": _number_highlighted_sections,
    "detectable_format:title": _title,
    "keywords:existence": _keywords_existence,
    "keywords:forbidden_words": _keywords_forbidden,
    "keywords:frequency": _keywords_frequency,
    "keywords:letter_frequency": _letter_frequency,
    "language:response_language": _response_language,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_words": _number_words,
    "punctuation:no_comma": _no_comma,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
}


def follows_instruction(instruction_id: str, response: str, kwargs: dict) -> bool:
    """Return whether a response satisfies one verifiable instruction."""
    checker = CHECKERS.get(instruction_id)
    if checker is None:
        raise KeyError(f"unknown IFEval instruction: {instruction_id}")
    if not response.strip():
        return False
    try:
        return bool(checker(response, kwargs))
    except (KeyError, ValueError, TypeError):
        return False
