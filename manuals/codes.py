"""Code-aware text for BM25 (rung 2): add one canonical token per fault code, on chunks AND questions.

BM25 splits "WARNING 23" into two common words ("warn" is on hundreds of chunks, "23" is a parameter and
terminal number too), and "A13" never matches "ALARM 13". So next to every code we append a canonical token:

    WARNING 23 / W23                         -> w23
    ALARM 13 / A13 / WARNING/ALARM 13        -> a13 (and w13)
    F-03 / F03 / F3 / fault 3                -> f03 f3
    Error 6 / E-6                            -> e6
    7 flashes / 7 blinks / flashing 7 times  -> flash7
    P-450 / P450 / p 450, FC 202 / FC202     -> p450, fc202
    0416-8371                                -> 04168371

The original words stay: tokens are only ADDED, so nothing BM25 matched before stops matching. A canonical
token sits on very few chunks, so BM25's IDF weights it heavily: an exact code match now outweighs common words.
Dense search keeps the plain text (it reads meaning, not tokens).
"""

import re

# keyword + number: "warning 23", "alarm 13", "error 6", "fault 3" (the keyword's first letter is the prefix)
KEYWORD_NUM = re.compile(r"\b(warning|alarm|error|fault)s?\s*(?:/\s*alarm\s*)?#?\s*(\d{1,3})\b", re.I)
# "warning/alarm 13" belongs to both families
BOTH_NUM = re.compile(r"\bwarning\s*/\s*alarm\s+(\d{1,3})\b", re.I)
# short letter code, optionally split by "-" or a space: "F-03", "A13", "P 450", "FC 202", "GA-500"
LETTER_CODE = re.compile(r"\b([A-Za-z]{1,3})([- ]?)(\d{1,4})\b")
# flash / blink counts in either order
FLASH = re.compile(r"\b(\d)[- ]?(?:flash|flashes|blink|blinks)\b|\bflash(?:es|ing)?:?\s*(\d)(?:\s*times)?\b", re.I)
# part numbers split by a dash: "0416-8371" (both halves 3+ digits, so parameter ranges like "1-20" are left alone)
SPLIT_DIGITS = re.compile(r"\b(\d{3,})-(\d{3,})\b")
# words that look like letter codes when followed by a separate number ("at 50", "to 100", "a 5%"); glued to the
# number ("A13") they are always codes
NOT_CODES = {"a", "an", "i", "at", "to", "in", "on", "of", "or", "by", "is", "be", "up", "as", "no", "if",
             "and", "the", "for", "all", "set", "max", "min", "sec", "ms", "hz", "kw", "hp", "ph", "mm", "db", "v"}


def code_tokens(text: str) -> list[str]:
    out: list[str] = []
    for word, num in KEYWORD_NUM.findall(text):
        out.append(f"{word[0].lower()}{int(num)}")
    for num in BOTH_NUM.findall(text):
        out += [f"w{int(num)}", f"a{int(num)}"]
    for letters, sep, num in LETTER_CODE.findall(text):
        if sep and letters.lower() in NOT_CODES:
            continue
        code = f"{letters.lower()}{num}"
        out.append(code)
        if num.startswith("0") and len(num) > 1:  # F03 -> also f3, so "fault 3" and "F3" meet it
            out.append(f"{letters.lower()}{int(num)}")
    for a, b in FLASH.findall(text):
        out.append(f"flash{a or b}")
    for a, b in SPLIT_DIGITS.findall(text):
        out.append(a + b)
    return list(dict.fromkeys(out))  # once each: a repeated token would count double in BM25


def with_codes(text: str) -> str:
    """The text BM25 indexes in rung 2: the original plus its canonical code tokens."""
    tokens = code_tokens(text)
    return f"{text}\n{' '.join(tokens)}" if tokens else text
