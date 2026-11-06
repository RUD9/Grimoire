"""Dependency-free helpers. These checks are heuristics, not semantic grading."""
import re
from decimal import Decimal, InvalidOperation


def final_answer(trace: str) -> str:
    """Extract only text after the last protocol block; reject incomplete blocks.

    Also supports tool calls terminated by the native end-of-turn token: an
    executed call is followed by a closed tool_response in generate_text().
    """
    text = trace.rsplit('</tool_response>', 1)[-1]
    if '<tool_response>' in text:
        return ''
    if '</tool_call>' in text:
        text = text.rsplit('</tool_call>', 1)[-1]
    if '<tool_call>' in text:
        return ''
    return text.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()


def contains_number(text: str, expected: str) -> bool:
    """Exact numeric-token match (not substring); not a correctness proof."""
    try:
        target = Decimal(expected.replace(',', '.'))
    except InvalidOperation:
        return False
    tokens = re.findall(r'(?<![\w.,+\-])[-+]?\d+(?:[.,]\d+)?(?!\w|[.,]\d)', text)
    return any(Decimal(token.replace(',', '.')) == target for token in tokens)


def contains_fact(text: str, fact: str) -> bool:
    """Legacy textual marker check, applied ONLY to final assistant text."""
    return fact.casefold() in text.casefold()
