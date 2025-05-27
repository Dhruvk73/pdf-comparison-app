import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Cleans a given text string by stripping whitespace from each line
    and joining non-empty lines.
    """
    if not text:
        return ""
    if "\n" in text:
        lines = text.splitlines()
        processed_lines = []
        for line_content in lines:
            stripped_line = line_content.strip()
            if stripped_line:
                processed_lines.append(" ".join(stripped_line.split()))
        return "\n".join(processed_lines)
    else:
        stripped_line = text.strip()
        if stripped_line:
            return " ".join(stripped_line.split())
        return ""
