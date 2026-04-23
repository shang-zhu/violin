"""Shared language code mapping for the Violin pipeline."""

# BCP-47 codes for Cartesia Sonic 3's 42 supported languages.
LANG_CODES: dict[str, str] = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
    "italian": "it",
    "dutch": "nl",
    "russian": "ru",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "hindi": "hi",
    "turkish": "tr",
    "polish": "pl",
    "swedish": "sv",
    "arabic": "ar",
    "indonesian": "id",
    "vietnamese": "vi",
    "thai": "th",
    "greek": "el",
    "czech": "cs",
    "danish": "da",
    "finnish": "fi",
    "norwegian": "no",
    "romanian": "ro",
    "slovak": "sk",
    "ukrainian": "uk",
    "hungarian": "hu",
    "catalan": "ca",
    "bulgarian": "bg",
    "croatian": "hr",
    "malay": "ms",
    "tamil": "ta",
}


def language_code(language: str) -> str:
    """Convert a language name to a BCP-47 code, falling back to a 2-char slice."""
    return LANG_CODES.get(language.lower(), language.lower()[:2])


def all_languages() -> dict[str, str]:
    """Return the full name → BCP-47 code mapping."""
    return dict(LANG_CODES)
