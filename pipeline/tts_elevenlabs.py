"""ElevenLabs TTS backend.

Uses the multilingual `eleven_v3` model (or whatever is in config) with two
catalogs of voices:

1. ``_NATIVE_VOICES['en']`` — the 21 official premade voices that ship with
   every ElevenLabs account. All English-recorded but multilingual via the
   model.
2. ``_NATIVE_VOICES[<lang>]`` — handpicked Voice Library voices recorded by
   native speakers of that language (Mandarin, Japanese, Korean, etc.). The
   API exposes these via ``client.voices.get_shared(language=...)``.

For each language the first entry is the primary male, the second is the
primary female, used as defaults when the caller doesn't pick a voice.

Output is fetched as MP3 and converted in one ffmpeg pass to PCM WAV (with
optional tail silence) so the downstream merger sees the same file format
as the Cartesia backend.
"""

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from elevenlabs.client import ElevenLabs

from . import config as _conf
from .costs import CostTracker
from .ffmpeg_utils import FFMPEG_EXE
from .transcriber import Segment


# Voice metadata structure: name → {id, gender, accent, description}.
# Order within a language dict matters: index 0 = primary male, 1 = primary
# female (used as defaults).
_NATIVE_VOICES: dict[str, dict[str, dict]] = {
    "en": {
        # 21 official premade voices. Multilingual via the model.
        "Adam":      {"id": "pNInz6obpgDQGcFmaJgB", "gender": "male",    "accent": "american",  "description": "Bright tenor that cuts through; brash, openly confident, with unwavering certainty."},
        "Sarah":     {"id": "EXAVITQu4vr4xnSDxMaL", "gender": "female",  "accent": "american",  "description": "Young adult woman — confident, warm, mature; reassuring, professional tone."},
        "Brian":     {"id": "nPczCjzI2devNBz1zQrb", "gender": "male",    "accent": "american",  "description": "Middle-aged man with a deep, resonant, comforting tone. Great for narrations and ads."},
        "Bill":      {"id": "pqHfZKP75CvOlQylNhV4", "gender": "male",    "accent": "american",  "description": "Older, friendly and comforting voice — wise, mature, balanced; ready to narrate stories."},
        "Roger":     {"id": "CwhRBWXzGAHq8TQ4Fs17", "gender": "male",    "accent": "american",  "description": "Easy-going, casual, resonant — perfect for laid-back conversations."},
        "Eric":      {"id": "cjVigY5qzO86Huf0OWal", "gender": "male",    "accent": "american",  "description": "Smooth tenor from a man in his 40s — trustworthy, classy, agentic feel."},
        "Chris":     {"id": "iP95p4xoKVk53GoZ742B", "gender": "male",    "accent": "american",  "description": "Natural, real, charming, down-to-earth — versatile for many use cases."},
        "Will":      {"id": "bIHbv24MWmeRgasZH58o", "gender": "male",    "accent": "american",  "description": "Conversational and laid back; relaxed optimist."},
        "Liam":      {"id": "TX3LPaxmHKxFdv7VOQHJ", "gender": "male",    "accent": "american",  "description": "Young adult with energy and warmth — suitable for reels and shorts."},
        "Harry":     {"id": "SOYHLrjzK2X1ezoPC6cr", "gender": "male",    "accent": "american",  "description": "Animated warrior — rough, fierce, ready to charge forward."},
        "Callum":    {"id": "N2lVS1w4EtoT3dr4eOWO", "gender": "male",    "accent": "american",  "description": "Deceptively gravelly with an unsettling edge — husky trickster."},
        "Charlie":   {"id": "IKne3meq5aSn9XLyUdCD", "gender": "male",    "accent": "australian","description": "Young Australian male — confident, energetic, hyped."},
        "George":    {"id": "JBFqnCBsd6RMkjVDRZzb", "gender": "male",    "accent": "british",   "description": "Warm, mature British male — captivating storyteller resonance."},
        "Daniel":    {"id": "onwK4e9ZLuTAKqWW03F9", "gender": "male",    "accent": "british",   "description": "Strong, formal British male — perfect for professional broadcast or news."},
        "Bella":     {"id": "hpp4J3VqNfWAUOO0d1Us", "gender": "female",  "accent": "american",  "description": "Warm, bright, professional middle-aged American female — polished narrative quality."},
        "Matilda":   {"id": "XrExE9yKIg1WjnnlVkGX", "gender": "female",  "accent": "american",  "description": "Professional middle-aged American woman; pleasing alto, upbeat, knowledgeable."},
        "Jessica":   {"id": "cgSgspJ2msm6clMCkdW9", "gender": "female",  "accent": "american",  "description": "Young, playful, bright, warm American female — perfect for trendy content."},
        "Laura":     {"id": "FGY2WhTYpPnrIDTdsKH5", "gender": "female",  "accent": "american",  "description": "Young adult American female — sunny enthusiasm with a quirky, sassy attitude."},
        "Alice":     {"id": "Xb7hH8MSUJpSbSDYk0k2", "gender": "female",  "accent": "british",   "description": "Clear, engaging middle-aged British female — friendly, professional, suitable for e-learning."},
        "Lily":      {"id": "pFZP5JQG7iQjIQuC4Bku", "gender": "female",  "accent": "british",   "description": "Velvety middle-aged British female — confident; news and narrations with warmth and clarity."},
        "River":     {"id": "SAz9YHcvj6GT2YYXdXww", "gender": "neutral", "accent": "american",  "description": "Relaxed, neutral, calm — informative voice ready for narrations or conversational projects."},
    },
    "zh": {
        "Lin":       {"id": "UFDAUkGzdLAEJlINT3Fx", "gender": "male",   "accent": "mandarin",  "description": "Warm, friendly, conversational male — clean articulation; built for real-time conversational AI."},
        "Lingyue":   {"id": "ROfZIBAOQbwvPvLR3GMu", "gender": "female", "accent": "mandarin",  "description": "Pleasant, gentle female — great for audiobooks and narration."},
        "Jing":      {"id": "zYD0xJl1ponKr8TwFBmJ", "gender": "male",   "accent": "mandarin",  "description": "Natural, bright, conversational male — direct delivery, slightly fast pace."},
        "Nina":      {"id": "El018FmI047NtSsCfyrY", "gender": "female", "accent": "mandarin",  "description": "Young, soft yet energetic Chinese female — warm, friendly, uplifting tone."},
    },
    "ja": {
        "Shohei":    {"id": "AIg1eVkw9X3Bz8LsQQS3", "gender": "male",   "accent": "japanese",  "description": "Calm, composed Japanese male in mid-30s — corporate intro and explainer style."},
        "Maiko":     {"id": "deKmbWEKZdwxcKxxcfvP", "gender": "female", "accent": "japanese",  "description": "Calm, slightly low-pitched Japanese female narrator — measured pace."},
        "Yoshio":    {"id": "f7UUeltR22mzvXAsYavl", "gender": "male",   "accent": "japanese",  "description": "Clear, calm Japanese male voice — narration-friendly."},
        "Kana":      {"id": "dhGvgIx0X6G3xzSWqOye", "gender": "female", "accent": "japanese",  "description": "Pleasant, balanced Japanese female in her 30s — bright but calm, soft but clear."},
    },
    "ko": {
        "Joon-ho":   {"id": "NpneagLVR101ytYGxUPX", "gender": "male",   "accent": "korean",    "description": "Mature Korean male — rich, mellow timbre with a naturally reassuring tone."},
        "Soo":       {"id": "3H7nUJE8YbxkmPW1GFts", "gender": "female", "accent": "korean",    "description": "Warm Korean contralto inspired by midnight radio — emotional, lo-fi storytelling."},
        "Jae-won":   {"id": "yvcSjT5PVWkS89U5mD6w", "gender": "male",   "accent": "korean",    "description": "Deep, youthful Korean male with smooth Seoul accent — rich, resonant, commanding."},
        "Onyu":      {"id": "NaQdbkW5gNZD8wfwXeTV", "gender": "female", "accent": "korean",    "description": "Young Korean female — calm, friendly; great for narrations."},
    },
    "es": {
        "Carlos":    {"id": "Y4HwpMNrBWN5dFKi5Lw8", "gender": "male",   "accent": "colombian", "description": "Warm, clear male Colombian Spanish — neutral Latin American; ideal for educational content."},
        "Valeria":   {"id": "nfyTTmgO0f6GV9CKrMWL", "gender": "female", "accent": "neutral",   "description": "Female voice in neutral Latin American Spanish — warm, professional, persuasive."},
        "Brian C":   {"id": "Czw3Dn181ypdrCOnPfif", "gender": "male",   "accent": "neutral",   "description": "Clean, professional, warm young announcer — engaging, approachable, natural."},
        "Paola":     {"id": "PoLFkTquRWtbexdwW3Xa", "gender": "female", "accent": "argentine", "description": "Professional Argentine Spanish (Rioplatense) — neutral, versatile delivery."},
    },
    "fr": {
        "Lior":      {"id": "ljrJ081VvO1DCmonP7R8", "gender": "male",   "accent": "french",    "description": "Natural French male — calm, warm tone; ideal for narration, tech tutorials, podcasts."},
        "Virginie":  {"id": "NzCI2wsmQgzQiufNpYi7", "gender": "female", "accent": "french",    "description": "Soft, melodic, delicate French female — naturally warm and pleasant."},
        "Augustin":  {"id": "kKgyAHjGAbeWHCNd7qoC", "gender": "male",   "accent": "french",    "description": "Standard French male voice — narration, conversation, podcast."},
        "Marilène":  {"id": "tRQeD4idfj7AuhU7ApjT", "gender": "female", "accent": "french",    "description": "Versatile French female narrator — intense, passionate, soft, or whispered."},
    },
    "de": {
        "Daniel":    {"id": "wcqN36SUOZ0EhToc2OIu", "gender": "male",   "accent": "german",    "description": "Natural, calm, conversational German male — warm, authentic; perfect for UGC, ads."},
        "Sina":      {"id": "sgKauqXbUxSBZgugAiOl", "gender": "female", "accent": "german",    "description": "Warm, approachable German female in her 30s — natural narrative delivery."},
        "Arthur":    {"id": "3nMIMZ7RlGwsq1WLgxY3", "gender": "male",   "accent": "german",    "description": "Precise, well-articulated German male with natural authority — narrative & educational."},
        "Helmut":    {"id": "OwYx4Jd2f6DQax9m8SVa", "gender": "male",   "accent": "german",    "description": "Natural, conversational German male — friendly character; ads and casual content."},
    },
    "it": {
        "Raffaele":  {"id": "YGp1lBJLaHhfIFT0yeDE", "gender": "male",   "accent": "italian",   "description": "Mature Italian male in mid-40s — rich, resonant, deep timbre; corporate-grade."},
        "Chiara":    {"id": "mT0eqrjKfAPl6gQBlfBa", "gender": "female", "accent": "italian",   "description": "Italian female ~35 — natural, warm, approachable, professional."},
        "Paolo":     {"id": "mcMi8FJDhg35bMpWHv2R", "gender": "male",   "accent": "italian",   "description": "Dynamic Italian male radio voice — promos, commercials, podcasts."},
        "Cornelia":  {"id": "SKEVNjRKCergbPKum64u", "gender": "female", "accent": "italian",   "description": "Calm, warm, professional Italian female — customer support and narration."},
    },
    "nl": {
        "Ronald":    {"id": "cfzqtNeB4pAo40KOcI0v", "gender": "male",   "accent": "dutch",     "description": "Senior Dutch professional voiceover — warm, articulate, engaging."},
        "Jolanda":   {"id": "DiUBVrSFwkMaPz4XqWvR", "gender": "female", "accent": "dutch",     "description": "Pleasant, soothing Dutch female — natural narrative energy; e-learning."},
        "Thijs":     {"id": "DYfqtQfWhbc1Z0SLiCWw", "gender": "male",   "accent": "dutch",     "description": "Natural, expressive, conversational Dutch male — agent/customer care friendly."},
        "Peter":     {"id": "Kv97WYcYaIv0A06FfXcK", "gender": "male",   "accent": "dutch",     "description": "Beloved Dutch podcast voice — clear, warm, experienced."},
    },
    "ru": {
        "Ivo":       {"id": "hD8aK7CmEPgH3mbFO08e", "gender": "male",   "accent": "russian",   "description": "Calm, chill, narrative Russian male in his 20s-30s — clear native pronunciation."},
        "Xenia":     {"id": "vZopxKc6jrT5l48V9W5w", "gender": "female", "accent": "russian",   "description": "Warm, smooth, silky Russian female in mid-30s — calm, confident storyteller."},
        "Anton":     {"id": "13JzN9jg1ViUP8Pf3uet", "gender": "male",   "accent": "russian",   "description": "Calm, clear, confident Russian male — natural conversational tone."},
        "Morna":     {"id": "9AE7A1Ivnw8jKr8Us0ch", "gender": "female", "accent": "russian",   "description": "High-status Russian female — velvet intellectual authority, realistic narrative."},
    },
    "pt": {
        "Medeiros":  {"id": "E9hPCbqBRCgB3UEiCUuE", "gender": "male",   "accent": "brazilian", "description": "Casual, authentic Brazilian Portuguese male - friendly, conversational, free of drama."},
        "Luna":      {"id": "jotBQRDYDizrWQAbv9VO", "gender": "female", "accent": "brazilian", "description": "Soft, calm, soothing Brazilian female — gentle tone, slow-paced, warm."},
        "Guilherme": {"id": "Mn7FDiiQr3aIwMWsLE7r", "gender": "male",   "accent": "brazilian", "description": "Brazilian male ~45 — low-to-mid range, excellent for narration."},
        "Carolina":  {"id": "4NRXT5DGqWzIcL6iVqtF", "gender": "female", "accent": "brazilian", "description": "Vibrant, young Brazilian female — naturally engaging presence."},
    },
    "hi": {
        "Yatin":     {"id": "46xKhzYfIJMejACevZhq", "gender": "male",   "accent": "hindi",     "description": "Experienced, wise classroom-professor Hindi male — engaging, turns dense material into stories."},
        "Madhusmita":{"id": "vQmxatd5nc5hsnpVWWpJ", "gender": "female", "accent": "hindi",     "description": "Soft, velvety, deeply expressive Hindi female narrator."},
        "Niraj":     {"id": "4aFkTNmCXWi67j9dQQVC", "gender": "male",   "accent": "hindi",     "description": "Deep, rich, cinematic Hindi male — intense documentary narrator."},
        "Kalpana":   {"id": "KBzY47BBXGaDJVIbCIFU", "gender": "female", "accent": "hindi",     "description": "Energetic, expressive, refreshingly real Hindi female creator voice."},
    },
    "tr": {
        "Sinan":     {"id": "Rn6ATQ4HFbyhBC6mze4Z", "gender": "male",   "accent": "turkish",   "description": "Deep, trustworthy, versatile Turkish male — educational tutorials, documentaries."},
        "Aura":      {"id": "wYM8FTSOj72VzJ7kgprn", "gender": "female", "accent": "turkish",   "description": "Versatile, expressive Turkish female — global voice for high-end productions."},
        "Hakan":     {"id": "EJ861k94E3y4EM3kZh71", "gender": "male",   "accent": "turkish",   "description": "Calm, contemplative, fluid Turkish male — natural intimate narrative style."},
        "Dilek":     {"id": "3K9SrW7hHlFO6yMzQgJV", "gender": "female", "accent": "turkish",   "description": "Young Turkish female — good for social media and narration."},
    },
    "pl": {
        "Gregor":    {"id": "xPYhbnzKk9nCh7pnUehm", "gender": "male",   "accent": "polish",    "description": "Polish low male voice 45+ — warm, professional, universal; studio quality."},
        "Jola":      {"id": "mgxZQ8b1CA0Zl9YUTX8Z", "gender": "female", "accent": "polish",    "description": "Middle-aged Polish female — professional voiceover for narration, audiobooks."},
        "Kris":      {"id": "jnAzri3VrjEtIf7MMQ4d", "gender": "male",   "accent": "polish",    "description": "Warm, natural Polish male — calm and engaging; medium-deep pitch."},
        "Asia":      {"id": "Bz1e1clEKwgN71Vx7cxj", "gender": "female", "accent": "polish",    "description": "Warm, feminine Polish female — clear diction; audiobooks, commercials, podcasts."},
    },
    "sv": {
        "Andreas":   {"id": "TIMFVcMCO4bdy7J79GWF", "gender": "male",   "accent": "swedish",   "description": "Deep, manly, calm Swedish male — professional actor with good articulation."},
        "Louise":    {"id": "kpTdKfohzvarfFPnwuHW", "gender": "female", "accent": "swedish",   "description": "Calm, clear, soothing Swedish female — warm narrator presence."},
        "Peter":     {"id": "oJEeOXECH9V31Oci9WHK", "gender": "male",   "accent": "swedish",   "description": "Middle-aged Swedish male — natural, engaging; commercials, podcasts."},
        "Olivia":    {"id": "cLAH1kXlkAivJHxCW601", "gender": "female", "accent": "swedish",   "description": "Bright, youthful, engaging Swedish female — customer service / conversational."},
    },
    "ar": {
        "Faris":     {"id": "RL04FaPrUG6vS8aWd9NZ", "gender": "male",   "accent": "levantine", "description": "Warm middle-aged Levantine Arabic male — gentle, unhurried cadence; pleasant, reassuring."},
        "Haneen":    {"id": "iIIMrkfGuAyFOZsShpgf", "gender": "female", "accent": "levantine", "description": "Poised middle-aged Levantine Arabic female — gentle accent, clear, unhurried."},
        "Ashraf":    {"id": "QfMIySsRuJWjZPZCnDQp", "gender": "male",   "accent": "egyptian",  "description": "Friendly middle-aged Egyptian Arabic male — rich, inviting tone, easygoing."},
        "Sawsan":    {"id": "mS4cERRqrNy5Kmlx8Udf", "gender": "female", "accent": "egyptian",  "description": "Mature Egyptian Arabic female — rich, velvety tone, calm, unhurried."},
    },
}


def native_voices_for(language_code: str) -> list[str]:
    """Return [primary_male, primary_female] voice names for a language.

    Falls back to the English premade voices when the language isn't covered.
    """
    voices = _NATIVE_VOICES.get(language_code) or _NATIVE_VOICES["en"]
    names = list(voices.keys())
    # Pick the first male and first female by gender label, preserving order.
    male = next((n for n in names if voices[n]["gender"] == "male"), names[0] if names else "Adam")
    female = next((n for n in names if voices[n]["gender"] == "female"), names[1] if len(names) > 1 else male)
    return [male, female]


def all_voices() -> dict[str, list[str]]:
    """Return all voices grouped by language code."""
    return {lang: list(voices.keys()) for lang, voices in _NATIVE_VOICES.items()}


def voice_descriptions() -> dict[str, str]:
    """Map voice name → human-readable description for LLM voice matching."""
    out: dict[str, str] = {}
    for voices in _NATIVE_VOICES.values():
        for name, meta in voices.items():
            out[name] = f"{meta['gender']}, {meta['accent']} accent — {meta['description']}"
    return out


def _resolve_voice_id(voice: str) -> str:
    """Accept either a voice name (any language) or a raw voice ID."""
    for voices in _NATIVE_VOICES.values():
        entry = voices.get(voice)
        if entry:
            return entry["id"]
    return voice


def _to_wav(mp3_path: str, wav_path: str, tail_ms: int) -> None:
    """Convert MP3 to mono 44100 PCM WAV, optionally appending silence."""
    af = []
    if tail_ms > 0:
        af = ["-af", f"apad=pad_dur={tail_ms / 1000:.3f}"]
    subprocess.run(
        [FFMPEG_EXE, "-y", "-i", mp3_path,
         *af,
         "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
         wav_path],
        check=True, capture_output=True,
    )


def synthesize_segment(
    text: str,
    voice: str,
    output_path: str,
    client: ElevenLabs,
    language: str = "en",
    speed: float | None = None,
    emotion: str | None = None,
) -> str:
    """Synthesize one segment to a WAV file via ElevenLabs."""
    cfg = _conf.get()
    tts_entry = cfg["models"]["tts"]
    model_id = tts_entry["model"] if isinstance(tts_entry, dict) else "eleven_v3"

    voice_settings = {}
    if speed is not None and 0.7 <= speed <= 1.2:
        voice_settings["speed"] = speed

    kwargs = dict(
        voice_id=_resolve_voice_id(voice),
        text=text,
        model_id=model_id,
        output_format="mp3_44100_128",
    )
    if voice_settings:
        kwargs["voice_settings"] = voice_settings

    audio = client.text_to_speech.convert(**kwargs)

    mp3_path = output_path + ".tmp.mp3"
    with open(mp3_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)

    tcfg = cfg.get("tts", {})
    if re.search(r'[.!?。！？]\s*$', text):
        tail_ms = tcfg.get("sentence_tail_silence_ms", tcfg.get("tail_silence_ms", 0))
    else:
        tail_ms = tcfg.get("tail_silence_ms", 0)
    _to_wav(mp3_path, output_path, tail_ms)
    Path(mp3_path).unlink(missing_ok=True)
    return output_path


def synthesize_segments(
    segments: list[Segment],
    voice: str,
    output_dir: str,
    client: ElevenLabs,
    language: str = "en",
    voice_map: dict[str, str] | None = None,
    tracker: CostTracker | None = None,
    speed: float | None = None,
    emotion: str | None = None,
) -> list[str]:
    total = len(segments)
    paths = [""] * total
    vm = voice_map or {}

    def _do(idx: int, seg: Segment) -> tuple[int, str]:
        path = str(Path(output_dir) / f"seg_{seg.id:05d}.wav")
        seg_voice = vm.get(seg.speaker, voice)
        synthesize_segment(seg.text, seg_voice, path, client, language, speed, emotion)
        if tracker:
            tracker.add_tts_usage(len(seg.text))
        return idx, path

    done_count = 0
    with ThreadPoolExecutor(max_workers=_conf.get()["tts"]["workers"]) as pool:
        futures = {pool.submit(_do, i, seg): i for i, seg in enumerate(segments)}
        for future in as_completed(futures):
            idx, path = future.result()
            paths[idx] = path
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"      TTS progress: {done_count}/{total} segments done")

    return paths
