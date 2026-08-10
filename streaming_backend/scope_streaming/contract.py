from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final = "scope-streaming.v1"
DENSE_THRESHOLD: Final = 0.85
NUM_CLASSES: Final = 25
SAMPLE_RATE: Final = 16_000
SAMPLES_PER_CHUNK: Final = 16_000

LLP_CLASSES: Final[tuple[str, ...]] = (
    "Speech",
    "Car",
    "Cheering",
    "Dog",
    "Cat",
    "Frying_(food)",
    "Basketball_bounce",
    "Fire_alarm",
    "Chainsaw",
    "Cello",
    "Banjo",
    "Singing",
    "Chicken_rooster",
    "Violin_fiddle",
    "Vacuum_cleaner",
    "Baby_laughter",
    "Accordion",
    "Lawn_mower",
    "Motorcycle",
    "Helicopter",
    "Acoustic_guitar",
    "Telephone_bell_ringing",
    "Baby_cry_infant_cry",
    "Blender",
    "Clapping",
)

BRANCHES: Final = ("audio", "visual", "audio_visual")
