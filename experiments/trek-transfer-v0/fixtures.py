"""Small public factual QA fixtures. No scripts, dialogue, or sealed oracle items."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    answer: str
    source_type: str
    source_url: str


# Independently encoded structured facts; the linked sources support the facts.
# These *same* pairs train the donor and probe retention. They are NOT unseen
# factual knowledge or evidence of English paraphrase generalization.
TREK_FACTS = (
    Fact("VOYAGER", "CAPTAIN", "JANEWAY", "official", "https://www.startrek.com/series/star-trek-voyager"),
    Fact("VOYAGER", "REGISTRY", "NCC_74656", "official", "https://www.startrek.com/news/celebrating-the-ships-of-the-line-uss-voyager-ncc-74656"),
    Fact("VOYAGER", "CLASS", "INTREPID", "official", "https://www.startrek.com/news/celebrating-the-ships-of-the-line-uss-voyager-ncc-74656"),
    Fact("SPOCK", "HERITAGE", "HUMAN_VULCAN", "official", "https://www.startrek.com/news/ethan-peck-cast-as-discovery-s-spock"),
    Fact("DS9", "COMMANDER", "SISKO", "official", "https://www.startrek.com/series/deep-space-nine"),
    Fact("WORF", "SPECIES", "KLINGON", "official", "https://www.startrek.com/news/the-wisdom-of-worf"),
)


# Original noncanonical strings are separated from the canonical Trek fixture.
# They are available for a neutral methodological repeat, never labeled canon.
SYNTHETIC_FACTS = (
    Fact("KEPLER_HARBOR", "CAPTAIN", "LUMA", "synthetic_noncanon", ""),
    Fact("KEPLER_HARBOR", "REGISTRY", "H_117", "synthetic_noncanon", ""),
    Fact("AURORA_9", "CLASS", "CIRRUS", "synthetic_noncanon", ""),
    Fact("SOLARIS_OUTPOST", "COMMANDER", "VERIN", "synthetic_noncanon", ""),
    Fact("MIRA", "HERITAGE", "RIVERBORN", "synthetic_noncanon", ""),
    Fact("TORO", "SPECIES", "SKYFOLK", "synthetic_noncanon", ""),
)
