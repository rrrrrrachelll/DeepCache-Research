"""Deterministic, prompt-disjoint week-one prompt set."""
from __future__ import annotations

import hashlib
import json


CATEGORIES = {
    "portrait": {
        "subjects": ["an elderly violin maker", "a young marine biologist", "a middle-aged baker", "a retired astronaut", "a street photographer"],
        "settings": ["in a softly lit workshop", "beside a rain-covered window", "in a quiet library"],
        "details": ["natural skin texture and expressive eyes", "carefully detailed hands and clothing", "subtle rim lighting and realistic hair"],
    },
    "people": {
        "subjects": ["two dancers rehearsing", "three scientists discussing a map", "a family preparing dinner", "four hikers resting", "two mechanics repairing a bicycle"],
        "settings": ["inside a bright community hall", "at an outdoor market", "under warm evening light"],
        "details": ["distinct faces and complete bodies", "clear interactions between every person", "natural poses without merged limbs"],
    },
    "animal": {
        "subjects": ["a red fox", "a barn owl", "two emperor penguins", "a golden retriever", "a small green tree frog"],
        "settings": ["in fresh winter snow", "among wet forest leaves", "beside a calm mountain lake"],
        "details": ["highly detailed fur or feathers", "a sharp wildlife photograph", "natural anatomy and soft daylight"],
    },
    "architecture": {
        "subjects": ["a Gothic railway station", "a modern glass museum", "a narrow Mediterranean house", "an Art Deco theater", "a wooden mountain temple"],
        "settings": ["at blue hour", "after a light rain", "in clear morning sunlight"],
        "details": ["symmetrical facade and fine materials", "straight structural lines and realistic perspective", "detailed entrances, windows, and roof"],
    },
    "interior": {
        "subjects": ["a compact Japanese kitchen", "a Victorian reading room", "a modern ceramics studio", "a Scandinavian bedroom", "an old neighborhood cafe"],
        "settings": ["with sunlight entering from the left", "illuminated by warm practical lamps", "photographed on an overcast afternoon"],
        "details": ["coherent furniture layout", "realistic materials and uncluttered geometry", "many small but clearly separated objects"],
    },
    "objects": {
        "subjects": ["a brass compass, a folded map, and two blue pencils", "three pears, a white bowl, and a silver spoon", "a camera, two film rolls, and red glasses", "four seashells, a candle, and a green bottle", "a watch, an old key, and two postcards"],
        "settings": ["arranged on a dark wooden table", "placed on pale linen", "displayed on a reflective metal shelf"],
        "details": ["every object fully visible", "accurate materials and contact shadows", "a clean still-life composition"],
    },
    "spatial": {
        "subjects": ["a red cube left of a blue sphere", "a bicycle behind a yellow bench", "a cat beneath a wooden chair", "a green vase between two white books", "a small boat in front of a lighthouse"],
        "settings": ["in a simple studio scene", "with an uncluttered background", "under neutral daylight"],
        "details": ["the spatial relationship must be unambiguous", "all named objects clearly separated", "realistic perspective and shadows"],
    },
    "counting": {
        "subjects": ["exactly three red balloons", "exactly five oranges", "exactly two ceramic birds", "exactly four wooden chairs", "exactly six seashells"],
        "settings": ["against a plain gray background", "on a clean white table", "in a sunlit room"],
        "details": ["with no additional repeated objects", "all items non-overlapping and countable", "a centered photographic composition"],
    },
    "texture": {
        "subjects": ["handwoven red and gold fabric", "weathered blue painted wood", "a cracked glazed ceramic tile", "dark green moss on stone", "a close-up of layered paper fibers"],
        "settings": ["in a macro photograph", "under soft raking light", "filling the entire frame"],
        "details": ["individual surface details clearly visible", "sharp micro-texture without blur", "natural color variation and depth"],
    },
    "text": {
        "subjects": ["a flower shop sign reading BLOOM", "a cafe chalkboard reading TODAY SOUP", "a blue book cover titled OCEAN", "a theater marquee reading NIGHT SHOW", "a parcel label reading NORTH"],
        "settings": ["viewed directly from the front", "in even daylight", "with a simple surrounding scene"],
        "details": ["the requested words large and legible", "no extra letters near the requested text", "clean typography and realistic materials"],
    },
    "landscape": {
        "subjects": ["a turquoise river through a canyon", "terraced fields below snowy mountains", "a rocky coast with a distant lighthouse", "a desert valley after rain", "a quiet lake surrounded by autumn trees"],
        "settings": ["at sunrise", "under dramatic clouds", "in crisp afternoon light"],
        "details": ["deep foreground-to-background detail", "natural atmospheric perspective", "a realistic wide-angle photograph"],
    },
    "art": {
        "subjects": ["a floating city above an ocean", "a mechanical bird carrying a seed", "a library inside a giant shell", "a moonlit train crossing the sky", "a garden of translucent crystal plants"],
        "settings": ["as a detailed watercolor illustration", "as a cinematic concept painting", "as a richly textured paper collage"],
        "details": ["coherent composition and lighting", "fine intentional details", "a restrained harmonious color palette"],
    },
}


def build_prompts() -> list[dict]:
    rows = []
    categories = list(CATEGORIES)
    for category_index, category in enumerate(categories):
        spec = CATEGORIES[category]
        for index in range(15):
            subject = spec["subjects"][index % 5]
            setting = spec["settings"][(index // 5) % 3]
            detail = spec["details"][(index * 2 + index // 5) % 3]
            if index < 10:
                split = "train"
            elif index < 12:
                split = "validation"
            elif index < 14:
                split = "development"
            else:
                split = "validation" if category_index % 2 == 0 else "development"
            rows.append({
                "id": f"{category}_{index:02d}",
                "category": category,
                "split": split,
                "prompt": f"A high quality image of {subject} {setting}, {detail}.",
            })
    assert len(rows) == 180
    assert len({row["id"] for row in rows}) == 180
    assert {split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "development")} == {
                "train": 120, "validation": 30, "development": 30}
    return rows


def prompt_hash() -> str:
    payload = json.dumps(build_prompts(), sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


SEEDS = {"train": (1701,), "validation": (2101, 2201), "development": (3101, 3201)}
