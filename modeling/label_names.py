from __future__ import annotations


# English names follow the class-name table published with BST. The ordering is
# part of the checkpoint contract and must not change.
STROKE_TYPES_ENGLISH = [
    "net shot",
    "return net",
    "smash",
    "lob",
    "clear",
    "drive",
    "drop",
    "push",
    "rush",
    "cross-court net shot",
    "short service",
    "long service",
]

STROKE_TYPE_ENGLISH_BY_SOURCE = {
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "挑球": "lob",
    "長球": "clear",
    "平球": "drive",
    "切球": "drop",
    "推球": "push",
    "撲球": "rush",
    "勾球": "cross-court net shot",
    "發短球": "short service",
    "發長球": "long service",
    "未知球種": "none",
    "點扣": "wrist smash",
    "防守回挑": "defensive return lob",
    "後場抽平球": "back-court drive",
    "過渡切球": "passive drop",
    "防守回抽": "defensive return drive",
    "小平球": "small drive",
}


def get_merged_stroke_types_english() -> list[str]:
    return ["none"] + ["Top_" + stroke for stroke in STROKE_TYPES_ENGLISH] + [
        "Bottom_" + stroke for stroke in STROKE_TYPES_ENGLISH
    ]


def translate_stroke_type(value: str) -> str:
    return STROKE_TYPE_ENGLISH_BY_SOURCE.get(value, value)


def translate_side_aware_label(value: str) -> str:
    if "_" not in value:
        return translate_stroke_type(value)
    side, stroke = value.split("_", 1)
    return f"{side}_{translate_stroke_type(stroke)}"
