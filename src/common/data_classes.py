from dataclasses import dataclass

leads = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]

@dataclass
class Range:
    start: int
    peak: int
    end: int


@dataclass
class AnnotationSymbols:
    p: list[Range]
    N: list[Range]
    t: list[Range]