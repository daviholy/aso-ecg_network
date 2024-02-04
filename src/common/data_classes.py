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

    def get_min_start_idx(self) -> int:
        res = [s[0].start for s in [self.p, self.N, self.t] if len(s) > 0]        
        return min(res)
    
    def get_max_end_idx(self) -> int:
        res = [s[-1].end for s in [self.p, self.N, self.t] if len(s) > 0]        
        return max(res)