from common.data_classes import AnnotationSymbols, Range
from wfdb import Annotation
from wfdb.processing import resample_ann
from typing import cast

def parse_annotations(annotation: Annotation, resample_fs: int = 100) -> AnnotationSymbols:
        if not annotation.symbol:
            raise Exception("the file need to have symbols loaded")

        current_start = None
        current_peak = None
        current_symbol = None
        intervals = {"p": [], "N": [], "t": []}

        for symbol, index in zip(annotation.symbol, resample_ann(annotation.sample, annotation.fs, fs_target=resample_fs)):
            match symbol:
                case "(":
                    current_start = index
                case "p" | "N" | "t":
                    current_symbol = cast(str, symbol)
                    current_peak = index
                case ")":
                    if not (current_peak and isinstance(current_symbol, str)):
                        raise Exception("need to have peak specified before end of the interval")
                    if not current_start:
                        if not len(intervals[current_symbol]) == 0:
                            raise Exception("need to have start of the interval before parse end")
                        current_start = 0
                    intervals[current_symbol].append(Range(current_start, current_peak, index))

        return AnnotationSymbols(**intervals)