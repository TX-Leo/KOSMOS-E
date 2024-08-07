import os
import gzip
from sre_parse import SPECIAL_CHARS
import numpy as np
from random import Random
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union
import collections

EOD_SYMBOL = "</doc>"
BOI_SYMBOL = "<image>"
EOI_SYMBOL = "</image>"
EOC_SYMBOL = "</chunk>"
EOL_SYMBOL = "</line>"

GRD_SYMBOL="<grounding>"
BOP_SYMBOL="<phrase>"
EOP_SYMBOL="</phrase>"
BOO_SYMBOL="<object>"
EOO_SYMBOL="</object>"
DOM_SYMBOL="</delimiter_of_multi_objects/>"

SPECIAL_SYMBOLS = [EOD_SYMBOL, BOI_SYMBOL, EOI_SYMBOL, EOC_SYMBOL, EOL_SYMBOL]

def add_location_symbols(quantized_size, locate_special_token=0):
    custom_sp_symbols = []
    for symbol in SPECIAL_SYMBOLS:
        custom_sp_symbols.append(symbol)
    for symbol in [BOP_SYMBOL, EOP_SYMBOL, BOO_SYMBOL, EOO_SYMBOL, DOM_SYMBOL]:
        custom_sp_symbols.append(symbol)
    if locate_special_token > 0:
        custom_sp_symbols.append(GRD_SYMBOL)
    for i in range(quantized_size ** 2):
        token_name = f"<patch_index_{str(i).zfill(4)}>"
        custom_sp_symbols.append(token_name)
    return custom_sp_symbols

