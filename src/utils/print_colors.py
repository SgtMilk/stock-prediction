# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the Colors class, a collection of colors for console printing.
"""


class Colors:
    """
    This class incorporates color codes for printing to console
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
