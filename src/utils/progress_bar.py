# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the progress_bar script, which will print a progress bar to the console.
"""

import sys

BAR_LEN = 60


def progress_bar(count, total, suffix=""):
    """
    Will print a progess bar and update it at each call.
    :param count: progress in the total
    :param total: the maximum the count can go to
    :param suffix: a message to add at the end of the progress bar
    """
    filled_len = int(round(BAR_LEN * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar_progress = "=" * filled_len + "-" * (BAR_LEN - filled_len)

    sys.stdout.write(f"[{bar_progress}] {percents}%    {suffix}\r")
    sys.stdout.flush()
