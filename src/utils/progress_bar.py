# taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters

import sys

def progress_bar(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s    %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()