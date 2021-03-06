# Copyright (c) 2022 Alix Routhier-Lalonde. Licence included in root of package.

"""
Contains the get_base_path function, which will return the relative path to the base of the project.
"""

import os


def get_base_path():
    """
    this function will return the relative path to return to base of the project
    :return: the relative path
    """
    path = os.path.abspath(".")
    splitted_path = path.split("/")
    relative_path = ""
    for _, value in enumerate(splitted_path[::-1]):
        if value == "stock-prediction":
            break
        relative_path += "../"
    if relative_path == "":
        return "./"
    return relative_path


if __name__ == "__main__":
    print(get_base_path())
