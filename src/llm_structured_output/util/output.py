# pylint: disable=missing-function-docstring
"""
Terminal colored output
"""

def info(*args, **kwargs):
    print("\033[34mℹ ", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def warning(*args, **kwargs):
    print("\033[43;37m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def debug(*args, **kwargs):
    print("\033[33m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def debugbold(*args, **kwargs):
    print("\033[1;33m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def bold(*args, **kwargs):
    print("\033[1;30m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def bolddim(*args, **kwargs):
    print("\033[1;2;30m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def boldalt(*args, **kwargs):
    print("\033[1;36m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def underline(*args, **kwargs):
    print("\033[4m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def inverse(*args, **kwargs):
    print("\033[7m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")
