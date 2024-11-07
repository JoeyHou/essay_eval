# Generic logging setup
# Created by Alejandro Ciuba, alc307@pitt.edu
from pathlib import Path

import logging

StrPath = str | Path

def make_loggers(*args: StrPath, levels: list[int] | int) \
    -> tuple[logging.Logger]:

    def make_logger(path, level, i) -> logging.Logger:

        logger = logging.getLogger(f"log-{i}")
        logger.setLevel(level)

        handler = logging.FileHandler(path)
        handler.setFormatter(fmt)

        logger.addHandler(handler)

        return logger

    # Set up logger
    # version_tracking = {"version": "VERSION %s" % VERSION}

    fmt = logging.Formatter(fmt="%(asctime)s : %(message)s",
                            datefmt='%Y-%m-%d %H:%M:%S')

    bundle = zip(args, levels) if isinstance(levels, list) else zip(args, [levels] * len(args))

    return tuple([make_logger(path, level, i) for i, (path, level) in enumerate(bundle)])
