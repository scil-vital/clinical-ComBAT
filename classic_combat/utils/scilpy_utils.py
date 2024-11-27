# -*- coding: utf-8 -*-
"""
Scripts duplicating functions originally available in Scilpy.

"""

import os

def assert_outputs_exist(parser, args, required, optional=None, check_dir_exists=True):
    """
    Assert that all outputs don't exist or that if they exist, -f was used.
    If not, print parser's usage and exit.

    Args:
        parser (argparse.ArgumentParser): Parser object
        args (list): Argument list
        required (str or list of paths to files): Required paths to be checked
        optional (str or list of paths to files): Optional paths to be checked
        check_dir_exists (bool): Test if output directory exists

    """

    def check(path):
        if os.path.isfile(path) and not args.overwrite:
            parser.error(
                "Output file {} exists. Use -f to force " "overwriting".format(path)
            )

        if check_dir_exists:
            path_dir = os.path.dirname(path)
            if path_dir and not os.path.isdir(path_dir):
                parser.error(
                    "Directory {}/ \n for a given output file "
                    "does not exists.".format(path_dir)
                )

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        if optional_file:
            check(optional_file)


def add_overwrite_arg(parser):
    """
    Add the overwrite argument to the parser.

    Args:
        parser (argparse.ArgumentParser): Parser object
    """
    parser.add_argument(
        "-f",
        dest="overwrite",
        action="store_true",
        help="Force overwriting of the output files.",
    )


def add_verbose_arg(parser):
    """
    Add the verbose argument to the parser.

    Args:
        parser (argparse.ArgumentParser): Parser object
    """
    parser.add_argument(
        "-v",
        default="WARNING",
        const="INFO",
        nargs="?",
        choices=["DEBUG", "INFO", "WARNING"],
        dest="verbose",
        help="Produces verbose output depending on "
        "the provided level. \nDefault level is warning, "
        "default when using -v is info.",
    )
