from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TextIO


def _print_help(file: TextIO) -> None:
    print(
        """usage: mt <command> [args]

Commands:
  serve       Start the maxMLX API server
  shell       Chat with a maxMLX server in the terminal
  ctl         Query and manage a running maxMLX server
  daemon      Run the maxMLX supervisor (persistent engine service)
  launch      Configure and launch an agent against a maxMLX server

Use "mt <command> --help" for command-specific options.
Use "mt --version" to print the maxMLX version.""",
        file=file,
    )


def _run_serve(argv: list[str]) -> int:
    from maxmlx.server import launch_server

    launch_server(argv=argv, prog="mt serve")
    return 0


def _run_shell(argv: list[str]) -> int:
    from maxmlx.shell import main

    return main(argv, prog="mt shell")


def _run_launch(argv: list[str]) -> int:
    from maxmlx.launch import main

    return main(argv, prog="mt launch")


def _run_ctl(argv: list[str]) -> int:
    from maxmlx.control_cli import main

    return main(argv, prog="mt ctl")


def _run_daemon(argv: list[str]) -> int:
    from maxmlx.daemon import main  # torch-free supervisor

    return main(argv, prog="mt daemon")


COMMANDS = {
    "serve": "_run_serve",
    "shell": "_run_shell",
    "ctl": "_run_ctl",
    "daemon": "_run_daemon",
    "launch": "_run_launch",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_help(sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        _print_help(sys.stdout)
        return 0
    if args[0] in {"-V", "--version"}:
        from maxmlx.version import __version__

        print(f"maxmlx version {__version__}")
        return 0

    from maxmlx.paths import migrate_legacy_dirs

    migrate_legacy_dirs()
    command = args[0]
    if command.startswith("-"):
        # Flags where a command was expected mean serve. `python -m maxmlx`
        # used to BE the server and took its flags directly, so that spelling
        # is in people's shells and scripts; routing it here keeps it working
        # while `python -m maxmlx serve ...` -- the form every doc shows --
        # starts working too.
        return _run_serve(args)
    runner_name = COMMANDS.get(command)
    if runner_name is None:
        print(f"unknown mt command: {command}", file=sys.stderr)
        _print_help(sys.stderr)
        return 2

    runner = globals()[runner_name]
    return runner(args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
