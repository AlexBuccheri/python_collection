#!/usr/bin/python3

# Copyright (c) 2024 A Buccheri

"""Count total lines of files with specified extensions, as a function of git commit history.

This script iterates through a range of Git commits on a specified branch, checking out each commit
in a temporary workspace using git workdir. For each commit, it counts the total lines of code for files with
specified extensions within the repository. Git submodules are not automatically checked out, and therefore
do not need to be explicitly omitted from the search directories.

This script must be run in a git repository.

This is written specifically for the Octopus code.

Development Notes:
 - Use of `git diff --numstat {current_commit} {prior_commit}`
   to track the net change in line count of a given file type, between adjacent
   commits, fails because total lines does not need to be conserved (established empirically).
- Use of `git s-tree -r` is not appropriate as it only returns files
  touched for a given commit, not all files present in the repo for
  that commit.
- This could be simplified by removing multiprocessing
- This could be simplified by returning git hashes at fixed intervals when querying the commit history
  rather than striding the full list of commits.
"""
from __future__ import annotations

import datetime
from multiprocessing import Pool
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Optional, List, Tuple, Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


def total_lines(ext: str, directory: Optional[str] = None) -> int:
    """Find the total number of lines for all files with a specified extension.

    :param ext: File extension, for example 'f90'. Case-insensitive.
    :param directory: Optional directory in which to run the command
    :return Total line count
    """
    # Specific to Octopus
    blacklist_dirs = [".git", "external_libs"]
    exclude_str = " ".join(f"-path './{ignore}' -prune -o" for ignore in blacklist_dirs)
    command = f"find . {exclude_str} -iname '*.{ext}' -print0 | xargs -0 cat | wc -l"

    # shell=True handles pipes in `command`
    result = subprocess.run(
        command, shell=True, cwd=directory, text=True, check=True, capture_output=True
    )
    if result.stderr:
        print(f"Command {command} has raised the error {result.stderr}")
        raise subprocess.SubprocessError(result.returncode, command)

    line_count_str = result.stdout
    return int(line_count_str)


def checkout_git_worktree(directory: str | Path):
    """Checkout a git worktree.

    NB, one must run this command within a git repository

    :param directory: Directory in which to check out the worktree to
    """
    print(f"Creating work directory {directory}")

    worktree = ["git", "worktree", "add", "-f", "--detach", directory]
    subprocess.run(worktree, check=True, capture_output=True)

    # git worktree should create a directory if it does not exist
    if not os.path.isdir(directory):
        raise NotADirectoryError(
            f"The expected git worktree directory, {directory}, is not present"
        )


def get_git_commit_date(commit: str, **kwargs) -> datetime.datetime:
    """Get git commit date

    :param commit: Git hash
    :param kwargs: run options
    :return datetime instance
    """
    result = subprocess.run(
        ["git", "show", "-s", "--format=%cd", "--date=short", commit],
        check=True,
        encoding="utf-8",
        capture_output=True,
        **kwargs,
    )
    commit_date = result.stdout.strip()
    date = datetime.datetime.strptime(commit_date, "%Y-%m-%d")
    return date


def get_git_commits(
    branch, start_date: datetime.datetime, end_date: datetime.datetime
) -> List[str]:
    """Get the list of git commit hashes between start_date and end_date

    :param branch: Branch to check out
    :param start_date: Start of commit history
    :param end_date: End of commit history
    :return commits: List of commit hashes
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    command = [
        "git",
        "rev-list",
        "--reverse",
        f'--before="{end_date_str}T00:00:00Z"',
        f'--after="{start_date_str}T00:00:00Z',
        branch,
    ]
    result = subprocess.run(command, check=True, capture_output=True, encoding="utf-8")
    commits = result.stdout.split("\n")
    return commits


def line_count_history_serial(
    commits: List[str], extensions: List[str]
) -> Tuple[list, dict]:
    """Get the line count as a function of git commits

    :param commits: List of git commit hashes
    :param extensions: List of file extensions to get line count for
    :return dates, line_length: Dates and line lengths for each commit
    """
    # Checkout git worktree
    work_dir = f"{tempfile.mkdtemp()}/oct_tmp_count_files"

    if Path(work_dir).is_dir():
        cmd = ["git", "worktree", "remove", work_dir]
        result = subprocess.run(cmd, capture_output=True, check=True, encoding="utf-8")
        print(result.stdout)

    checkout_git_worktree(work_dir)

    # Extract total line counts per commit, for specified extensions
    line_length = {ext: np.empty(shape=len(commits)) for ext in extensions}
    dates = []

    for i, commit in enumerate(commits):
        subprocess.run(
            ["git", "checkout", "--force", commit],
            cwd=work_dir,
            capture_output=True,
            check=True,
        )
        dates.append(get_git_commit_date(commit, cwd=work_dir))
        for ext in extensions:
            n_lines = total_lines(ext, directory=work_dir)
            line_length[ext][i] = n_lines

    return dates, line_length


def run_wrapper(inputs: tuple):
    """Wrapper around line_count_history, to unpack inputs *args
    for use with multiprocessing.

    :param inputs: Tuple containing all args to line_count_history_serial
    """
    return line_count_history_serial(*inputs)


def pooled_runner(run_func: Callable, inputs: list, n_processes: int) -> list:
    """Pooled function execution, with dynamic load balancing.

    NOTE. MUST be called from within `if __name__ == '__main__':`

    :param run_func: A function that can evaluate (run) a calculation. Cannot be a lambda
    :param inputs: A list of inputs.
    :param n_processes: Number of processes to run concurrently.
    :return: List of result instances, where a result must be serialisable.
    """
    assert len(inputs) == n_processes, "Expect an input per process"
    with Pool(n_processes) as p:
        results = p.map(run_func, inputs)
    return results


def line_count_history_parallel(
    commits: List[str], extensions: List[str], n_processes: int
) -> Tuple[list, dict]:
    """Counts lines of file extensions, as a function of commit history.

    :param commits: List of git commit hashes
    :param extensions: List of file extensions to get line count for
    :param n_processes: Number of parallel processes
    :return dates, line_length: Dates and line lengths for each commit
    """
    # Distribute work, without worrying about optimising load-balancing
    # of remainder
    inputs = []
    base_length = len(commits) // n_processes
    for i in range(n_processes - 1):
        i1 = i * base_length
        i2 = i1 + base_length
        inputs.append((commits[i1:i2], extensions))
    inputs.append((commits[i2:], extensions))

    results: list = pooled_runner(run_wrapper, inputs, n_processes)

    # Collate the results
    line_length = {ext: np.empty(shape=0) for ext in extensions}
    dates = []
    for my_dates, my_counts in results:
        dates += my_dates
        for ext in extensions:
            line_length[ext] = np.concatenate((line_length[ext], my_counts[ext]))

    return dates, line_length


def line_count_history(
    commits: List[str], extensions: List[str], n_processes: Optional[int] = 1
) -> Tuple[list, dict]:
    """Wrapper for line_count_history

    :param commits: List of git commit hashes
    :param extensions: List of file extensions to get line count for
    :param n_processes: Number of parallel processes. If set to one, multiprocessing is not used.
    :return dates, line_length: Dates and line lengths for each commit
    """
    assert n_processes > 0, "n_processes must be > 0"
    if n_processes > 1:
        return line_count_history_parallel(commits, extensions, n_processes)
    else:
        return line_count_history_serial(commits, extensions)


if __name__ == "__main__":
    # Number of concurrent processes
    n_processes = 4
    # Percentage of commits to perform line-counting on
    alpha = 0.1
    # read or compute
    mode = "compute"
    branch = "main"
    extensions = ["f90", "c", "h", "cl", "cc"]

    if mode == "compute":
        print(
            "Computing line account as a function of commit history\n"
            "Warning: This can take several minutes when requesting the full history"
        )
        # Git history range
        start_date = datetime.datetime(2002, 1, 25)
        commits = get_git_commits(branch, start_date, datetime.datetime.now())
        assert (
            commits[0] == "aa23c8bfa38b64e48feeef199ea173f80c25f6d7"
        ), "First relevant commit"

        # Sample the commit history to significantly speed up the calculation
        # This assumes that the date between individual commits << (end_date - start_date)
        if alpha != 1.0:
            print(f"Sampling {10 * alpha}% of the commits")
            stride = int(1 / alpha)
            commits = commits[::stride]

        dates, line_length = line_count_history(commits, extensions, n_processes)

        # Dump results: Git hash, date, f90, c, h, cl, cc
        with open(file="lines.dat", mode="w") as fid:
            for i, commit in enumerate(commits):
                count_str = " ".join(str(line_length[ext][i]) for ext in extensions)
                fid.write(
                    f'{commit} {dates[i].strftime("%Y-%m-%d")} ' + count_str + "\n"
                )
    else:
        print("Parsing line count file")
        try:
            counts = np.genfromtxt("lines.dat", delimiter=" ", usecols=[2, 3, 4, 5, 6])
            hash_date = np.genfromtxt(
                "lines.dat", dtype=None, delimiter=" ", usecols=[0, 1], encoding=None
            )
        except FileNotFoundError:
            raise FileNotFoundError("lines.dat not found")

        # Repopulate data layout
        line_length = {ext: np.empty(shape=len(hash_date)) for ext in extensions}
        for i, ext in enumerate(extensions):
            line_length[ext] = counts[:, i]

        # Commits and dates
        commits = []
        dates = []
        for commit, date in hash_date:
            commits.append(commit)
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
            dates.append(date)

        start_date = dates[0]

    # Plot result
    fig, ax = plt.subplots()
    mdates.set_epoch(start_date.strftime("%Y-%m-%d"))
    x_points = mdates.date2num(dates)

    for ext in extensions:
        ax.plot(x_points, line_length[ext], label=f"{ext}")

    # Set the desired number of ticks
    locator = mdates.AutoDateLocator(minticks=5, maxticks=24)
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.xlabel("Date")
    plt.ylabel(r"Number of Lines ($10^3$)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("octopus_lines_history.pdf", dpi=300, bbox_inches="tight")
