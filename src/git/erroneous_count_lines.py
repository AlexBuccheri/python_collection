""" Count the net line contribution per git commit

Note, this script does not work as expected.
 - Use of `git diff --numstat {current_commit} {prior_commit}`
   to track the net change in line count of a given file type, between adjacent
   commits, fails because total lines does not need to be conserved (established empirically).
"""

import subprocess
import datetime
import matplotlib.pyplot as plt


def get_commit_range(start_date, end_date):
    """
    Get a list of commits between start_date and end_date.
    """
    command = f'git rev-list --reverse --before="{end_date}T00:00:00Z" --after="{start_date}T00:00:00Z" main'
    commits = subprocess.check_output(command, shell=True).decode('utf-8').strip().split('\n')
    return commits


def git_commit_date(commit):
    """Resolved at the level of a day i.e. 2024-07-08
    """
    cmd = f"git show -s --format=%cd --date=short {commit}"
    date_string = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
    return date_string


def get_lines_added_removed(current_commit, prior_commit, ext):
    # -- ':!external_libs/** 
    command = f"git diff --numstat {current_commit} {prior_commit} | grep -v 'external_libs' | grep -i '\.{ext}$' | awk '{{total_added += $1; total_deleted += $2}} END {{print total_added, total_deleted}}'"
    string = subprocess.check_output(command, shell=True).decode('utf-8').strip()
    added_removed = [int(s) for s in string.split()]
    if added_removed == []:
        added_removed = [0, 0] 
    return added_removed


# Define the start date and end date
effective_first_commit = 'aa23c8bfa38b64e48feeef199ea173f80c25f6d7'
start_year, start_month = 2002, 1
start_date = f"{start_year}-{start_month:02d}-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")

# Get the list of commits between start_date and end_date
commits = get_commit_range(start_date, end_date)
assert commits[0] == 'aa23c8bfa38b64e48feeef199ea173f80c25f6d7'

# Found from manual inspection of the effective_first_commit
# because git ls-tree does not work (assume due to SVN origins)
# find . -name '*.F90' -exec wc -l {} + | awk '{s+=$1} END {print s}'
# year.month.day
line_length = {'F90': [('2002-01-25', 16510)],
               'c':  [('2002-01-25', 0)],
               'h':  [('2002-01-25', 14)],
               'cl': [('2002-01-25', 0)],
               'cc': [('2002-01-25', 0)]
               }

extensions = list(line_length.keys())

for i in range(1, len(commits)):
    current_commit = commits[i]
    prior_commit = commits[i-1]

    current_commit_date = git_commit_date(current_commit)
    added_removed = get_lines_added_removed(current_commit, prior_commit, "F90")
    net_change = added_removed[0] - added_removed[1]

    prior_length = line_length['F90'][-1][1]
    current_length = prior_length + net_change
    line_length['F90'].append((current_commit_date, current_length))

    print(current_commit_date, added_removed[0], added_removed[1], current_length)
