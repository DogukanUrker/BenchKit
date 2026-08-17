# secret-in-history

`setup.sh SEED WORKSPACE` creates a five-commit repository with fixed Git
identity and timestamps. The generated AWS-style credential appears in the
second commit, while the third commit replaces the same line. Removing the
credential commit therefore causes a real replay conflict. Later commits change
the retry policy and documentation.

`verify.sh SEED WORKSPACE` emits tab-separated, deterministic state checks. It
uses `rev-list`, `cat-file`, `merge-base`, exact commit subjects, source checks,
and Python's offline unittest runner. BenchKit combines those checks with Pi's
native tool trace for the history-search and rewrite-path checkpoints.

To validate the task by hand, build or run the Git Surgery image, execute the
setup script, enter `/workspace/secret-in-history`, and repair the repository
with normal Git commands. Then run:

```bash
bash /opt/git-surgery/secret-in-history/verify.sh \
  424242 /workspace/secret-in-history
```

A correct state reports `1` for repository continuity, preserved history,
secret absence, and tests, and `0` for destructive reinitialization. The two
trace-derived checkpoints are added by BenchKit during an actual Pi run.
