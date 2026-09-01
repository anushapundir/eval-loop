## Summary

<!-- One or two sentences: what does this change and why? -->

## What changed

<!-- Bullet the notable changes. Mention any new module and its responsibility. -->

-

## How this was verified

<!-- Paste the actual output. "Tests pass" without evidence is not enough. -->

```
$ ruff check .

$ pytest -q

```

## Checklist

- [ ] `ruff check .` is clean
- [ ] `pytest` passes, and the suite still runs fully offline (no live model calls)
- [ ] No secrets, API keys, or local absolute paths added
- [ ] Type hints and docstrings on new public functions
- [ ] Docs updated if behaviour changed
- [ ] Cost rules respected (agent stays on the free local runtime; judge stays sampled and out of the loop)
