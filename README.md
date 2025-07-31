# hv-anndata

Holoviz Anndata Interface

## Hacking

In order to run the notebooks, install the `hv-anndata` kernel:

```bash
hatch run docs:install-kernel
hatch env find docs  # if you need the path for e.g. VS Code
```

- Tests: `hatch test`
- Docs: `hatch docs:build`
- Lints: `pre-commit run --all-files` (use `pre-commit install` to install Git hooks)
