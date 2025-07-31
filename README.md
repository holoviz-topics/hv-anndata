# hv-anndata

Holoviz Anndata Interface

## Hacking

In order to run the notebooks, install the `hv-anndata` kernel:

```bash
hatch run docs:install-kernel
```

- Tests: `hatch test`
- Docs: `hatch docs:build`
- Lints: `pre-commit run --all-files` (use `pre-commit install` and `nbstripout --install` to install Git hooks)
