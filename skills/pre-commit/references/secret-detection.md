# Secret and Artifact Detection at Commit Time

Scope: keeping credentials, private keys, and large model artifacts out of git history
using pre-commit hooks.

## Built-in Key Detection

```yaml
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.6.0
  hooks:
    - id: detect-private-key
    - id: check-added-large-files
      args: ['--maxkb=5000']
```

- **detect-private-key** prevents committing private keys (SSH, RSA, etc.). It matches the
  standard PEM header block, so it catches an accidental `id_rsa` or service-account key.
- **check-added-large-files** prevents accidentally committing large files (models,
  datasets). The `--maxkb=5000` argument sets a 5 MB limit — checkpoints and datasets
  belong in object storage or an artifact registry.

## Custom Hardcoded-Secret Hook

```yaml
- repo: local
  hooks:
    - id: no-secrets-in-config
      name: Check for hardcoded secrets
      entry: bash -c 'if grep -rn "api_key\s*=\s*[\"'\''][^\"'\'']*[\"'\'']" "$@" 2>/dev/null; then echo "ERROR: Possible hardcoded API key detected." && exit 1; fi'
      language: system
      types: [python]
```

This catches the common `api_key = "sk-..."` shape in Python sources. Extend the pattern
for the credential names your project actually uses (`token`, `secret`, `password`,
`WANDB_API_KEY`, `AWS_SECRET_ACCESS_KEY`).

## Blocking Model Artifacts

Model weights are the ML-specific equivalent of a leaked secret: they bloat history
permanently and cannot be removed without a rewrite.

```yaml
- repo: local
  hooks:
    - id: no-model-files
      name: Check for model files
      entry: bash -c 'for f in "$@"; do case "$f" in *.pt|*.pth|*.onnx|*.pkl|*.h5) echo "ERROR: Model file $f should not be committed. Use object storage or an artifact registry instead." && exit 1;; esac; done'
      language: system
      types: [file]
```

## Operating Notes

- A hook only sees **staged** files, so it prevents new leaks; it does not clean history.
  If a secret was already committed, rotate the credential first, then rewrite history.
- Keep credentials in environment variables or a secrets manager, and keep `.env` in
  `.gitignore` so the detection hooks never have to fire.
- Secret hooks are cheap; leave them enabled on every commit rather than moving them to
  `stages: [manual]`.
