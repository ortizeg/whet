---
name: github-repo-setup
description: >
  Use this skill when initializing or configuring a GitHub repository for a CV/ML project
  — creating the repo, branch protection rules, PR and issue templates, CODEOWNERS, merge
  strategy, and gh CLI automation. Reach for it any time you'd otherwise click through
  GitHub settings by hand or ask "how should I set up this repo", even if the user just
  says "create the repo" or "protect main". For the CI/CD workflows that run on those
  branches see github-actions.
---

# GitHub Repository Setup Skill

Best practices for initializing and configuring GitHub repositories for CV/ML projects.
This page holds repo creation and the branch-protection core you need almost every time;
the deep dives below cover templates, CODEOWNERS, merge strategy, and full `gh` CLI
automation. For the CI/CD workflows that satisfy the required status checks, see the
`github-actions` skill.

## Repository Initialization

Create repositories with consistent structure from the start.

```bash
# Create a new public repo with README, LICENSE, and .gitignore
gh repo create my-cv-project \
  --public \
  --clone \
  --license MIT \
  --gitignore Python \
  --description "Computer vision training pipeline"

cd my-cv-project

# Initialize with standard structure
mkdir -p src/my_cv_project tests docs .github/workflows .github/ISSUE_TEMPLATE
touch src/my_cv_project/__init__.py
touch src/my_cv_project/py.typed
touch tests/__init__.py
```

### Standard Files Checklist

Every repository must have:

- `README.md` — Project overview, quick start, badges
- `LICENSE` — MIT for open-source CV/ML projects
- `.gitignore` — Python template + ML-specific exclusions (model weights, datasets, wandb/)
- `pyproject.toml` — Project metadata and tool configuration
- `pixi.toml` — Environment and dependency management

## The branch-protection core

Protect `main` from day one so every change lands through a reviewed, CI-verified PR:

```bash
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["test", "lint"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
```

The `contexts` entries must match CI **job names** exactly — a job named `test` in
`.github/workflows/test.yml` becomes the status check `test`. Pair this with squash-merge
defaults and branch deletion on merge:

```bash
gh api repos/{owner}/{repo} \
  --method PATCH \
  --field allow_squash_merge=true \
  --field allow_merge_commit=false \
  --field allow_rebase_merge=false \
  --field delete_branch_on_merge=true
```

## Conventions

1. **Protect main from day one** — Enable branch protection before the first collaborator joins.
2. **Use squash merge** — Keeps main history clean with one commit per PR.
3. **Delete branches on merge** — Prevents stale branch accumulation.
4. **Require status checks** — Never merge without CI passing.
5. **Use CODEOWNERS** — Automate review assignments and enforce ownership.
6. **Template everything** — PR templates, issue templates, and CONTRIBUTING.md reduce friction.
7. **Disable unused features** — Wiki, Projects, Discussions add noise if unused.
8. **Enable auto-merge** — Let PRs merge automatically when all checks pass.
9. **Script your setup** — Use `gh` CLI scripts so configuration is reproducible across repos.
10. **Review protection quarterly** — As the team grows, adjust required reviewers and status checks.

## Anti-Patterns

- **No branch protection** — Pushing directly to main risks breaking the project.
- **Too many required reviewers** — More than 2 for small teams creates bottlenecks.
- **Merge commits for feature branches** — Pollutes history with merge noise.
- **Manual repository setup** — Click-through configuration is not reproducible.
- **Skipping PR templates** — Leads to empty PR descriptions and lost context.
- **Overly broad CODEOWNERS** — `* @everyone` means no one owns anything.
- **Ignoring stale reviews** — Approving code that has changed since review is dangerous.
- **Force pushing to main** — Rewrites shared history and breaks collaborator checkouts.

## Deep dives

- `references/branch-protection.md` — read when tuning protection rules field by field, choosing a merge strategy, or figuring out which status-check names are available.
- `references/pr-and-issue-templates.md` — read when adding `.github/PULL_REQUEST_TEMPLATE.md` or YAML issue forms for bug reports and feature requests.
- `references/codeowners.md` — read when writing `.github/CODEOWNERS` or debugging which pattern matched a reviewer assignment.
- `references/gh-cli-automation.md` — read when scripting the whole setup: repository settings, GitHub Pages, a reusable `setup-repo.sh`, or a typed Pydantic config model.
