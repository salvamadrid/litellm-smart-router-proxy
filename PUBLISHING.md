## Publish to GitHub

This repo is ready to be published as a standalone repository.

### 1) Initialize git

```bash
git init
git add .
git commit -m "chore: initial public release"
```

### 2) Create GitHub repo + push

```bash
git remote add origin git@github.com:<your-user>/<your-repo>.git
git push -u origin main
```

### Notes

- Do **not** commit `.env` or `artifacts/` (already in `.gitignore`).

