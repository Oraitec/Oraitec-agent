# Contributing to Oraitec Agent Platform

Thank you for contributing to **Oraitec Agent** 
This document provides contribution guidelines for all internal developers.

---

## Branching Strategy

We follow a **feature → dev → main** workflow:

* **`main`**

  * Always stable and deployable
  * Protected: requires Pull Request, ≥1 reviewer, CI must pass

* **`dev`**

  * Integration branch for development
  * All new features are merged here

* **`feature/*`**

  * Feature branches, e.g. `feature/hrv-core`, `feature/app-ui`
  * Created from `dev`, merged back via PR

## 🛠 Development Workflow

1. **Create a branch**

   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/<name>
   ```

2. **Make changes & test locally**

   * Follow coding style guidelines (see below).
   * Run unit tests before committing.

3. **Commit changes**

   ```bash
   git add .
   git commit -m "feat: add HRV analysis module"
   ```

4. **Push branch**

   ```bash
   git push origin feature/<name>
   ```

5. **Open a Pull Request**

   * Target branch: `dev`
   * CI must pass
   * At least one reviewer approval required

6. **Release process**

   * Maintainers merge `dev` → `main`
   * Tag release (`v0.x.x`)

---

## Commit Message Guidelines

We use **Conventional Commits**:

```
<type>(<scope>): <short description>
```

**Types:**

* `feat`: new feature
* `fix`: bug fix
* `docs`: documentation only
* `style`: formatting, no code change
* `refactor`: code restructure
* `test`: add or modify tests
* `chore`: maintenance tasks

**Examples:**

* `feat(agent-core): add HRV analysis module`
* `fix(app-mobile): resolve crash on startup`
* `docs: update README with setup instructions`

---

## Code Style & Testing

* **Python**: PEP8 + Black auto-format
* **JavaScript/TypeScript**: ESLint + Prettier
* **Tests**: All features must include unit tests in `/tests/`
* Run `pytest` (Python) or `npm test` (JS) before PR submission

---

## Directory Conventions

* Each directory must contain actual files or a `.gitkeep` placeholder.
* Documentation for each module goes into a `docs/` folder.

---

## Review Process

* PRs require at least **1 reviewer approval**.
* CI/CD checks must pass before merge.
* Maintainers decide on **squash merge** vs **merge commit**.

---

## Release Process

* When `dev` is stable, Maintainers open PR → `main`.
* Versioning: **Semantic Versioning** (`v0.1.0`, `v0.2.0`).
* Each release is tagged and published on GitHub.

---

## Acknowledgements

Thanks to all Oraitec team members for building this platform together.

---

