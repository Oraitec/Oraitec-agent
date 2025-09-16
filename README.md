# Oraitec Agent 

**Oraitec Agent** is the internal development repository for building the **Vertical AI Agent Core** and **Application Layer**.

---

## 📌 Project Overview

The Oraitec Agent is designed as a **vertical AI agent framework** that integrates:

* **Agent Core**: multimodal signal processing (PPG, EDA, IMU, camera) and inference engine.
* **Application Layer**: mobile, web, and integration SDKs.
* **Data Services**: preprocessing, loaders, and synthetic data generation.

The goal is to support **next-generation AI glasses** and digital health applications with real-time multimodal insights.

---

## 📂 Project Structure
```
Oraitec-agent/
│
├── agent-core/                  # Core engine of the vertical agent
│   ├── models/                  # Model configs & checkpoint references
│   ├── training/                # Training pipeline & preprocessing
│   ├── inference/               # Inference APIs & wrappers
│   ├── evaluation/              # Metrics & benchmarks
│   └── docs/                    # Core architecture documentation
│
├── app/                         # Application layer
│   ├── mobile/                  # iOS/Android (Flutter / React Native)
│   ├── web/                     # Web client
│   ├── integration/             # SDKs & API bindings
│   └── docs/                    # App design & API docs
│
├── data/                        # Data utilities (no raw data committed)
│   ├── loaders/                 
│   ├── synthetic/               # Synthetic data generators
│   └── docs/
│
├── examples/                    # Example scripts & notebooks
│   ├── quickstart/              
│   ├── notebooks/               
│
├── tests/                       # Unit & integration tests
│
├── scripts/                     # Utility scripts (deployment, CI/CD)
│
├── .github/                     # GitHub settings
│   ├── workflows/               # CI/CD workflows
│   └── ISSUE_TEMPLATE/          # Templates for issues/PRs
│
├── LICENSE
├── README.md
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```
---

## 🌱 Branching Strategy

We adopt a **feature → dev → main** workflow:

* **`main`**

  * Always stable and deployable.
  * Protected: requires PR, ≥1 review, CI must pass.

* **`dev`**

  * Integration branch for ongoing development.
  * All features are merged here before release.

* **`feature/*`**

  * Per-feature branches (e.g., `feature/hrv-core`, `feature/app-ui`).
  * Always branched from `dev`, merged back via PR.
<img width="1979" height="1180" alt="output" src="https://github.com/user-attachments/assets/f82c26be-33d4-42ad-94f4-2f5b04c6d07c" />

---

## 🛠️ Development Workflow

1. **Clone the repo**

   ```bash
   git clone git@github.com:Oraitec/Oraitec-agent.git
   cd Oraitec-agent
   ```

2. **Create a feature branch**

   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/<name>
   ```

3. **Commit & push changes**

   ```bash
   git add .
   git commit -m "feat: add HRV analysis module"
   git push origin feature/<name>
   ```

4. **Open a Pull Request**

   * Target branch: `dev`
   * CI tests must pass
   * At least one reviewer approval required

5. **Release process**

   * `dev` → `main` via PR
   * Tag a release (`v0.x.x`)

---

## 🤝 Contribution

* Internal contributors should follow [`CONTRIBUTING.md`](./CONTRIBUTING.md).
* Code style, commit message conventions, and PR review process are enforced.
* All directories must contain either actual files or a `.gitkeep` placeholder.

---

## 📅 Roadmap

* **MVP**: Build Agent Core with HRV and emotion analysis modules.
* **Phase 2**: Extend multimodal integration (EDA, IMU, camera).
* **Phase 3**: Deploy production-ready applications with cloud sync and personalization.

---

## 📜 License

This project is currently unlicensed.

---
