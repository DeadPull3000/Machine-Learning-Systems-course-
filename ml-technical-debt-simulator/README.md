# ML Technical Debt Simulator

> **An interactive browser-based systems simulation for a 15-minute presentation on:**  
> *“Hidden Technical Debt in Machine Learning Systems”* — D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison (Google Inc., NeurIPS 2015).

---

## 1. Central Narrative

> **“The model keeps getting better, while the system around the model becomes progressively more fragile.”**

This simulation demonstrates how reasonable, isolated engineering decisions (adding features, patching corner cases with auxiliary models, connecting downstream consumers, closing user feedback loops, adding transformation pipelines) accumulate dependencies until the ML system becomes brittle and resistant to change.

---

## 2. Quick Start

This project is completely self-contained with **zero external runtime dependencies** or build steps.

### Option A: Local HTTP Server (Recommended)
From this directory:

```bash
# Using Python
python -m http.server 8000

# OR using Node.js npx
npx serve .
```
Then navigate to `http://localhost:8000` in your web browser.

### Option B: Open directly
Open `index.html` directly in modern browsers (Chrome, Edge, Firefox, Safari).

---

## 3. Presenter Workflow (Step-by-Step Guide)

| Step | Action | Paper Concept | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **0** | **Clean Baseline** | System Boundaries | `DATA → MODEL → PREDICTION`. Acc: 92.4%, Debt: LOW. Clean, simple, transparent. |
| **1** | Click **+ Add Feature** (4 times) | **CACE (Changing Anything Changes Everything)** | Accuracy rises to 94.5%, but features are co-dependent. |
| **2** | Click **+ Correction Model** | **Correction Cascades** | Instead of retraining Model A, a patch model $A'$ is added. Local fix, global debt. |
| **3** | Click **+ Undeclared Consumer** | **Undeclared Consumers** | An unmonitored consumer scrapes predictions. Model changes now risk breaking unknown systems. |
| **4** | Click **+ Feedback Loop** | **Direct/Hidden Feedback Loops** | Prediction influences users; user actions become future training data. |
| **5** | Click **+ Pipeline Stage** (3 times) | **Pipeline Jungles & Glue Code** | ETL 1, ETL 2, and Feature Processing enclose the model. 95% of the codebase is now plumbing. |
| **6** | Click **Change Feature A** (`C`) | **Dependency Closure** | Graph BFS highlights every affected node across boundaries. |
| **7** | Click **Run Debt Analysis** (`D`) | **Static Debt Analysis (Proposed Extension)** | Audits anti-patterns matching sections 2–5 of the paper. |

---

## 4. Keyboard Shortcuts

- <kbd>Space</kbd> or <kbd>→</kbd> : Advance to the next presentation step
- <kbd>P</kbd> : Toggle **Presentation Mode** (fullscreen 16:9 layout, large high-contrast typography for projectors)
- <kbd>C</kbd> : Trigger **Change Feature A** impact cascade
- <kbd>D</kbd> : Open **Debt Analysis Audit Report**
- <kbd>R</kbd> : **Reset** back to Stage 0
- <kbd>Esc</kbd> : Close open modal dialogs / Exit Presentation Mode

---

## 5. Architectural Design & Philosophy

- **Internal Directed Graph Engine** (`js/graph.js`): All nodes and edges are stored in an in-memory graph data structure. Downstream impact calculations perform real BFS traversal with cycle detection.
- **Academic Aesthetic** (`styles.css`): Adheres strictly to an engineering research visualization aesthetic (warm off-white `#F4F1EA`, paper cream `#FAF9F5`, charcoal `#252525`, slate `#59636E`, ochre `#B07A32`, brick red `#A94A3A`). No glowing neon or AI SaaS cliches.
- **Accessible & Responsive**: Fully keyboard navigable, visible focus rings, responsive viewport scaling, and `@media (prefers-reduced-motion)` support.

---

## 6. Important Conceptual Disclaimers

1. **Simulated Debt & Accuracy**: As explicitly noted by Sculley et al., technical debt has no strict quantitative metric. The simulated debt counter, meter, and accuracy numbers are fictional illustrative presentation devices.
2. **Static Debt Analysis**: The debt audit report is an illustrative proposed extension inspired by the paper's categorized anti-patterns, not a feature created in the original 2015 paper.
