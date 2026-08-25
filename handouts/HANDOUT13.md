# Amharic Character Recognition — Handout 13 (Interactive GUI)

This handout covers our mid-phase goal: Creating an interactive Graphical User Interface (GUI) before we proceed to Phase 22.

---

# New Files

```text
src/
└── gui.py
```

---

# The Interface 

While our command-line script (`predict.py`) works perfectly for developers, it is not user-friendly. Real users don't type `python predict.py data/ሀ/synthetic_001.png` into a terminal.

We rebuilt `gui.py` from the ground up using `customtkinter` with a clean, minimalistic two-panel layout inspired by modern 2026 design language.

### Layout
The window is split into two panels:

**Left Panel — Image Selector**
- Displays 5 randomly chosen sample images as clean, labeled cards.
- A **Shuffle** button at the bottom instantly replaces them with a new random set — the panel itself never moves or changes shape.

**Right Panel — Results Display**
- Starts with a subtle "No image selected" empty state.
- When a sample is clicked, a smooth progress bar animates across two stages ("Analyzing" → "Running model").
- The result appears as a centered card containing:
  - A **status pill** (green "High confidence" or red "Low confidence").
  - The **predicted Amharic character** displayed prominently.
  - A separate **confidence percentage** with a matching colored progress bar.
  - A **Developer Details** section at the bottom with source file, ground-truth label, raw logits, and softmax probabilities.

### Design Choices
- **No chatbot icons or robot emojis.** The interface is purely typographic.
- **Segoe UI / Consolas** fonts for a native Windows feel.
- **Color palette**: off-white background, white surface cards, green accent (`#0EA47A`), red for warnings (`#EF4444`).

You can launch the interface by running:
```bash
.venv\Scripts\python.exe src/gui.py
```
