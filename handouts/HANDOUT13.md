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

We rebuilt `gui.py` using `customtkinter`, a modern UI library that provides a polished, light-mode, ChatGPT-style chat interface.

### Features
1. **Curated Selection**: The assistant begins the conversation by dynamically pulling exactly 5 random images from our vast `data/` folder.
2. **Progressive Interactive Inference**: It acts as a chat. When you select an image, it appears as a "User Message". The model then progressively analyzes the image ("Analyzing features...", "Applying Softmax...") before giving the final result.
3. **Advanced Visuals**: The UI includes chat bubbles, avatars (🤖 and 👤), and dynamic coloring (e.g., highlighting warnings in red if confidence is below 80%).
4. **Developer Transparency**: Embedded right in the chat response is a "Developer Logs" section showing the source file, actual ground truth label, raw logits, and exact softmax probabilities.

You can launch this interface right now by running:
```bash
python src/gui.py
```
