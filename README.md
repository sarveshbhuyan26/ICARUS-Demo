# PrepPilot — Adaptive AI Study Companion

**Hackathon early-evaluation prototype · PS 2**

PrepPilot selects quiz questions using chapter-level ability estimates and response
signals: correctness, confidence, timing and hints. It explains question selection
and optionally uses Gemini for generation, explanations and doubt answering.

## Files

- `quizbot.py`: the combined, runnable app, with backend, frontend and question data embedded.
- `README.md`: setup, features, evidence and limitations.
- `DESCRIPTION.txt`: ready-to-copy GitHub description and project abstract.

The combined file contains all application source. It unpacks its embedded files
into a temporary directory while running; you do not need the earlier split files.

## Run

Requires Python 3.12+ and a modern browser. Open a terminal in this folder:

```bash
python -m pip install fastapi==0.162.0 uvicorn==0.54.0
python quizbot.py
```

Open **http://localhost:8000**. If port 8000 is occupied, stop the earlier server
or use `python quizbot.py --port 8001` and open http://localhost:8001.

## Learning flow

Continue as guest or sign in → choose class/exam, subjects, starting difficulty and
focus time → choose a subject and chapter → Start quiz.

Setup offers 25-, 40- and 60-minute blocks. The timer starts when the chapter quiz
begins. A bank-backed demo without an AI key: choose **JEE Advanced**, **Comfortable**,
then **Maths → Quadratic Equations**.

## Features

- Class 1–12, JEE Advanced and NEET syllabus maps.
- Subject cards, searchable chapters and chapter-focused quizzes.
- IRT-inspired Elo ability model, subject priors and adaptive difficulty targeting.
- “Why this question?” explanations and a learner dashboard.
- Confidence-aware feedback, heuristic error diagnosis and wellbeing signals.
- Hints, timed questions, focus blocks and server-enforced breaks.
- Optional Gemini generation with a second solve, explanations and doubt support.
- Answer-specific reports that rebuild progress without deleting shared questions.
- Account password hashing and same-server quiz resume across browser reloads.

## Optional Gemini setup

Set variables before launching:

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your-key"
# Optional override, if the default is unavailable
$env:GEMINI_MODEL="your-accessible-model-id"
python quizbot.py
```

```bash
# macOS / Linux
export GEMINI_API_KEY="your-key"
python quizbot.py
```

The current code defaults to `gemini-3.5-flash` and falls back to
`gemini-flash-lite-latest`. Provider/model availability varies. The app does not
load .env automatically. Never share your key. Uncovered chapters need AI; supported
bank chapters work without it. Fonts, KaTeX and Chart.js use online CDNs.

## How adaptation works

`P(correct) = sigmoid(ability − question difficulty)`. A chapter ranking balances
weakness, exploration, misconception flags and recent repetition. Item selection
aims near 60% estimated success when suitable questions exist. Responses update
the learner model; explicit chapter focus restricts selection to that chapter.

## Evidence and limits

The embedded bank contains 154 questions across 46 chapter labels: 54 numeric
answers recomputed by the checker, 26 stored conceptual reviews and 74 imported
source-key items with structural checks. Educator review is still required.
Agreement between two AI solves does not guarantee correctness.

The underlying split-source build passed its engine and bank checks and 8 API
regression tests. Browser checks covered setup, chapter selection, timer start,
resume and mobile layout. These test files are not bundled in this three-file package.

A synthetic simulation reported 44%, 26% and 21% fewer questions for three
artificial profiles. Its learning assumptions reward difficulty near ability;
these are not measured learning gains in real students.

Progress, tokens, reports and generated questions live in server memory. Restarting
resets them. Accounts persist in `quizbot_users.json`, generated beside the launcher;
**do not upload that file to GitHub**. Run one process. APIs are not protected by
account authorization, and durable account-linked progress is not implemented.
This is a local evaluation prototype, not a public student service.

Predicted scores, diagnosis and fatigue signals are illustrative heuristics.
JEE marking is simplified to +3/−1. Curriculum alignment, accessibility, security,
permanent storage and real learner validation remain future work. AI requests may
send question/answer context or doubt text to Google; avoid personal student data.

## Credits and licensing

No project-wide license has been selected. The code was developed with AI coding
assistance and needs human review.

Imported data comes from [JEEBench](https://github.com/dair-iitd/jeebench), associated
with Arora, Singh and Mausam (2023),
[Have LLMs Advanced Enough? A Challenging Problem Solving Benchmark For Large Language Models](https://aclanthology.org/2023.emnlp-main.468/), EMNLP 2023.
The upstream MIT notice is preserved below for the imported material; it does not
license PrepPilot's original code. Dependencies retain their upstream licenses.
The project is not affiliated with JEEBench, exam authorities or Physics Wallah.

### JEEBench upstream license

```text
MIT License

Copyright (c) 2023 Data Analytics and Intelligence Research (DAIR) Group, IIT Delhi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
