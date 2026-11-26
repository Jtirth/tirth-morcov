# Predictive Text and Next-Word Predictor Using Markov Chains

**NextWord Lab** is a Probability and Statistics course project. Its Flask website predicts next words using first-order and second-order Markov chains, displaying the observed counts and conditional probabilities behind every suggestion. It uses original teaching text, Python collections, HTML/CSS/JavaScript, and Chart.js; no paid APIs, API keys, or neural models are required.

## Quick start

Requirements: Python 3.10 or newer and pip.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python backend/app.py
```

Open `http://127.0.0.1:5000`. The Flask server serves the website and JSON API. Chart.js is loaded from jsDelivr; prediction and statistics are computed locally in Python. To rerun the automated checks, use a second terminal and run `python -m unittest discover -s tests -v`.

## Project files

- `backend/app.py` - Flask pages, JSON routes, validation, and temporary custom-corpus training.
- `backend/markov_model.py` - tokenization, transition counts, predictions, statistics, and held-out scoring.
- `backend/statistics.py` - reusable frequency and relative-frequency helpers.
- `frontend/templates/` - homepage, analytics dashboard, overview, and mathematics pages.
- `frontend/static/` - responsive CSS and browser interactions / Chart.js visualizations.
- `data/training.txt` - illustrative training corpus; `data/testing.txt` - separate held-out corpus.
- `data/classroom_example.txt` - three-line conditional-probability example.
- `screenshots/` - actual desktop and mobile captures of the running website.
- `documentation/` - report, reproducible report/deck generator, and supporting project notes.
- `presentation/` - editable 15-slide PowerPoint.
- `tests/` - unit and Flask API integration tests.

## Website routes

- `/` - predictor, model selector, corpus visuals, transitions, and temporary training.
- `/dashboard` - statistics, word-frequency and transition analysis.
- `/mathematics` - formulas and computed worked example.
- `/about` - project scope and interpretation.
- `/api/predict?text=I%20love&order=2&limit=5` - returns the used context, observations, ranked predictions, and full probability distribution.
- `POST /api/train` with `{"text":"..."}` - tokenizes and uses a custom corpus for this running server session.
- `POST /api/reset` - restores the included training corpus.
- `GET /api/dashboard` - training statistics and transition tables.
- `GET /api/compare` - recalculates first- and second-order held-out metrics from the training and testing files.
- `GET /api/example` - computes the three-sentence worked example.
- `GET /health` - server health.

Custom text is limited to 100,000 characters and is held in memory only; it is not written to disk and resets on server restart. Do not paste sensitive text. This classroom build is intentionally unauthenticated and should not be used as a multi-user production service without isolating training state by user.

## Mathematical model

For context `c` and next word `w`, `P(w | c) = Count(c followed by w) / Count(c)`. First order uses the immediately preceding word; second order uses the previous two. Frequency is a raw count; relative frequency is a count divided by the relevant total. The model calculates probabilities from the actual training corpus. The UI plots the full distribution, although the ranked suggestion list is limited to five. For every observed context, probabilities sum to one. Ties are stable and sorted alphabetically.

The separate `data/classroom_example.txt` contains `I love Python`, `I love programming`, and `I love coding`; therefore each continuation is observed once out of three and each conditional probability is 1/3. This is a worked classroom corpus, not a claim about typical English. The main corpus is also authored for demonstration and is deliberately pattern-rich. Evaluation reports top-1 and top-3 among covered held-out contexts and gives coverage separately. The test file is not used to build model counts; results describe this small split only.

## Five-person work distribution

1. **Probability theory, mathematical formulas, and research:** review the Markov property, frequency/probability distinction, formulas, assumptions, and references.
2. **Dataset collection, cleaning, and frequency analysis:** document the authored corpus, tokenization, train/test separation, counts, and data limitations.
3. **Python programming, Markov chain, and backend:** own model implementation, Flask endpoints, input validation, and API tests.
4. **Frontend development, website design, and charts:** own responsive layouts, accessible controls, navigation, and Chart.js visuals.
5. **Testing, GitHub management, documentation, and presentation:** coordinate integration checks, review pull requests, maintain the report and slides, and package releases.

Add the actual five names and each person's completed contribution to the presentation/report before submission. Suggested collaboration: invite all teammates as repository collaborators, create separate branches for backend, data, frontend, and testing/docs work, and use reviewed pull requests.

## GitHub setup

The ZIP contains a clean project folder without a repository URL. Create a new repository first, then from this folder:

```bash
git init
git add .
git commit -m "Initial project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Never commit credentials or private data. Replace the placeholder remote with the URL for a repository you control.

## Reproducible deliverables

With dependencies installed, run `python documentation/generate_deliverables.py` to regenerate the report and editable slides from the included corpus, evaluation split, and current screenshots. The generated files are `documentation/Final_Report.pdf` and `presentation/Predictive_Text_Markov_Chain_15_Slides.pptx`. The final ZIP is `Predictive_Text_Markov_Chain_Project.zip` and contains a top-level `Predictive_Text_Markov_Project/` directory.
