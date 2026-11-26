from pathlib import Path
from datetime import date
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.markov_model import MarkovModel, compare_models, tokenize

TRAIN = (ROOT / "data" / "training.txt").read_text(encoding="utf-8")
TEST = (ROOT / "data" / "testing.txt").read_text(encoding="utf-8")
MODEL = MarkovModel(TRAIN)
STATS = MODEL.stats()
METRICS = compare_models(TRAIN, TEST)
EXAMPLE = (ROOT / "data" / "classroom_example.txt").read_text(encoding="utf-8")
EXAMPLE_RESULT = MarkovModel(EXAMPLE).predict("I love", order=2)
DOCS = ROOT / "documentation"
PRESENTATION_DIR = ROOT / "presentation"
DOCS.mkdir(exist_ok=True)
PRESENTATION_DIR.mkdir(exist_ok=True)
SCREENSHOT = ROOT / "screenshots" / "dashboard.png"
HOME_SCREENSHOT = ROOT / "screenshots" / "homepage.png"


def percent(value):
    return f"{value * 100:.1f}%"


def make_report():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

    navy = colors.HexColor("#152b49")
    blue = colors.HexColor("#4169e1")
    muted = colors.HexColor("#617087")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=27, leading=32, textColor=navy, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=12, leading=18, textColor=muted, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=navy, spaceBefore=13, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="SubSection", parent=styles["Heading2"], fontSize=11, leading=14, textColor=blue, spaceBefore=8, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyText2", parent=styles["BodyText"], fontSize=9.2, leading=14, textColor=colors.HexColor("#26354c"), spaceAfter=7))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=11, textColor=muted, spaceAfter=5))
    styles.add(ParagraphStyle(name="Equation", parent=styles["BodyText"], fontName="Courier", fontSize=9, leading=14, textColor=navy, backColor=colors.HexColor("#f1f5ff"), borderPadding=8, spaceBefore=4, spaceAfter=8))
    story = [Spacer(1, 1.3 * inch), Paragraph("Predictive Text and<br/>Next-Word Prediction Using Markov Chains", styles["CoverTitle"]), Paragraph("A transparent, interactive study of frequency-based language prediction", styles["SubTitle"]), Spacer(1, .45 * inch)]
    cover = [["COURSE PROJECT", "IMPLEMENTATION + EVALUATION"], ["NextWord Lab", "First- and second-order models"], ["Prepared for", "Classroom demonstration"], ["Date", date.today().strftime("%B %d, %Y")]]
    table = Table(cover, colWidths=[2.25 * inch, 3.8 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f7fb")), ("TEXTCOLOR", (0, 1), (-1, -1), navy), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9), ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#dce3ee")), ("PADDING", (0, 0), (-1, -1), 9)]))
    story += [table, Spacer(1, .35 * inch), Paragraph("Authors: add team member names and institution before submission", styles["SubTitle"]), PageBreak()]

    def section(title, body):
        story.append(Paragraph(title, styles["Section"]))
        if isinstance(body, list):
            story.extend(body)
        else:
            story.append(Paragraph(body, styles["BodyText2"]))

    def para(text):
        return Paragraph(text, styles["BodyText2"])

    section("Abstract", "This project implements an interactive educational next-word predictor using first- and second-order Markov models. The Flask backend learns from observed adjacent word sequences, then normalizes transition counts into conditional probabilities. The application exposes the ranked next-word distribution, corpus statistics, transition tables, and a separate held-out evaluation. The included corpus is an original, small teaching example designed to make repeated patterns inspectable; it is not representative of natural language. No neural model or paid service is involved.")
    section("Introduction", "Predictive text is familiar, but modern systems often obscure why a word was proposed. A Markov model offers an auditable baseline: define a finite context, count what followed it in training, and normalize those counts. This project studies how conditioning on one preceding word versus two changes the candidate distribution and evaluation coverage.")
    section("Problem statement", "A useful classroom predictor should connect each suggestion to observed evidence. How do first- and second-order context lengths affect next-word probabilities, prediction coverage, and held-out accuracy on a small teaching corpus? The application must expose counts and distinguish an unobserved context from a low-probability prediction.")
    section("Objectives", "Implement both model orders; explain frequency, relative frequency, conditional probability, and the Markov property; visualize corpus behavior; permit temporary custom-corpus training; and calculate held-out top-1, top-3, and coverage from a separate test file. The project focuses on mathematical transparency, not neural-network complexity.")
    section("Literature and background study", "Markov chains provide a formal framework for transitions among states (Norris, 1997). In text prediction, a word or finite context can be treated as a state and observed continuations estimate transition probabilities. This project is a simple n-gram baseline: it counts contiguous sequences and uses maximum-likelihood estimates, an interpretable technique discussed in introductory information-retrieval and language-processing material (Manning, Raghavan, and Schuetze, 2008; Jurafsky and Martin, Speech and Language Processing). It is distinct from modern neural language models, which learn distributed representations and longer-range patterns. No claim is made that this small demonstration corpus yields general-purpose predictions.")
    section("Mathematical foundation", [para("A Markov chain represents movement between states. For text, a token is an observed state and a context is one or more recent tokens. The first-order model conditions on one previous token; the second-order model conditions on a previous pair. Frequency is a count. Relative frequency divides a count by a relevant total, producing an empirical probability."), Paragraph("P(w | c) = Count(c followed by w) / Count(c)", styles["Equation"]), Paragraph("P(w_t | w_(t-1)) = Count(w_(t-1), w_t) / Count(w_(t-1))", styles["Equation"]), Paragraph("P(w_t | w_(t-2), w_(t-1)) = Count(w_(t-2), w_(t-1), w_t) / Count(w_(t-2), w_(t-1))", styles["Equation"]), para("For each context with observed next words, the conditional probabilities sum to one. The transition-probability matrix is represented sparsely by context-to-counter mappings; each row is normalized only when that context is requested. A context absent from training has no empirical estimate in this unsmoothed model."), para("Worked example from data/classroom_example.txt: " + "; ".join([f"P({row['word']} | I love) = {row['count']}/{EXAMPLE_RESULT['observations']} = {row['probability']:.3f}" for row in EXAMPLE_RESULT["distribution"]]) + f". The three probabilities sum to {sum(row['probability'] for row in EXAMPLE_RESULT['distribution']):.2f}.")])
    section("Dataset and methodology", [para(f"The training file contains {len(MODEL.tokens)} normalized tokens and {len(MODEL.unigrams)} unique terms. The separate held-out file contains {len(tokenize(TEST))} tokens and is not used to build transitions. The separate three-sentence worked-example file has exactly three 'I love' continuations for Python, programming, and coding."), para("Tokenization uses a regular expression to extract alphanumeric words and simple apostrophe contractions, lowercases text, normalizes curly apostrophes, and treats punctuation as a boundary. First-order counters are built from adjacent token pairs; second-order counters use adjacent triples. Evaluation walks through each test position and scores only contexts observed in training, while separately recording all evaluable cases."), para("The corpus is project-created illustrative text, deliberately pattern-rich for teaching. It is authored for this demonstration, not sampled from a population. Results must not be generalized to real-world language performance.")])
    section("System architecture", "The browser sends a context and selected order to Flask. Python collections.Counter objects store observed next-token frequencies under first-order and second-order contexts. The model normalizes the matching counts into a full conditional distribution; Flask returns these values as JSON. The dashboard renders the response and separate training-corpus statistics. The held-out route evaluates data/testing.txt. API routes also support custom training, reset, a calculated classroom example, and health checks.")
    section("Algorithm", [Paragraph("Train: tokenize the training text; for each adjacent pair, increment first[previous][next]; for each adjacent triple, increment second[(previous_1, previous_2)][next]. Predict: select the last k tokens as context; retrieve candidate counts; divide each count by the sum for that context; sort by descending count; return the full distribution and a limited ranked display.", styles["Equation"]), Paragraph("Evaluate for each test position t from order k onward: context = tokens[t-k:t]; actual = tokens[t]. Count all cases. If context occurs in training, increment covered and test top-ranked candidates. Top-1 and top-3 are computed on covered cases; coverage = covered / all evaluable cases.", styles["Equation"]), para("Accuracy and coverage must be read together. A highly specific order can have conditional accuracy on a small number of cases, so the interface reports both.")])
    section("Python implementation", "The code is divided into beginner-readable modules. backend/markov_model.py contains tokenization, transition counters, prediction, dashboard summaries, and held-out scoring. backend/statistics.py contains reusable frequency-distribution helpers. backend/app.py provides the Flask pages, request validation, JSON routes, and temporary training/reset behavior. The unit tests exercise the model and Flask test client. NumPy and Pandas are included in requirements for optional statistical exploration; core transition calculations use Python collections for transparency.")
    section("Statistical analysis and results",  [para(f"The current run used {STATS['total_words']} training tokens, {STATS['unique_words']} unique tokens, {STATS['first_transition_count']} first-order transitions, and {STATS['second_transition_count']} second-order transitions."), Table([["Order", "Test cases", "Covered", "Coverage", "Top-1 (covered)", "Top-3 (covered)"], *[[str(order), str(METRICS[str(order)]["test_cases"]), str(METRICS[str(order)]["covered"]), percent(METRICS[str(order)]["coverage"]), percent(METRICS[str(order)]["top1"]), percent(METRICS[str(order)]["top3"])] for order in (1, 2)]], colWidths=[.7*inch, .8*inch, .7*inch, .85*inch, 1.1*inch, 1.1*inch], style=TableStyle([("BACKGROUND", (0,0),(-1,0), navy),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#f4f7fb")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#dce3ee")),("FONTSIZE",(0,0),(-1,-1),7.5),("PADDING",(0,0),(-1,-1),6)])), Spacer(1, 7), para("Top-k correctness is calculated only when the evaluation context was seen in training. A no-smoothing model cannot propose a word for unseen contexts. These values describe this small held-out demonstration set only; they are not estimates of general language-model performance."), para("The live app additionally reports top-word frequencies, conditional probabilities, and per-context transition counts. It renders the full prediction distribution separately from the top-five candidate list.")])
    section("6. Interface and implementation", [para("NextWord Lab provides a dark navy / blue classroom dashboard with a context input, order toggle, predicted candidate list, frequency and probability charts, a custom-corpus training panel, and transition tables. A dedicated corpus analytics page summarizes first- and second-order transitions. A mathematics page states the formulas and a worked example; an about page explains scope and limitations."), para("Backend files: backend/markov_model.py contains tokenization, counters, prediction, summary statistics, and held-out scoring; backend/app.py contains Flask routes. Frontend templates and static CSS/JavaScript implement the browser interface. Unit tests check observed conditional probabilities, empty context/error handling, and held-out coverage.")])
    if SCREENSHOT.exists() and HOME_SCREENSHOT.exists():
        story.append(PageBreak())
        section("Application screenshots", [para("Actual captures from the running Flask application during verification."), para("Predictor home page"), Image(str(HOME_SCREENSHOT), width=6.4*inch, height=3.25*inch, kind="proportional"), para("Statistics dashboard"), Image(str(SCREENSHOT), width=6.4*inch, height=3.25*inch, kind="proportional")])
    section("Limitations", "Higher-order context can distinguish phrases that collapse together under a unigram context, but exact context counts become sparse as order increases. The sample corpus is tiny and deliberately repetitive. Tokenization is simple and treats punctuation as a boundary. There is no smoothing, backoff, semantic representation, or long-range dependency. Custom text is temporary in-memory state and resets with a server restart. The app is an unauthenticated classroom prototype, not a production multi-user predictor.")
    section("Future scope", "Possible extensions include evaluating a larger licensed corpus, adding a clearly documented backoff/smoothing option, presenting confidence and context coverage, exporting transition matrices, and recording exact corpus hashes and software versions. A larger corpus improves evidence only when split and provenance are controlled; it does not guarantee better predictions.")
    section("Conclusion", "NextWord Lab connects a working predictor to inspectable frequency counts and conditional probability. The two model orders provide a useful classroom comparison of context specificity, while held-out accuracy and coverage make the evaluation trade-off explicit. The result is a reproducible teaching instrument, not a production keyboard or a benchmark claim.")
    section("References", [para("1. Norris, J. R. (1997). Markov Chains. Cambridge University Press."), para("2. Manning, C. D., Raghavan, P., and Schuetze, H. (2008). Introduction to Information Retrieval. Cambridge University Press, Chapter 1."), para("3. Jurafsky, D. and Martin, J. H. Speech and Language Processing, 3rd edition draft. Stanford University. Record the exact draft version and access date required by the course."), para("4. Project-created illustrative corpus: data/training.txt and data/testing.txt. Not an external research dataset."), para("5. Python Software Foundation. Python 3 documentation; Flask documentation; Chart.js documentation. Consult the versions specified by the execution environment.")])
    section("Appendix A. Team responsibilities", "Suggested five-person division: (1) theory and formula review; (2) corpus preparation and statistical analysis; (3) Flask backend and Markov implementation; (4) interface and visualizations; (5) test coordination, repository packaging, documentation, and presentation. Replace role labels with actual names and describe the work each member personally completed. The report does not claim these roles were independently verified.")
    section("Appendix B. Reproducibility and project contents", "The project includes a README with setup instructions, separate training and test files, Flask routes, a unit test suite, and this report/deck generator. Start the Flask server and open the dashboard; use the built-in comparison view to recalculate metrics. Custom training is transient. For a course submission, add team and institution information and verify the course's citation and originality requirements.")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dce3ee"))
        canvas.line(.7*inch, .55*inch, 7.8*inch, .55*inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(.7*inch, .38*inch, "NextWord Lab | Markov-chain predictive text study")
        canvas.drawRightString(7.8*inch, .38*inch, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(DOCS / "Final_Report.pdf"), pagesize=letter, rightMargin=.7*inch, leftMargin=.7*inch, topMargin=.7*inch, bottomMargin=.75*inch, title="Predictive Text and Next-Word Prediction Using Markov Chains", author="NextWord Lab Project Team")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def make_presentation():
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.enum.shapes import MSO_SHAPE

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    navy = RGBColor(21, 43, 73); blue = RGBColor(65, 105, 225); pale = RGBColor(241, 245, 251); ink = RGBColor(38, 53, 76); muted = RGBColor(97, 112, 135)
    slides = [
        ("Predictive Text", "Next-word prediction using Markov chains", ["NextWord Lab | Probability and Statistics course project", "Student 1: [Name] - probability theory and research", "Student 2: [Name] - dataset and frequency analysis", "Student 3: [Name] - Python, Markov model, and backend", "Student 4: [Name] - frontend, design, and charts", "Student 5: [Name] - testing, GitHub, docs, and presentation", "College: [College Name] | University: [University Name]", "Date: " + date.today().strftime("%B %d, %Y")], "cover"),
        ("Introduction to Predictive Text", "A familiar feature, made mathematically inspectable", ["Predictive text ranks possible continuations from a preceding context.", "This project builds a transparent frequency-based baseline.", "Students can inspect transitions, counts, probabilities, and evaluation.", "The application is an educational demonstration, not a production keyboard."], "bullets"),
        ("Problem Statement", "How does context length change next-word prediction?", ["A prediction is useful to study only when its evidence can be inspected.", "The project compares one-word and two-word context models.", "The central question is how specificity affects prediction and test coverage.", "All results must be traceable to included training and held-out text."], "bullets"),
        ("Project Objectives", "Build a complete and reproducible learning loop", ["Implement first- and second-order Markov text predictors.", "Explain raw frequency and conditional probability with observed counts.", "Display predictions, transition tables, and corpus-level charts.", "Evaluate on a separate test file; report top-k accuracy and coverage.", "Package code, documentation, a 15-slide presentation, and report."], "bullets"),
        ("What Is a Markov Chain?", "A finite context summarizes recent history", ["A state represents a token; a transition represents an observed next token.", "First order conditions on the immediately preceding word.", "Second order conditions on the preceding pair of words.", "The Markov assumption is a simplifying approximation, not a semantic model.", "Unseen contexts have no estimate when no smoothing is used."], "bullets"),
        ("Probability and Statistical Concepts", "Counts become a conditional distribution", ["Frequency is a raw count; relative frequency divides by total observations.", "P(next word | context) = Count(context, next word) / Count(context).", "The model uses maximum-likelihood estimates from observed transitions.", "All observed continuations for a context sum to probability 1.", "Prediction candidates are ranked by count; ties are deterministic."], "bullets"),
        ("Conditional Probability Calculations", "Three observed continuations of 'I love'", [f"The separate worked-example corpus contains {EXAMPLE_RESULT['observations']} occurrences of the context 'I love'.", *[f"P({row['word']} | I love) = {row['count']} / {EXAMPLE_RESULT['observations']} = {row['probability']:.3f} ({row['percent']:.1f}%)." for row in EXAMPLE_RESULT["distribution"]], f"Total probability = {sum(row['probability'] for row in EXAMPLE_RESULT['distribution']):.2f}.", "Counts are computed from data/classroom_example.txt by the same model code."], "worked"),
        ("Dataset and Data Processing", "Separate training, holdout, and worked example", [f"Training: {len(MODEL.tokens)} tokens; {len(MODEL.unigrams)} unique terms.", f"Held-out test: {len(tokenize(TEST))} tokens, stored separately.", "A separate three-sentence file makes the probability calculation auditable.", "Tokenizer lowercases tokens and treats punctuation as a boundary.", "Project-authored teaching text; not representative population data."], "bullets"),
        ("System Architecture", "From text to ranked continuations", ["Browser sends a phrase and model order to the Flask JSON API.", "Python counters store observed adjacent pairs and triples.", "The model normalizes the matching context counts into probabilities.", "The interface renders ranked suggestions, charts, and transition tables.", "Held-out evaluation reads the separate test file."], "pipeline"),
        ("Website Features and Screenshots", "An interactive tool for exploration", ["Responsive NextWord Lab dashboard with order selection and live prediction.", "Corpus statistics and frequency / conditional-probability visualizations.", "First-order and second-order transition tables.", "Temporary custom-corpus training and a reset to the included corpus.", "Mathematics and project overview pages explain scope and assumptions."], "screenshot"),
        ("First-order vs Second-order Markov Chains", "One-word context versus two-word context", [f"Order 1: coverage {percent(METRICS['1']['coverage'])}; top-1 {percent(METRICS['1']['top1'])}.", f"Order 2: coverage {percent(METRICS['2']['coverage'])}; top-1 {percent(METRICS['2']['top1'])}.", "Accuracy is conditional on covered contexts; compare coverage too."], "comparison"),
        ("Statistical Dashboard and Graphs", "Make the evidence visible", ["Top-word frequency chart summarizes the training corpus.", "The prediction chart displays the full conditional distribution.", "Transition tables retain counts and observed context information.", "The chart and tables update from the same Flask model responses."], "screenshot"),
        ("Testing and Accuracy Evaluation", "A separate held-out file tests next-token ranking", [f"Order 1: top-1 {percent(METRICS['1']['top1'])}; top-3 {percent(METRICS['1']['top3'])} on covered cases.", f"Order 2: top-1 {percent(METRICS['2']['top1'])}; top-3 {percent(METRICS['2']['top3'])} on covered cases.", f"Order 1 coverage: {percent(METRICS['1']['coverage'])}; order 2 coverage: {percent(METRICS['2']['coverage'])}.", "Top-k scores are conditional on coverage; unseen contexts are not guessed.", "These are descriptive results on the included teaching split only."], "bullets"),
        ("Limitations and Future Scope", "Interpretability does not guarantee generalization", ["The small authored corpus is repetitive and cannot represent general English.", "Simple tokenization; no smoothing, backoff, or long-range dependencies.", "Custom training exists in memory only and is reset on restart.", "Future: licensed larger corpora, smoothing, provenance, and reproducible hashes.", "Future metrics should use a fixed, documented split and report coverage."], "bullets"),
        ("Conclusion and References", "A reproducible classroom baseline", ["Markov counts connect frequency to conditional probability.", "More context improves specificity but can reduce observed coverage.", "The dashboard, held-out evaluation, and project package are reproducible.", "Norris (1997), Markov Chains, Cambridge University Press.", "Manning, Raghavan & Schuetze (2008), Introduction to Information Retrieval, Ch. 1.", "Jurafsky & Martin, Speech and Language Processing, 3rd edition draft.", "Add verified team, institution, and course-specific citation details before submission."], "bullets"),
    ]

    def textbox(slide, x, y, w, h, text, size, color, bold=False, font="Aptos", align=None):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = shape.text_frame; tf.clear(); tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = text; p.font.name = font; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
        if align is not None: p.alignment = align
        return shape

    for index, (title, subtitle, bullets, kind) in enumerate(slides, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        bg = slide.background.fill; bg.solid(); bg.fore_color.rgb = navy if index == 1 else RGBColor(250, 251, 253)
        if index == 1:
            textbox(slide, .8, 1.0, 11.7, .45, "NEXTWORD LAB  /  COURSE PROJECT", 13, RGBColor(166, 190, 255), True)
            textbox(slide, .8, 2.0, 11.5, 1.3, title, 42, RGBColor(255, 255, 255), True)
            textbox(slide, .82, 3.35, 10.8, .6, subtitle, 22, RGBColor(207, 218, 239))
            textbox(slide, .85, 4.7, 11.4, 2.15, "\n".join(bullets), 12, RGBColor(225, 232, 246))
            continue
        textbox(slide, .55, .28, 12.1, .28, f"MARKOV CHAINS  /  PREDICTIVE TEXT                                               {index:02d} / 15", 9, muted, True)
        textbox(slide, .6, .75, 12.0, .55, title, 28, navy, True)
        textbox(slide, .62, 1.37, 12.0, .35, subtitle, 14, blue, True)
        if kind == "worked":
            for i, row in enumerate(EXAMPLE_RESULT["distribution"]):
                x = .8 + i * 4.15
                shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(2.3), Inches(3.7), Inches(2.0))
                shape.fill.solid(); shape.fill.fore_color.rgb = pale; shape.line.color.rgb = blue
                textbox(slide, x + .15, 2.6, 3.4, .5, row["word"].title(), 25, navy, True, align=PP_ALIGN.CENTER)
                textbox(slide, x + .15, 3.3, 3.4, .55, f"{row['count']} / {EXAMPLE_RESULT['observations']} = {row['probability']:.3f}", 17, blue, True, align=PP_ALIGN.CENTER)
                textbox(slide, x + .15, 3.88, 3.4, .3, f"{row['percent']:.1f}% of observed continuations", 11, muted, align=PP_ALIGN.CENTER)
            textbox(slide, .9, 4.8, 11.4, .6, f"P(Python | I love) + P(programming | I love) + P(coding | I love) = {sum(row['probability'] for row in EXAMPLE_RESULT['distribution']):.2f}", 17, navy, True, align=PP_ALIGN.CENTER)
            textbox(slide, 1.0, 5.6, 11.2, .4, "Computed from data/classroom_example.txt; each transition is observed once.", 12, muted, align=PP_ALIGN.CENTER)
        elif kind == "comparison":
            labels = [("Order 1", METRICS["1"]), ("Order 2", METRICS["2"])]
            for row_index, (label, metric) in enumerate(labels):
                y = 2.2 + row_index * 1.55
                textbox(slide, .9, y, 1.1, .35, label, 13, navy, True)
                for col_index, (metric_label, value) in enumerate((("Coverage", metric["coverage"]), ("Top-1", metric["top1"]))):
                    x = 2.3 + col_index * 5.0
                    textbox(slide, x, y, 1.1, .3, metric_label, 11, muted, True)
                    bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y + .42), Inches(3.8), Inches(.38))
                    bg.fill.solid(); bg.fill.fore_color.rgb = pale; bg.line.fill.background()
                    bar_width = max(.08, 3.8 * min(1, max(0, value)))
                    bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y + .42), Inches(bar_width), Inches(.38))
                    bar.fill.solid(); bar.fill.fore_color.rgb = blue if col_index == 0 else RGBColor(69, 166, 145); bar.line.fill.background()
                    textbox(slide, x + 3.92, y + .43, .75, .3, percent(value), 11, navy, True)
            textbox(slide, 1.0, 5.8, 11.2, .6, "Top-1 accuracy is calculated on covered contexts only. Coverage shows how many test contexts the model can evaluate.", 13, muted, align=PP_ALIGN.CENTER)
        elif kind == "pipeline":
            labels = ["TRAINING\nTEXT", "TOKENIZE\n+ COUNT", "CONTEXT\nLOOKUP", "PROBABILITY\nDISTRIBUTION", "DASHBOARD\n+ EVALUATION"]
            for i, label in enumerate(labels):
                x = .7 + i * 2.5
                shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(3.0), Inches(1.8), Inches(1.2))
                shape.fill.solid(); shape.fill.fore_color.rgb = pale; shape.line.color.rgb = blue
                shape.text_frame.text = label
                for p in shape.text_frame.paragraphs:
                    p.alignment = PP_ALIGN.CENTER; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = navy
                if i < 4: textbox(slide, x + 1.85, 3.35, .6, .35, "→", 21, blue, True)
            textbox(slide, .8, 5.0, 11.5, .8, "Flask serves JSON; the browser renders predictions, charts, and transition tables.", 18, ink)
        elif kind == "formula":
            textbox(slide, .85, 2.2, 11.5, .75, bullets[2], 22, navy, True, "Aptos", PP_ALIGN.CENTER)
            textbox(slide, 1.1, 3.25, 10.9, 1.6, "\n".join(bullets[:2] + bullets[3:]), 18, ink)
            shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(.85), Inches(5.5), Inches(11.6), Inches(.65))
            shape.fill.solid(); shape.fill.fore_color.rgb = pale; shape.line.fill.background()
            textbox(slide, 1.0, 5.65, 11.3, .3, "Observed counts only; probabilities are conditional on the exact context.", 13, muted)
        elif kind == "screenshot":
            if SCREENSHOT.exists():
                screenshot_path = HOME_SCREENSHOT if title == "Website Features and Screenshots" else SCREENSHOT
                slide.shapes.add_picture(str(screenshot_path), Inches(.75), Inches(2.05), width=Inches(8.2))
                textbox(slide, 9.2, 2.2, 3.3, 3.3, "\n".join("• " + item for item in bullets), 14, ink)
            else:
                textbox(slide, .9, 2.3, 11.5, 3.8, "\n".join("• " + item for item in bullets), 18, ink)
        else:
            textbox(slide, .85, 2.05, 11.6, 4.5, "\n".join("•  " + item for item in bullets), 18, ink)
        textbox(slide, .65, 7.05, 12.0, .2, "NextWord Lab  |  Teaching corpus; not representative language data", 8, muted)
    prs.save(PRESENTATION_DIR / "Predictive_Text_Markov_Chain_15_Slides.pptx")


if __name__ == "__main__":
    make_report()
    make_presentation()
    print(f"Generated {DOCS / 'Final_Report.pdf'} and {PRESENTATION_DIR / 'Predictive_Text_Markov_Chain_15_Slides.pptx'}")
    print(f"Training: {STATS['total_words']} tokens; held-out metrics: {METRICS}")
