"""PrepPilot single-file distribution.
Install: python -m pip install fastapi==0.162.0 uvicorn==0.54.0
Run: python quizbot.py
The embedded app is unpacked into a temporary folder while the server runs.
"""
from pathlib import Path
import argparse
import sys
import tempfile

FILES = {}

# ---------------- engine.py ----------------
FILES['engine.py'] = r'''"""Adaptive engine: per-chapter Elo/IRT ability with subject-level priors.

Model:   P(correct) = 1 / (1 + e^-(theta - b))   (1-parameter IRT / Elo)
Update:  theta += K * w * (score - P), where w comes from the response pattern
         (confidence, time taken, hints used) and K shrinks as evidence grows.
Prior:   an unseen chapter starts at its subject's ability, and every answer
         nudges the subject, so the model has an estimate for all chapters.
Policy:  pick the chapter with the best weakness + exploration + misconception
         score, then the question whose P(correct) is closest to TARGET_P.
"""
import math
import random

K_MIN = 0.4  # floor keeps tracking a student who is learning mid-session
TARGET_P = 0.6  # "desirable difficulty": right ~60% of the time (tuned via simulate)
# difficulty the student must handle on exam day; school classes default to 0 (class level)
EXAM_TARGET_B = {"JEE": 1.5, "NEET": 0.5, "Class 11": 0.5, "Class 12": 0.5}
GUESS_SECS, FAST_SECS, SLOW_SECS = 3, 25, 150


def p_correct(theta, b):
    return 1 / (1 + math.exp(-(theta - b)))


def level(b):
    return "Easy" if b < -0.5 else "Medium" if b < 1 else "Advanced"


def evidence(correct, secs, confidence, hints):
    """Turn a response pattern into (score, weight, misconception) for the update."""
    if secs < GUESS_SECS:
        confidence = "guess"  # too fast to have solved it
    score = max(0.0, 1 - 0.3 * hints) if correct else 0.0
    weight = {"sure": 1.2, "unsure": 1.0, "guess": 0.5}.get(confidence, 1.0)
    if correct and secs > SLOW_SECS:
        weight *= 0.7  # right but laboured: weaker evidence of mastery
    if correct and secs < FAST_SECS and confidence == "sure":
        weight *= 1.2  # fluent recall
    misconception = not correct and confidence == "sure"
    if misconception:
        weight *= 1.3  # confidently wrong is the strongest signal of a gap
    return score, weight, misconception


class Learner:
    def __init__(self, chapters):
        self.subject_of = chapters  # {chapter: subject}
        self.theta, self.n, self.flags = {}, {}, {}
        self.subj = {}
        self.history = []

    def ability(self, ch):
        return self.theta.get(ch, self.subj.get(self.subject_of[ch], 0.0))

    def readiness(self, ch, exam):
        return p_correct(self.ability(ch), EXAM_TARGET_B.get(exam, 0.0))

    def update(self, ch, b, correct, secs=30, confidence="unsure", hints=0):
        subject, theta = self.subject_of[ch], self.ability(ch)
        p = p_correct(theta, b)
        score, weight, misconception = evidence(correct, secs, confidence, hints)
        k = max(K_MIN, 0.8 / math.sqrt(1 + self.n.get(ch, 0)))
        delta = k * weight * (score - p)
        self.theta[ch] = theta + delta
        self.n[ch] = self.n.get(ch, 0) + 1
        self.subj[subject] = self.subj.get(subject, 0.0) + 0.3 * delta  # leaks to unseen chapters
        if misconception:
            self.flags[ch] = self.flags.get(ch, 0) + 1
        entry = dict(chapter=ch, subject=subject, b=b, correct=correct, p=p, score=score,
                     theta=self.theta[ch], before=theta, delta=delta, secs=round(secs, 1),
                     confidence=confidence, hints=hints, misconception=misconception)
        self.history.append(entry)
        return entry


def rank_chapters(learner, chapters, exam, recent=()):
    """Score every candidate chapter; highest first. Each row explains itself."""
    rows = []
    for ch in chapters:
        n = learner.n.get(ch, 0)
        parts = {
            "weakness": 1 - learner.readiness(ch, exam),
            "explore": 0.25 / math.sqrt(1 + n),
            "misconception": 0.25 * min(learner.flags.get(ch, 0), 2),
            "repeat_penalty": -0.6 if ch in list(recent)[-2:] else 0.0,
        }
        score = sum(parts.values()) + random.random() * 0.03  # tie-break
        rows.append(dict(chapter=ch, score=score, n=n, **parts))
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def target_b(learner, ch, easier=False):
    # P(correct) = TARGET_P  <=>  b = theta - logit(TARGET_P)
    return learner.ability(ch) - math.log(TARGET_P / (1 - TARGET_P)) - (0.6 if easier else 0.0)


def pick_item(items, tb, seen):
    fresh = [q for q in items if q["id"] not in seen]
    return min(fresh, key=lambda q: abs(q["b"] - tb)) if fresh else None


# ---------- why the student is wrong, not just that they are wrong ----------

ERROR_TYPES = {
    "concept_gap": ("Concept gap", "You don't have this idea yet. Read the explanation; next is an easier question on the same concept."),
    "careless": ("Careless slip", "You know this one: your level says you should get it. Slow down and re-read before locking in."),
    "time_pressure": ("Time pressure", "Ran out of time. Practise speed: aim to decide on this type in under a minute."),
    "trap": ("Fell for a trap option", "You were sure, but picked a tempting distractor. Before answering, ask what the trap option assumes."),
}


def diagnose(p, secs, confidence, switched_from_correct=False, timed_out=False, time_frac=0.0):
    """Classify a wrong answer from the response pattern.
    p = model's P(correct) before answering; time_frac = share of the time limit used (timed mode)."""
    if timed_out or time_frac > 0.85:
        return "time_pressure"
    rushed = GUESS_SECS <= secs < FAST_SECS * 0.6 and confidence != "guess"  # under 3 s is a guess, not a slip
    if switched_from_correct or (p >= 0.65 and rushed):
        return "careless"  # had it right, or should have known it and rushed
    if confidence == "sure":
        return "trap"  # confidently chose a wrong option: the distractor did its job
    return "concept_gap"


# ---------- behaviour: fatigue and frustration over the session ----------

def wellbeing(history, skips=0, window=6):
    """'ok' | 'frustrated' | 'tired', with the signals that triggered it."""
    recent = history[-window:]
    if len(recent) >= 3 and not any(h["correct"] for h in recent[-3:]):
        return "frustrated", ["3 wrong answers in a row"]
    if len(history) < 2 * window:
        return "ok", []
    early, late = history[:-window], recent
    acc = lambda hs: sum(h["correct"] for h in hs) / len(hs)
    secs = lambda hs: sum(h["secs"] for h in hs) / len(hs)
    reasons = []
    if acc(early) - acc(late) >= 0.3:
        reasons.append(f"accuracy fell from {acc(early):.0%} to {acc(late):.0%}")
    if secs(late) > 1.5 * secs(early) and secs(late) > FAST_SECS:
        reasons.append(f"answers slowed from {secs(early):.0f}s to {secs(late):.0f}s")
    if skips >= 2:
        reasons.append(f"{skips} recent skips")
    if sum(h.get("switches", 0) >= 2 for h in late) >= 3:
        reasons.append("lots of answer switching")
    return ("tired", reasons) if len(reasons) >= 2 else ("ok", reasons)


# ---------- exam strategy: marking scheme, calibration, predicted score ----------

# (+marks for right, -marks for wrong) per single-correct MCQ
MARKING = {"JEE": (3, 1), "NEET": (4, 1)}
CONF_LEVELS = ("sure", "unsure", "guess")


def expected_marks(p, exam):
    plus, minus = MARKING.get(exam, (1, 0))
    return p * plus - (1 - p) * minus


def calibration(history, exam):
    """Accuracy per confidence level, the expected marks of answering at that level, and advice."""
    out = []
    for c in CONF_LEVELS:
        hs = [h for h in history if h["confidence"] == c]
        if not hs:
            continue
        acc = sum(h["correct"] for h in hs) / len(hs)
        ev = expected_marks(acc, exam)
        advice = ("Attempt: positive expected marks." if ev > 0 else
                  "Skip these in the exam: guessing loses marks." if exam in MARKING else "No negative marking: always attempt.")
        out.append(dict(confidence=c, n=len(hs), accuracy=acc, expected=ev, advice=advice))
    return out


# ponytail: a paper is modelled as an even mix of 4 difficulty bands around the exam target;
# replace with real per-chapter weights and difficulty mix from past papers if available
PAPER_BANDS = (-2.0, -1.0, 0.0, 1.0)


def _chapter_marks(theta, exam):
    """(expected marks answering every question, answering only positive-EV ones) per question."""
    target = EXAM_TARGET_B.get(exam, 0.0)
    evs = [expected_marks(p_correct(theta, target + d), exam) for d in PAPER_BANDS]
    return sum(evs) / len(evs), sum(max(e, 0) for e in evs) / len(evs)


def predicted_score(learner, chapters, exam):
    """Expected % of max marks on a mixed-difficulty paper: answering everything vs skipping
    questions with negative expected value (the strategy gain)."""
    plus = MARKING.get(exam, (1, 0))[0]
    marks = [_chapter_marks(learner.ability(c), exam) for c in chapters]
    all_in = max(sum(a for a, _ in marks) / len(marks) / plus, 0)
    smart = sum(s for _, s in marks) / len(marks) / plus
    return dict(attempt_all=all_in, with_strategy=smart, strategy_gain=smart - all_in)


def best_gains(learner, chapters, exam, step=0.5, k=5):
    """Chapters where +step ability adds the most expected marks: where study time pays most."""
    rows = []
    for c in chapters:
        th = learner.ability(c)
        gain = _chapter_marks(th + step, exam)[1] - _chapter_marks(th, exam)[1]
        rows.append(dict(chapter=c, gain=gain, readiness=learner.readiness(c, exam)))
    return sorted(rows, key=lambda r: -r["gain"])[:k]


# ---------- simulation: adaptive vs static on synthetic students ----------

PROFILES = {
    "Uneven (2 strong, 4 weak)": [1.6, 1.3, -1.0, -1.5, -0.5, -2.0],
    "Weak all-round": [-1.5, -1.8, -1.2, -2.0, -1.6, -1.4],
    "Nearly exam-ready": [1.2, 0.8, 1.5, 0.5, 1.9, 0.9],
}


def _student_answers(rng, true_theta, ch, b):
    correct = rng.random() < p_correct(true_theta[ch], b)
    # ponytail: learning gain peaks when b matches true ability (zone of proximal development)
    true_theta[ch] += 0.15 * math.exp(-((b - true_theta[ch]) ** 2) / 2)
    return correct


def simulate(policy, start, exam="JEE", seed=0, max_q=800):
    """Questions until every chapter's TRUE readiness reaches 0.7."""
    rng = random.Random(seed)
    random.seed(seed)
    chapters = {f"Ch{i + 1}": "Sim" for i in range(len(start))}
    true_theta = dict(zip(chapters, start))
    goal = EXAM_TARGET_B.get(exam, 0.0) + math.log(0.7 / 0.3)
    learner, recent = Learner(chapters), []
    for q in range(1, max_q + 1):
        if policy == "adaptive":
            ch = rank_chapters(learner, chapters, exam, recent)[0]["chapter"]
            b = target_b(learner, ch) + rng.gauss(0, 0.3)
        else:  # static bank: fixed chapter order, difficulty spread across the bank
            ch = list(chapters)[q % len(chapters)]
            b = rng.uniform(-2, 3)
        correct = _student_answers(rng, true_theta, ch, b)
        learner.update(ch, b, correct)
        recent.append(ch)
        if all(t >= goal for t in true_theta.values()):
            return q
    return max_q


def compare(runs=20, exam="JEE"):
    out = []
    for name, start in PROFILES.items():
        a = sum(simulate("adaptive", start, exam, s) for s in range(runs)) / runs
        st = sum(simulate("static", start, exam, s) for s in range(runs)) / runs
        out.append(dict(profile=name, adaptive=round(a), static=round(st), saved=round(100 * (1 - a / st))))
    return out


if __name__ == "__main__":
    L = Learner({"A": "S", "B": "S", "C": "S"})
    t0 = L.ability("A")
    L.update("A", 0.0, True, 20, "sure")
    assert L.ability("A") > t0, "correct answer must raise ability"
    assert L.ability("B") > t0, "subject prior must lift unseen chapters"
    t1 = L.ability("A")
    L.update("A", 0.0, False, 20, "sure")
    assert L.ability("A") < t1 and L.flags["A"] == 1, "confident wrong must lower ability and flag"
    g = Learner({"X": "S"}); g.update("X", 0.0, False, 1, "sure")
    s = Learner({"X": "S"}); s.update("X", 0.0, False, 30, "sure")
    assert g.ability("X") > s.ability("X"), "1-second answer is a guess, weaker penalty"
    for _ in range(3):
        L.update("C", 0.0, False, 30, "sure")
    assert rank_chapters(L, ["A", "B", "C"], "JEE")[0]["chapter"] == "C", "weakest chapter first"
    assert diagnose(0.5, 40, "unsure", timed_out=True) == "time_pressure"
    assert diagnose(0.8, 5, "unsure") == "careless"
    assert diagnose(0.3, 40, "unsure", switched_from_correct=True) == "careless"
    assert diagnose(0.3, 40, "sure") == "trap"
    assert diagnose(0.3, 40, "guess") == "concept_gap"
    h = lambda ok, s=20: dict(correct=ok, secs=s, confidence="unsure")
    assert wellbeing([h(1), h(0), h(0), h(0)])[0] == "frustrated"
    assert wellbeing([h(1, 15)] * 6 + [h(0, 60), h(1, 60)] * 3)[0] == "tired"
    assert wellbeing([h(1)] * 12)[0] == "ok"
    cal = calibration([dict(correct=False, confidence="guess")] * 5 + [dict(correct=True, confidence="guess")], "NEET")
    assert cal[0]["expected"] < 0 and "Skip" in cal[0]["advice"], "17% guesses lose marks at +4/-1 (break-even 20%)"
    ps = predicted_score(Learner({"A": "S", "B": "S"}), ["A", "B"], "JEE")
    assert ps["with_strategy"] >= ps["attempt_all"], "skipping negative-EV questions never hurts"
    res = compare(runs=10)
    assert all(r["adaptive"] < r["static"] for r in res), res
    print("engine ok", res)
'''

# ---------------- syllabus.py ----------------
FILES['syllabus.py'] = r'''"""Full syllabus map: JEE Advanced (Physics, Chemistry, Maths) and NEET (Physics, Chemistry, Biology).

Chapters marked JEE-only below are those outside the current NEET syllabus.
Verify the exam tags against the latest NTA / JEE Advanced brochures before the final demo.
"""

BOTH, JEE, NEET = ("JEE", "NEET"), ("JEE",), ("NEET",)

SYLLABUS = {
    "Physics": {
        "Units and Measurements": BOTH, "Kinematics": BOTH, "Laws of Motion": BOTH,
        "Work, Energy and Power": BOTH, "Rotational Motion": BOTH, "Gravitation": BOTH,
        "Mechanical Properties of Solids": BOTH, "Mechanical Properties of Fluids": BOTH,
        "Thermal Properties of Matter": BOTH, "Thermodynamics": BOTH, "Kinetic Theory of Gases": BOTH,
        "Oscillations": BOTH, "Waves": BOTH, "Electrostatics": BOTH, "Capacitance": BOTH,
        "Current Electricity": BOTH, "Moving Charges and Magnetism": BOTH, "Magnetism and Matter": BOTH,
        "Electromagnetic Induction": BOTH, "Alternating Current": BOTH, "Electromagnetic Waves": BOTH,
        "Ray Optics": BOTH, "Wave Optics": BOTH, "Dual Nature of Matter and Radiation": BOTH,
        "Atoms": BOTH, "Nuclei": BOTH, "Semiconductor Electronics": BOTH, "Experimental Physics": JEE,
    },
    "Chemistry": {
        "Some Basic Concepts of Chemistry": BOTH, "Atomic Structure": BOTH,
        "Classification of Elements and Periodicity": BOTH, "Chemical Bonding": BOTH,
        "States of Matter": JEE, "Chemical Thermodynamics": BOTH, "Chemical Equilibrium": BOTH,
        "Ionic Equilibrium": BOTH, "Redox Reactions": BOTH, "Solutions": BOTH, "Electrochemistry": BOTH,
        "Chemical Kinetics": BOTH, "Solid State": JEE, "Surface Chemistry": JEE, "s-Block Elements": JEE,
        "p-Block Elements": BOTH, "d- and f-Block Elements": BOTH, "Coordination Compounds": BOTH,
        "Metallurgy": JEE, "Qualitative Analysis": BOTH, "General Organic Chemistry": BOTH,
        "Hydrocarbons": BOTH, "Haloalkanes and Haloarenes": BOTH, "Alcohols, Phenols and Ethers": BOTH,
        "Aldehydes, Ketones and Carboxylic Acids": BOTH, "Amines": BOTH, "Biomolecules": BOTH,
        "Polymers": JEE,
    },
    "Maths": {
        "Sets, Relations and Functions": JEE, "Complex Numbers": JEE, "Quadratic Equations": JEE,
        "Sequences and Series": JEE, "Permutations and Combinations": JEE, "Binomial Theorem": JEE,
        "Matrices and Determinants": JEE, "Trigonometric Functions": JEE, "Inverse Trigonometric Functions": JEE,
        "Trigonometric Equations": JEE, "Properties of Triangles": JEE, "Straight Lines": JEE, "Circles": JEE,
        "Parabola": JEE, "Ellipse": JEE, "Hyperbola": JEE, "Limits": JEE,
        "Continuity and Differentiability": JEE, "Application of Derivatives": JEE,
        "Indefinite Integration": JEE, "Definite Integration": JEE, "Area Under Curves": JEE,
        "Differential Equations": JEE, "Vectors": JEE, "3D Geometry": JEE, "Probability": JEE,
        "Statistics": JEE,
    },
    "Biology": {
        "The Living World": NEET, "Biological Classification": NEET, "Plant Kingdom": NEET,
        "Animal Kingdom": NEET, "Morphology of Flowering Plants": NEET, "Anatomy of Flowering Plants": NEET,
        "Structural Organisation in Animals": NEET, "Cell: The Unit of Life": NEET, "Biomolecules (Biology)": NEET,
        "Cell Cycle and Cell Division": NEET, "Photosynthesis in Higher Plants": NEET,
        "Respiration in Plants": NEET, "Plant Growth and Development": NEET,
        "Breathing and Exchange of Gases": NEET, "Body Fluids and Circulation": NEET,
        "Excretory Products and their Elimination": NEET, "Locomotion and Movement": NEET,
        "Neural Control and Coordination": NEET, "Chemical Coordination and Integration": NEET,
        "Sexual Reproduction in Flowering Plants": NEET, "Human Reproduction": NEET,
        "Reproductive Health": NEET, "Principles of Inheritance and Variation": NEET,
        "Molecular Basis of Inheritance": NEET, "Evolution": NEET, "Human Health and Disease": NEET,
        "Microbes in Human Welfare": NEET, "Biotechnology: Principles and Processes": NEET,
        "Biotechnology and its Applications": NEET, "Organisms and Populations": NEET, "Ecosystem": NEET,
        "Biodiversity and Conservation": NEET,
    },
}

# School classes, CBSE / NCERT aligned. Classes 1-5 and 7-8 use topic names (they work across
# textbook editions); Class 6 follows the new NCERT books; Classes 11-12 reuse the exam chapter
# names above so the verified bank questions apply. Verify against current NCERT before the demo.
ENGLISH_PRIMARY = ["Alphabet and Phonics", "Naming Words (Nouns)", "Action Words (Verbs)", "Describing Words (Adjectives)",
                   "Opposites and Rhyming Words", "Simple Sentences", "Reading Comprehension"]
ENGLISH_MIDDLE = ["Nouns and Pronouns", "Verbs and Tenses", "Adjectives and Adverbs", "Articles and Determiners",
                  "Prepositions and Conjunctions", "Active and Passive Voice", "Direct and Indirect Speech",
                  "Vocabulary: Synonyms and Antonyms", "Reading Comprehension", "Writing Skills"]
ENGLISH_SECONDARY = ["Tenses", "Modals", "Subject-Verb Agreement", "Active and Passive Voice", "Reported Speech",
                     "Determiners", "Clauses and Sentence Transformation", "Reading Comprehension", "Writing Skills"]

CLASSES = {
    1: {
        "Maths": ["Shapes and Space", "Numbers 1 to 9", "Addition", "Subtraction", "Numbers 10 to 20", "Time",
                  "Measurement", "Numbers 21 to 50", "Data Handling", "Patterns", "Numbers 51 to 100", "Money"],
        "English": ENGLISH_PRIMARY,
        "EVS": ["My Body", "My Family", "Food We Eat", "Plants Around Us", "Animals Around Us", "Our Home",
                "Clothes We Wear", "Good Habits and Safety", "Seasons and Weather"],
    },
    2: {
        "Maths": ["Numbers up to 100", "Place Value", "Addition with Carrying", "Subtraction with Borrowing",
                  "Multiplication Basics", "Shapes and Patterns", "Measuring Length", "Weight and Capacity",
                  "Time and Calendar", "Money", "Data Handling"],
        "English": ENGLISH_PRIMARY + ["Singular and Plural", "Punctuation Basics"],
        "EVS": ["Our Body and Health", "Food and Nutrition", "Plants", "Animals and Birds", "Water", "Air",
                "Our Neighbourhood", "Means of Transport", "Festivals"],
    },
    3: {
        "Maths": ["Numbers up to 1000", "Addition and Subtraction", "Multiplication", "Division",
                  "Halves and Quarters", "Shapes and Designs", "Measurement", "Time", "Money", "Patterns",
                  "Data Handling"],
        "English": ENGLISH_MIDDLE[:5] + ["Punctuation", "Reading Comprehension"],
        "EVS": ["Family and Friends", "Plants and Leaves", "Animals and Their Homes", "Food and Cooking", "Water",
                "Houses and Shelter", "Travel and Communication", "Our Earth and Sky", "Work and Play"],
    },
    4: {
        "Maths": ["Large Numbers", "Addition and Subtraction", "Multiplication", "Division", "Factors and Multiples",
                  "Fractions", "Introduction to Decimals", "Lines and Shapes", "Perimeter and Area", "Measurement",
                  "Time", "Money", "Patterns", "Data Handling"],
        "English": ENGLISH_MIDDLE[:5] + ["Subject and Predicate", "Punctuation", "Vocabulary: Synonyms and Antonyms",
                                         "Reading Comprehension"],
        "EVS": ["Our Body Systems", "Food and Digestion", "Plants and Their Parts", "Animal Life", "Water Sources",
                "Soil and Agriculture", "Travel and Maps", "Natural Resources", "Our Environment"],
    },
    5: {
        "Maths": ["Large Numbers and Place Value", "Operations on Numbers", "Factors and Multiples", "Fractions",
                  "Decimals", "Measurement", "Angles and Geometry", "Perimeter, Area and Volume", "Time",
                  "Money and Profit", "Patterns", "Data Handling and Graphs"],
        "English": ENGLISH_MIDDLE,
        "EVS": ["Senses of Animals", "Food and Health", "Water and Its Conservation", "Plants and Seeds", "Our Body",
                "Maps and Directions", "Weather and Climate", "Natural Disasters", "Fuels and Energy"],
    },
    6: {
        "Maths": ["Patterns in Mathematics", "Lines and Angles", "Number Play", "Data Handling and Presentation",
                  "Prime Time", "Perimeter and Area", "Fractions", "Playing with Constructions", "Symmetry",
                  "The Other Side of Zero"],
        "Science": ["The Wonderful World of Science", "Diversity in the Living World", "Mindful Eating",
                    "Exploring Magnets", "Measurement of Length and Motion", "Materials Around Us",
                    "Temperature and its Measurement", "States of Water", "Methods of Separation",
                    "Living Creatures and their Characteristics", "Nature's Treasures", "Beyond Earth"],
        "Social Science": ["Locating Places on the Earth", "Oceans and Continents", "Landforms and Life",
                           "Timeline and Sources of History", "India, That Is Bharat",
                           "The Beginnings of Indian Civilisation", "India's Cultural Roots", "Unity in Diversity",
                           "Family and Community", "Grassroots Democracy", "The Value of Work",
                           "Economic Activities Around Us"],
        "English": ENGLISH_MIDDLE,
    },
    7: {
        "Maths": ["Integers", "Fractions and Decimals", "Data Handling", "Simple Equations", "Lines and Angles",
                  "The Triangle and its Properties", "Comparing Quantities", "Rational Numbers", "Perimeter and Area",
                  "Algebraic Expressions", "Exponents and Powers", "Symmetry", "Visualising Solid Shapes"],
        "Science": ["Nutrition in Plants", "Nutrition in Animals", "Heat", "Acids, Bases and Salts",
                    "Physical and Chemical Changes", "Respiration in Organisms",
                    "Transportation in Animals and Plants", "Reproduction in Plants", "Motion and Time",
                    "Electric Current and its Effects", "Light", "Forests: Our Lifeline", "Wastewater Story"],
        "Social Science": ["Tracing Changes Through a Thousand Years", "New Kings and Kingdoms", "The Delhi Sultans",
                           "The Mughal Empire", "Environment", "Inside Our Earth", "Our Changing Earth", "Air",
                           "Water", "Equality in Indian Democracy", "State Government", "Markets Around Us"],
        "English": ENGLISH_MIDDLE,
    },
    8: {
        "Maths": ["Rational Numbers", "Linear Equations in One Variable", "Understanding Quadrilaterals",
                  "Data Handling", "Squares and Square Roots", "Cubes and Cube Roots", "Comparing Quantities",
                  "Algebraic Expressions and Identities", "Mensuration", "Exponents and Powers",
                  "Direct and Inverse Proportions", "Factorisation", "Introduction to Graphs"],
        "Science": ["Crop Production and Management", "Microorganisms", "Coal and Petroleum", "Combustion and Flame",
                    "Conservation of Plants and Animals", "Reproduction in Animals",
                    "Reaching the Age of Adolescence", "Force and Pressure", "Friction", "Sound",
                    "Chemical Effects of Electric Current", "Some Natural Phenomena", "Light"],
        "Social Science": ["How, When and Where", "From Trade to Territory", "Ruling the Countryside",
                           "Colonialism and the City", "Women, Caste and Reform", "The Making of the National Movement",
                           "Resources", "Land, Soil, Water, Natural Vegetation and Wildlife", "Agriculture",
                           "Industries", "The Indian Constitution", "Parliament and the Making of Laws",
                           "The Judiciary"],
        "English": ENGLISH_MIDDLE,
    },
    9: {
        "Maths": ["Number Systems", "Polynomials", "Coordinate Geometry", "Linear Equations in Two Variables",
                  "Introduction to Euclid's Geometry", "Lines and Angles", "Triangles", "Quadrilaterals", "Circles",
                  "Heron's Formula", "Surface Areas and Volumes", "Statistics"],
        "Science": ["Matter in Our Surroundings", "Is Matter Around Us Pure", "Atoms and Molecules",
                    "Structure of the Atom", "The Fundamental Unit of Life", "Tissues", "Motion",
                    "Force and Laws of Motion", "Gravitation", "Work and Energy", "Sound",
                    "Improvement in Food Resources"],
        "Social Science": ["The French Revolution", "Socialism in Europe and the Russian Revolution",
                           "Nazism and the Rise of Hitler", "India: Size and Location", "Physical Features of India",
                           "Drainage", "Climate", "Natural Vegetation and Wildlife", "Population",
                           "What is Democracy?", "Constitutional Design", "Electoral Politics",
                           "Working of Institutions", "Democratic Rights", "The Story of Village Palampur",
                           "People as Resource", "Poverty as a Challenge", "Food Security in India"],
        "English": ENGLISH_SECONDARY,
    },
    10: {
        "Maths": ["Real Numbers", "Polynomials", "Pair of Linear Equations in Two Variables", "Quadratic Equations",
                  "Arithmetic Progressions", "Triangles", "Coordinate Geometry", "Introduction to Trigonometry",
                  "Some Applications of Trigonometry", "Circles", "Areas Related to Circles",
                  "Surface Areas and Volumes", "Statistics", "Probability"],
        "Science": ["Chemical Reactions and Equations", "Acids, Bases and Salts", "Metals and Non-metals",
                    "Carbon and its Compounds", "Life Processes", "Control and Coordination",
                    "How do Organisms Reproduce?", "Heredity", "Light: Reflection and Refraction",
                    "The Human Eye and the Colourful World", "Electricity", "Magnetic Effects of Electric Current",
                    "Our Environment"],
        "Social Science": ["The Rise of Nationalism in Europe", "Nationalism in India", "The Making of a Global World",
                           "The Age of Industrialisation", "Print Culture and the Modern World",
                           "Resources and Development", "Forest and Wildlife Resources", "Water Resources",
                           "Agriculture", "Minerals and Energy Resources", "Manufacturing Industries",
                           "Lifelines of National Economy", "Power Sharing", "Federalism",
                           "Gender, Religion and Caste", "Political Parties", "Outcomes of Democracy", "Development",
                           "Sectors of the Indian Economy", "Money and Credit", "Globalisation and the Indian Economy"],
        "English": ENGLISH_SECONDARY,
    },
    11: {
        "Physics": ["Units and Measurements", "Kinematics", "Laws of Motion", "Work, Energy and Power",
                    "Rotational Motion", "Gravitation", "Mechanical Properties of Solids",
                    "Mechanical Properties of Fluids", "Thermal Properties of Matter", "Thermodynamics",
                    "Kinetic Theory of Gases", "Oscillations", "Waves"],
        "Chemistry": ["Some Basic Concepts of Chemistry", "Atomic Structure",
                      "Classification of Elements and Periodicity", "Chemical Bonding", "Chemical Thermodynamics",
                      "Chemical Equilibrium", "Ionic Equilibrium", "Redox Reactions", "General Organic Chemistry",
                      "Hydrocarbons"],
        "Maths": ["Sets, Relations and Functions", "Trigonometric Functions", "Complex Numbers", "Quadratic Equations",
                  "Linear Inequalities", "Permutations and Combinations", "Binomial Theorem", "Sequences and Series",
                  "Straight Lines", "Circles", "Parabola", "Ellipse", "Hyperbola", "Introduction to 3D Geometry",
                  "Limits", "Statistics", "Probability"],
        "Biology": list(SYLLABUS["Biology"])[:19],
    },
    12: {
        "Physics": ["Electrostatics", "Capacitance", "Current Electricity", "Moving Charges and Magnetism",
                    "Magnetism and Matter", "Electromagnetic Induction", "Alternating Current",
                    "Electromagnetic Waves", "Ray Optics", "Wave Optics", "Dual Nature of Matter and Radiation",
                    "Atoms", "Nuclei", "Semiconductor Electronics"],
        "Chemistry": ["Solutions", "Electrochemistry", "Chemical Kinetics", "d- and f-Block Elements",
                      "Coordination Compounds", "Haloalkanes and Haloarenes", "Alcohols, Phenols and Ethers",
                      "Aldehydes, Ketones and Carboxylic Acids", "Amines", "Biomolecules"],
        "Maths": ["Relations and Functions", "Inverse Trigonometric Functions", "Matrices and Determinants",
                  "Continuity and Differentiability", "Application of Derivatives", "Indefinite Integration",
                  "Definite Integration", "Area Under Curves", "Differential Equations", "Vectors", "3D Geometry",
                  "Linear Programming", "Probability"],
        "Biology": list(SYLLABUS["Biology"])[19:],
    },
}

# level key -> display name, in menu order
EXAMS = {"JEE": "JEE Advanced", "NEET": "NEET"}
LEVELS = {**{f"Class {n}": f"Class {n}" for n in CLASSES}, **EXAMS}
BANK_LEVELS = {"Class 11", "Class 12", "JEE", "NEET"}  # where the verified PCMB bank applies


def subjects_for(level):
    """{subject: [chapters]} in syllabus order."""
    if level in EXAMS:
        return {s: [c for c, ex in chs.items() if level in ex] for s, chs in SYLLABUS.items()
                if any(level in ex for ex in chs.values())}
    return CLASSES[int(level.split()[1])]


def chapters_for(level):
    """{chapter: subject} for every chapter at this level."""
    return {ch: subj for subj, chs in subjects_for(level).items() for ch in chs}


if __name__ == "__main__":
    for lv in LEVELS:
        subj = subjects_for(lv)
        n = sum(len(c) for c in subj.values())
        assert len(chapters_for(lv)) == n, f"{lv}: chapter name repeated across subjects"
        print(f"{lv:9} {n:3} chapters  {', '.join(subj)}")
    assert "Chemical Coordination and Integration" in CLASSES[11]["Biology"]
    assert CLASSES[12]["Biology"][0] == "Sexual Reproduction in Flowering Plants"
'''

# ---------------- verify_bank.py ----------------
FILES['verify_bank.py'] = r'''"""Re-check every answer key in questions.json.

Numeric questions: the answer is recomputed here from the question's data, and exactly one
option must equal it (this also catches "no correct option" and duplicate-option bugs).
Conceptual questions: a second, independent review names the correct option text.
Imported past-paper questions ("key_source": "official") carry the official exam key.
Every other question must be covered by one of the two lists, so nothing enters the bank unchecked.

Run: python verify_bank.py
"""
import json
import math
import re
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
# questions_imported.json is written by import_jeebench.py (official JEE Advanced keys)
QS = [q for name in ("questions.json", "questions_imported.json") if (HERE / name).exists()
      for q in json.loads((HERE / name).read_text(encoding="utf-8"))]

COMPUTED = {
    # Kinematics
    "phy-kin-1": (20 - 0) / 5,
    "phy-kin-2": 20 ** 2 / (2 * 10),
    "phy-kin-3": 20 ** 2 * math.sin(math.radians(90)) / 10,
    "phy-kin-4": (lambda x: abs(x(1) - x(0)) + abs(x(3) - x(1)) + abs(x(4) - x(3)))(lambda t: t**3 - 6 * t**2 + 9 * t),
    "phy-kin-5": 2 * 20 * 30 / (20 + 30),
    "phy-kin-6": 0 + 10 / 2 * (2 * 3 - 1),
    "phy-kin-7": (30**2 - 10**2) / (2 * 100),
    "phy-kin-8": 3 * 400 / 5,
    "phy-kin-9": math.sin(math.radians(30)) ** 2 / math.sin(math.radians(60)) ** 2,
    "phy-kin-10": (lambda x: abs(x(4 / 3) - x(0)) + abs(x(2) - x(4 / 3)))(lambda t: t**3 - 2 * t**2),
    # Laws of Motion
    "phy-lom-1": 10 / 2,
    "phy-lom-2": 60 * (10 + 2),
    "phy-lom-3": (3 - 2) * 10 / (3 + 2),
    "phy-lom-4": 2 * min(18 / 6, 0.4 * 10),  # moves together if common a <= mu*g
    "phy-lom-5": 0.5 * (10 - (-10)),
    "phy-lom-6": (20 - 0.2 * 5 * 10) / 5,
    "phy-lom-7": 10 * (0.6 - 0.5 * 0.8),
    "phy-lom-8": math.sqrt(0.4 * 10 * 40),
    "phy-lom-9": 2 * (10 - (2 * 10 - 0.25 * 4 * 10) / 6),
    "phy-lom-10": 10 * 0.75,
    # Some Basic Concepts of Chemistry
    "che-mole-1": 36 / 18,
    "che-mole-2": (4 / 40) / 0.5,
    "che-mole-3": 2 * 22.4,
    "che-mole-4": 18 * 2 * min(10 / 2 / 2, 64 / 32),  # 2H2 + O2 -> 2H2O, limiting reagent
    "che-mole-5": 4.4 / 44 * 6.022e23,
    "che-mole-6": 4 * 16 / (2 + 32 + 64) * 100,
    "che-mole-8": (0.5 * 100 + 0.2 * 200) / 300,
    "che-mole-9": 0.66 * 12 / 44 / 0.2 * 100,
    "che-mole-10": 11.2 * 1.5,
    # Chemical Equilibrium
    "che-eq-3": 4 / (1 + 4),          # x/(1-x) = 4
    "che-eq-6": 1 / 4,
    "che-eq-7": 4 ** 2,
    "che-eq-8": 24.6 / (0.082 * 300),
    "che-eq-10": 2 * (8 / (2 + 8)),   # 2x/(1-x) = 8
    # Quadratic Equations
    "mat-quad-2": 3 ** 2 - 2 * 1,
    "mat-quad-5": 8 / 2,
    "mat-quad-7": F(5, 6),
    "mat-quad-8": min((x / 100) ** 2 - 6 * (x / 100) + 13 for x in range(-1000, 1001)),
    "mat-quad-9": sum(1 for k in range(-10, 11) if 4 * k * k - 16 < 0),
    "mat-quad-10": min(range(-5, 6), key=lambda a: (a - 2) ** 2 + 2 * (a - 1)),
    # Probability
    "mat-prob-1": 1 - F(1, 4),
    "mat-prob-2": F(4 + 13 - 1, 52),
    "mat-prob-3": F(sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == 7), 36),
    "mat-prob-4": F(1, 2) * F(3, 5) / (F(1, 2) * F(3, 5) + F(1, 2) * F(2, 5)),
    "mat-prob-5": F(len([p for p in range(1, 7) if p in (2, 3, 5)]), 6),
    "mat-prob-6": F(4, 52) * F(3, 51),
    "mat-prob-7": 0.4 + 0.5 - 0.4 * 0.5,
    "mat-prob-8": F(3, 8),
    "mat-prob-9": 1 - F(5, 6) ** 4,
    "mat-prob-10": F(1, 3),  # {BB} out of {BB, BG, GB}
    # Principles of Inheritance and Variation
    "bio-inh-4": F(1, 2),
    "bio-inh-8": 3 * 3,
    "bio-inh-9": F(1, 2) * F(1, 2),
    "bio-inh-10": F(1, 2) * F(1, 2),
}

# Conceptual / symbolic questions: (text that must appear in the keyed option, why it is correct)
REVIEWED = {
    "che-eq-1": ("Forward", "4 mol gas -> 2 mol gas; higher pressure favours fewer moles"),
    "che-eq-2": ("(RT)^{-2}", "delta n_g = 2 - 4 = -2"),
    "che-eq-4": (r"\alpha^2 P}{1-\alpha^2}", "partial pressures with total moles 1 + alpha"),
    "che-eq-5": ("unchanged", "catalyst speeds both directions equally"),
    "che-eq-9": ("backward", "Q = 0.25/0.5 = 0.5 > K = 0.2"),
    "che-mole-7": (r"\text{CH}_2\text{O}$", "C 3.33 : H 6.7 : O 3.33 = 1 : 2 : 1"),
    "mat-quad-1": ("2, 3", "(x-2)(x-3)"),
    "mat-quad-3": (r"\pm 6", "k^2 = 36"),
    "mat-quad-4": ("(-1, 3)", "roots a-1, a+1 inside (-2, 4)"),
    "mat-quad-6": ("x^2 + x - 6", "sum -1, product -6"),
    "bio-cell-1": ("Mitochondrion", "site of aerobic respiration / ATP"),
    "bio-cell-2": ("70S", "50S + 30S subunits"),
    "bio-cell-3": ("Ribosome", "only non-membranous option"),
    "bio-cell-4": ("9 + 0, triplets", "centriole cartwheel: 9 peripheral triplets, no central pair"),
    "bio-cell-5": ("Schleiden and Schwann", "1838-39; Virchow modified it in 1855"),
    "bio-cell-6": ("Lysosome", "acid hydrolases"),
    "bio-cell-7": ("Nuclear membrane", "prokaryote DNA lies in a nucleoid"),
    "bio-cell-8": ("Lipid synthesis", "SER: lipids and steroid hormones"),
    "bio-cell-9": ("plasma membrane", "mesosomes are plasma membrane infoldings"),
    "bio-cell-10": ("Glycoproteins and glycolipids", "NCERT: Golgi forms glycoproteins and glycolipids"),
    "bio-inh-1": ("3 : 1", "monohybrid F2 phenotype"),
    "bio-inh-2": ("9 : 3 : 3 : 1", "dihybrid F2 phenotype"),
    "bio-inh-3": ("Incomplete dominance", "pink F1 is intermediate"),
    "bio-inh-5": ("homozygous recessive", "definition of test cross"),
    "bio-inh-6": ("1 : 1", "Tt x tt"),
    "bio-inh-7": ("Codominance and multiple alleles", "IA, IB, i; AB expresses both"),
}


def value(option):
    """Numeric value of an option like '$4\\sqrt{10}\\,\\text{m/s}$' or '$\\frac{3}{4}$', else None."""
    s = option.replace("$", "")
    s = re.sub(r"\\text\{[^}]*\}(\^\{?-?\d+\}?)?", "", s)  # units, with any exponent
    s = s.replace(r"\%", "").replace(r"\,", "").replace(r"\dfrac", r"\frac").replace(r"\times", "*")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"*sqrt(\1)", s)
    s = re.sub(r"\^\{(-?\d+)\}", r"**(\1)", s)
    s = re.sub(r"^\s*\*", "", s).strip()
    if not re.fullmatch(r"[\d\s.+\-*/()sqrt]+", s):
        return None
    try:  # ponytail: eval on our own repo file only, characters whitelisted above
        v = eval(s, {"__builtins__": {}, "sqrt": math.sqrt})
    except Exception:
        return None
    return float(v) if isinstance(v, (int, float)) else None


def check():
    errors, ids = [], set()
    for q in QS:
        qid, opts, key = q["id"], q["options"], q["answer"]
        if qid in ids:
            errors.append(f"{qid}: duplicate id")
        ids.add(qid)
        if len(opts) != 4 or len(set(opts)) != 4 or key not in range(4):
            errors.append(f"{qid}: needs 4 distinct options and answer 0-3")
            continue
        if q.get("key_source") == "official":
            pass  # official exam answer key; structure was checked above
        elif qid in COMPUTED:
            want = float(COMPUTED[qid])
            hits = [i for i, o in enumerate(opts) if (v := value(o)) is not None and math.isclose(v, want, rel_tol=5e-3)]
            if hits != [key]:
                errors.append(f"{qid}: computed {want:.4g}, matching options {hits}, key {key}")
        elif qid in REVIEWED:
            text, _ = REVIEWED[qid]
            # exact match wins, so "3 : 1" is not confused with "9 : 3 : 3 : 1"
            hits = [i for i, o in enumerate(opts) if o == text] or [i for i, o in enumerate(opts) if text in o]
            if hits != [key]:
                errors.append(f"{qid}: reviewed answer '{text}' is not uniquely the keyed option")
        else:
            errors.append(f"{qid}: not checked: add it to COMPUTED or REVIEWED")
    stale = (set(COMPUTED) | set(REVIEWED)) - ids
    errors += [f"{s}: check exists but question is missing" for s in sorted(stale)]
    return errors


if __name__ == "__main__":
    errs = check()
    print(f"{len(QS)} questions: {sum(q['id'] in COMPUTED for q in QS)} recomputed, "
          f"{sum(q['id'] in REVIEWED for q in QS)} reviewed, "
          f"{sum(q.get('key_source') == 'official' for q in QS)} official exam keys")
    for e in errs:
        print("FAIL", e)
    assert not errs, f"{len(errs)} problem(s)"
    print("all answer keys verified")
'''

# ---------------- main.py ----------------
FILES['main.py'] = r'''"""PrepPilot API: adaptive quiz bot for JEE Advanced / NEET.

The engine (engine.py) decides WHAT to ask; Gemini only decides HOW to say it
(new questions for chapters the bank does not cover, wrong-answer explanations, doubts).
Run:  uvicorn main:app --reload
"""
import threading
from collections import defaultdict
from functools import wraps
import hashlib
import hmac
import json
import math
import secrets
import os
import re
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import engine
import verify_bank
from syllabus import BANK_LEVELS, EXAMS, LEVELS, chapters_for, subjects_for

ROOT = Path(__file__).parent
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
# second model is the fallback when the first is overloaded (503) or retired (404)
GEMINI_MODELS = [os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"), "gemini-flash-lite-latest"]

BANK = {}  # chapter -> verified questions (apply to BANK_LEVELS only)
_errors = verify_bank.check()
_failed = {e.split(":")[0] for e in _errors}
for e in _errors:
    print("Bank question rejected:", e)
for q in verify_bank.QS:
    if q["id"] not in _failed:  # only answer keys that pass verify_bank.py reach students
        BANK.setdefault(q["chapter"], []).append({**q, "verified": True})

GENERATED = {}  # (level, chapter) -> AI questions, cached per level so Class 4 and Class 7 "Fractions" differ

# ponytail: in-memory sessions, lost on restart; move to SQLite if you need persistence
SESSIONS = {}
POOL = ThreadPoolExecutor(max_workers=3)
PENDING = {}  # (level, chapter) -> Future of an in-flight background generation
COOLDOWN = [0.0]  # skip Gemini until this timestamp after rate limiting
app = FastAPI()


def parse_json(text):
    """Models often leave LaTeX backslashes unescaped: "\\frac" would parse as form-feed + "rac"."""
    text = re.sub(r'(?<!\\)\\(?=[a-zA-Z]{2,})', r'\\\\', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'(?<!\\)\\(?![\\"/u])', r'\\\\', text))


def gemini(prompt, as_json=False):
    if not GEMINI_KEY or time.time() < COOLDOWN[0]:
        return None
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if as_json:
        body["generationConfig"] = {"responseMimeType": "application/json"}
    limited = 0
    for model in GEMINI_MODELS:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY},
        )
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                text = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
            return parse_json(text) if as_json else text.strip()
        except Exception as e:  # network, quota, malformed JSON: caller falls back to the bank
            limited += getattr(e, "code", None) == 429
            print(f"Gemini error ({model}):", e)
    if limited == len(GEMINI_MODELS):
        COOLDOWN[0] = time.time() + 60  # every model rate-limited: stop paying the latency for a minute
    return None


def generate_question(chapter, subject, exam, b):
    """Gemini writes a question at the engine's target difficulty, then a second call must solve it
    from scratch (and may answer -1 for "no option is correct"); used only if both agree."""
    if exam in EXAMS:
        band = {"Easy": "NCERT / board level", "Medium": "JEE Main / NEET level",
                "Advanced": "JEE Advanced level, multi-step"}[engine.level(b)]
    else:
        band = {"Easy": "easy", "Medium": "standard", "Advanced": "challenging"}[engine.level(b)] + f" for {exam}"
    for _ in range(2):
        q = gemini(
            f"Write one original multiple-choice question for a {audience(exam)}.\n"
            f"Subject: {subject}. Chapter: {chapter}. Difficulty: {band}.\n"
            "Rules: exactly 4 options, exactly one correct, no diagrams or pictures needed, wording suited to the "
            "student's age, use $...$ LaTeX only for maths and units, numbers chosen so the answer is clean.\n"
            "Work in this order: write the question, fully solve it in 'solution', "
            "then build the options so one equals your solved result and three are common mistakes.\n"
            'Return JSON: {"q": str, "solution": str, "options": [str, str, str, str], "answer": 0-3, '
            '"hints": [2 progressive hints that do not reveal the answer], "explanation": str}', True)
        if not (isinstance(q, dict) and len(q.get("options", [])) == 4 and q.get("answer") in range(4)):
            continue
        opts = "\n".join(f"{i}. {o}" for i, o in enumerate(q["options"]))
        check = gemini(
            "Solve this exam question independently, step by step, before looking at the options. "
            "Then pick the option equal to your result. If no option matches exactly, answer -1.\n"
            f"{q['q']}\n{opts}\n"
            'Reply JSON: {"working": str, "answer": -1 to 3}', True)
        if isinstance(check, dict) and check.get("answer") == q["answer"]:
            q.update(id="ai-" + uuid.uuid4().hex[:8], chapter=chapter, b=b, verified=False,
                     hints=(q.get("hints") or ["Re-read the question.", "Eliminate options."])[:2])
            GENERATED.setdefault((exam, chapter), []).append(q)
            return q
    return None


def audience(level):
    return f"{EXAMS[level]} aspirant" if level in EXAMS else f"{level} student (CBSE / NCERT)"


def items(s, ch):
    """All questions available for this chapter at the session's level."""
    verified = BANK.get(ch, []) if s["exam"] in BANK_LEVELS else []
    return [q for q in verified + GENERATED.get((s["exam"], ch), [])
            if q["id"] not in s.get("reported", set())]


def get_session(sid):
    if sid not in SESSIONS:
        raise HTTPException(404, "Session expired. Reload the page.")
    return SESSIONS[sid]


SESSION_LOCKS = defaultdict(threading.RLock)


def session_locked(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        request = kwargs.get("req", args[0] if args else None)
        sid = kwargs.get("session_id", getattr(request, "session_id", "start"))
        with SESSION_LOCKS[sid]:
            return fn(*args, **kwargs)
    return wrapped


def scope(s):
    chs = list(s["chapters"])
    f = s["focus"]
    if f in s["chapters"].values():
        chs = [c for c in chs if s["chapters"][c] == f]
    elif f in s["chapters"]:
        chs = [f]
    if not GEMINI_KEY:  # without generation only chapters with unseen bank questions can be served
        chs = [c for c in chs if engine.pick_item(items(s, c), 0, s["seen"])]
    return chs


def ready(s, ch, tb):
    """Unseen question in this chapter close enough to the target difficulty, or None."""
    q = engine.pick_item(items(s, ch), tb, s["seen"])
    # real past-paper questions get a wider window: better a slightly hard real question than an AI one
    return q if q and abs(q["b"] - tb) <= (1.5 if q.get("key_source") == "official" else 0.75) else None


def prefetch(s):
    """Generate questions for the likely next chapters while the student is busy answering/reading."""
    if not GEMINI_KEY:
        return
    L = s["learner"]
    # ponytail: top-1 only, free-tier Gemini rate-limits (429) quickly; raise to [:2] on a paid key
    for r in engine.rank_chapters(L, scope(s), s["exam"], s["recent"])[:1]:
        ch, tb = r["chapter"], engine.target_b(L, r["chapter"])
        key = (s["exam"], ch)
        running = key in PENDING and not PENDING[key].done()
        if not running and not ready(s, ch, tb):
            PENDING[key] = POOL.submit(generate_question, ch, s["chapters"][ch], s["exam"], round(tb, 2))


def public(q):
    return {**{k: q[k] for k in ("id", "chapter", "q", "options", "b", "verified")},
            "source": q.get("source", ""), "official": q.get("key_source") == "official"}


def next_question(s):
    L, exam = s["learner"], s["exam"]
    chs = scope(s)
    if not chs:
        s["seen"].clear()  # bank exhausted: allow repeats
        chs = scope(s)
    if not chs:
        raise HTTPException(400, "No questions for this focus yet. Add some to questions.json or set GEMINI_API_KEY.")
    ranked = engine.rank_chapters(L, chs, exam, s["recent"])
    retry = s["retry"] if s["retry"] in chs else None
    boost = s.pop("boost", None)  # tired/frustrated: one confidence-builder from a strong chapter
    tb_of = lambda c: engine.target_b(L, c, easier=c == retry)
    # near-ties (within 0.05 of the top score) are broken by which chapter has a question ready
    near = [retry] if retry else [r["chapter"] for r in ranked if r["score"] >= ranked[0]["score"] - 0.05]
    if boost:
        last = s["recent"][-1] if s["recent"] else None
        strong = sorted((c for c in chs if c != last), key=lambda c: -L.readiness(c, exam)) or chs
        near, retry = strong[:3], None
        tb_of = lambda c: L.ability(c) - math.log(0.85 / 0.15)  # P(correct) ~ 85%
    ch = next((c for c in near if ready(s, c, tb_of(c))), None)
    if ch is None and GEMINI_KEY:
        ch = near[0]
        fut = PENDING.get((exam, ch))
        if fut:
            try:
                fut.result(timeout=45)  # prefetch already running for this chapter: wait for it
            except Exception:
                pass
        if not ready(s, ch, tb_of(ch)):
            generate_question(ch, s["chapters"][ch], exam, round(tb_of(ch), 2))
        if not ready(s, ch, tb_of(ch)):
            ch = None
    if ch is None:  # no AI or generation failed: best-ranked chapter with any unseen bank question
        ch = next((r["chapter"] for r in ranked if engine.pick_item(items(s, r["chapter"]), 0, s["seen"])), None)
    if ch is None:
        raise HTTPException(503, "Could not get a question. Check the Gemini key or add bank questions.")
    tb = tb_of(ch)
    q = engine.pick_item(items(s, ch), tb, s["seen"])

    row = next(r for r in ranked if r["chapter"] == ch)
    p = engine.p_correct(L.ability(ch), q["b"])
    s.update(current=q, asked_at=time.time(), hints=0)
    s["seen"].add(q["id"])
    s["recent"].append(ch)
    why = {
        "chapter": ch, "subject": s["chapters"][ch], "retry": ch == retry,
        "readiness": L.readiness(ch, exam), "answered": L.n.get(ch, 0), "flags": L.flags.get(ch, 0),
        "ability": L.ability(ch), "target_b": tb, "b": q["b"], "p": p, "level": engine.level(q["b"]),
        "source": (f"Real past paper: {q['source']}" if q.get("key_source") == "official"
                   else f"Verified bank ({q['source']}), answer key re-checked by verify_bank.py") if q.get("verified")
                  else "AI-generated, self-checked",
        "score_parts": {k: round(row[k], 3) for k in ("weakness", "explore", "misconception", "repeat_penalty")},
        "candidates": [{"chapter": r["chapter"], "score": round(r["score"], 3)} for r in ranked[:5]],
        "pool": len(chs), "boost": boost,
        "last_error": (engine.ERROR_TYPES[L.history[-1]["diagnosis"]][0]
                       if L.history and L.history[-1].get("diagnosis") else None),
    }
    s["why_log"].append(why)
    prefetch(s)
    return {"question": public(q), "why": why}


class Start(BaseModel):
    level: str = "JEE"
    demo: bool = False  # short focus blocks and breaks, so judges can see a full cycle
    resume_id: str = ""  # browser session to resume when preferences match
    block_start: float = 0  # carried over when switching class mid-block (epoch seconds)
    break_until: float = 0  # carried over so switching class can't skip a break
    # from the setup page
    subjects: list[str] = []  # empty = every subject at this level
    start_ability: float = 0.0  # -1 just starting, 0 comfortable, +1 confident (seeds the subject priors)
    block_minutes: int = 40
    explain_style: str = "steps"  # steps | short | analogy


EXPLAIN = {"steps": "Explain step by step, numbering each step.",
           "short": "Be brief and direct: the key idea in one or two sentences.",
           "analogy": "Use a simple everyday analogy, then connect it back to the concept."}


def rhythm(s):
    """Focus block and break lengths (seconds). Breaks scale with the block: 40 min -> 10 / 15 min."""
    if s["demo"]:
        return dict(block=120, short=30, long=45)
    block = s["block_minutes"] * 60
    return dict(block=block, short=round(block / 4), long=round(block * 3 / 8))


def clock(s, now=None):
    now = now or time.time()
    r = rhythm(s)
    return {"block_secs": r["block"], "block_left": max(0, s["block_start"] + r["block"] - now),
            "break_left": max(0, s["break_until"] - now), "block_start": s["block_start"],
            "break_until": s["break_until"], "breaks": s["breaks"]}


def start_break(s, reason):
    """Lock the session for a break; questions are refused until it ends."""
    hist = s["learner"].history
    state, _ = engine.wellbeing(hist, sum(i >= len(hist) - 6 for i in s["skips"]))
    r, now = rhythm(s), time.time()
    length = r["long"] if state != "ok" else r["short"]
    s.update(break_until=now + length, block_start=now + length, breaks=s["breaks"] + 1, current=None)
    done = hist[s["block_mark"]:]
    s["block_mark"] = len(hist)
    raise HTTPException(423, {**clock(s, now), "reason": reason, "long": state != "ok",
                              "block_answered": len(done),
                              "block_accuracy": sum(h["correct"] for h in done) / len(done) if done else 0})


def guard_break(s):
    now = time.time()
    if now < s["break_until"]:
        raise HTTPException(423, {**clock(s, now), "reason": "Break in progress", "long": False,
                                  "block_answered": 0, "block_accuracy": 0})
    if now - s["block_start"] >= rhythm(s)["block"]:
        start_break(s, "Focus block complete")


class Sid(BaseModel):
    session_id: str


class Answer(Sid):
    choice: int  # -1 = no answer (timed out)
    confidence: str = "unsure"
    picks: list[int] = []  # every option clicked before submitting, to detect answer switching
    time_limit: float = 0  # seconds, 0 = untimed
    timed_out: bool = False


class Report(Sid):
    answer_id: str


class Focus(Sid):
    focus: str = "all"


class Ask(Sid):
    text: str


@app.get("/")
def landing():
    return FileResponse(ROOT / "landing.html")


@app.get("/setup")
def setup_page():
    return FileResponse(ROOT / "setup.html")


@app.get("/study")
def study_page():
    return FileResponse(ROOT / "study.html")


@app.get("/api/catalog")
def catalog(level: str = "JEE"):
    if level not in LEVELS:
        raise HTTPException(400, "Unknown class or exam")
    return {"name": LEVELS[level], "ai": bool(GEMINI_KEY), "subjects": [
        {"subject": subject, "chapters": [
            {"chapter": ch, "questions": len(BANK.get(ch, [])) if level in BANK_LEVELS else 0}
            for ch in chapters]}
        for subject, chapters in subjects_for(level).items()]}


@app.get("/api/levels")
def levels():
    """Every class/exam with its subjects and chapter counts, for the setup page."""
    return [{"level": lv, "name": name,
             "subjects": [{"subject": s, "chapters": len(c)} for s, c in subjects_for(lv).items()]}
            for lv, name in LEVELS.items()]


@app.get("/login")
def login_page():
    return FileResponse(ROOT / "login.html")


@app.get("/app")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/site.css")
def site_css():
    return FileResponse(ROOT / "site.css")


# ---------- accounts ----------
# ponytail: JSON file + in-memory tokens, fine for a demo; move to a database before real users
USERS_FILE = ROOT / "users.json"
USERS_LOCK = threading.RLock()
TOKENS = {}  # token -> email


def _users():
    return json.loads(USERS_FILE.read_text(encoding="utf-8")) if USERS_FILE.exists() else {}


def _hash(password, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000).hex()


def _profile(u):
    return {k: u[k] for k in ("name", "email", "level")}


class Signup(BaseModel):
    name: str
    email: str
    password: str
    level: str = "JEE"


class Login(BaseModel):
    email: str
    password: str


@app.post("/api/signup")
def signup(req: Signup):
    email, name = req.email.strip().lower(), req.name.strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(400, "Enter a valid email address.")
    if not name or len(name) > 60:
        raise HTTPException(400, "Enter your name.")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    if req.level not in LEVELS:
        raise HTTPException(400, "Pick a class or exam.")
    # FastAPI runs sync handlers concurrently. Serialize the whole read/modify/write.
    with USERS_LOCK:
        users = _users()
        if email in users:
            raise HTTPException(409, "An account with this email already exists. Sign in instead.")
        salt = secrets.token_hex(16)
        users[email] = dict(name=name, email=email, level=req.level, salt=salt, hash=_hash(req.password, salt))
        temporary = USERS_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(users, indent=1), encoding="utf-8")
        os.replace(temporary, USERS_FILE)
    token = secrets.token_urlsafe(32)
    TOKENS[token] = email
    return {"token": token, "user": _profile(users[email])}


@app.post("/api/login")
def login(req: Login):
    u = _users().get(req.email.strip().lower())
    if not u or not hmac.compare_digest(u["hash"], _hash(req.password, u["salt"])):
        raise HTTPException(401, "Wrong email or password.")  # same message either way: don't reveal accounts
    token = secrets.token_urlsafe(32)
    TOKENS[token] = u["email"]
    return {"token": token, "user": _profile(u)}


@app.get("/api/me")
def me(token: str):
    email = TOKENS.get(token)
    if not email or email not in (users := _users()):
        raise HTTPException(401, "Signed out")
    return _profile(users[email])


@app.post("/api/logout")
def logout(token: str = ""):
    TOKENS.pop(token, None)
    return {"ok": True}


@app.post("/api/session")
@session_locked
def start(req: Start):
    if req.level not in LEVELS:
        raise HTTPException(400, f"level must be one of {list(LEVELS)}")
    subjects = {k: v for k, v in subjects_for(req.level).items() if not req.subjects or k in req.subjects}
    if not subjects:
        raise HTTPException(400, "Pick at least one subject available at this level.")
    chapters = {ch: subj for subj, chs in subjects.items() for ch in chs}
    resume = SESSIONS.get(req.resume_id)
    settings = (req.level, sorted(subjects), req.demo, max(-1.5, min(1.5, req.start_ability)),
                max(10, min(90, req.block_minutes)), req.explain_style if req.explain_style in EXPLAIN else "steps")
    if resume and resume.get("settings") == settings:
        return session_info(req.resume_id, resume)
    learner = engine.Learner(chapters)
    start_ability = max(-1.5, min(1.5, req.start_ability))
    learner.subj = {subj: start_ability for subj in subjects}  # prior from "how confident are you?"
    sid = uuid.uuid4().hex
    SESSIONS[sid] = dict(exam=req.level, chapters=chapters, learner=learner, focus="all",
                         reported=set(), reports=[], settings=settings, initial_subj=dict(learner.subj),
                         seen=set(), recent=[], retry=None, current=None, asked_at=0, hints=0, why_log=[],
                         skips=[], intervened_at=-99, pred_log=[], demo=req.demo, breaks=0, block_mark=0,
                         block_start=time.time(),
                         break_until=0, block_minutes=max(10, min(90, req.block_minutes)),
                         explain_style=req.explain_style if req.explain_style in EXPLAIN else "steps")
    return session_info(sid, SESSIONS[sid])


def session_info(sid, s):
    subjects = {}
    for ch, subj in s["chapters"].items():
        subjects.setdefault(subj, []).append(ch)
    banked = sum(1 for c in s["chapters"] if BANK.get(c)) if s["exam"] in BANK_LEVELS else 0
    return {"session_id": sid, "exam": LEVELS[s["exam"]], "marking": engine.MARKING.get(s["exam"]),
            "time_limit": {"JEE": 150, "NEET": 60}.get(s["exam"], 90),
            "subjects": [{"subject": k, "chapters": v} for k, v in subjects.items()],
            "chapters": len(s["chapters"]), "banked_chapters": banked, "ai": bool(GEMINI_KEY),
            "focus": s["focus"], "clock": clock(s),
            "active": {"question": public(s["current"]), "why": s["why_log"][-1],
                       "elapsed": max(0, time.time() - s["asked_at"])} if s["current"] else None}


@app.post("/api/next")
@session_locked
def next_(req: Sid):
    s = get_session(req.session_id)
    guard_break(s)  # 423 when a break is due or running; a question already on screen can still be answered
    return {**next_question(s), "clock": clock(s)}


@app.post("/api/break")
@session_locked
def take_break(req: Sid):
    """Student (or the fatigue nudge) starts a break early."""
    start_break(get_session(req.session_id), "Break taken early")


@app.post("/api/hint")
@session_locked
def hint(req: Sid):
    s = get_session(req.session_id)
    q = s["current"]
    if not q:
        raise HTTPException(400, "No active question")
    if s["hints"] >= len(q["hints"]):
        return {"hint": None, "used": s["hints"]}
    s["hints"] += 1
    return {"hint": q["hints"][s["hints"] - 1], "used": s["hints"]}


@app.post("/api/answer")
@session_locked
def answer(req: Answer):
    s = get_session(req.session_id)
    q, L = s["current"], s["learner"]
    if not q:
        raise HTTPException(400, "No active question")
    secs = time.time() - s["asked_at"]
    correct = req.choice == q["answer"]
    before = L.readiness(q["chapter"], s["exam"])
    h = L.update(q["chapter"], q["b"], correct, secs, req.confidence, s["hints"])
    h.update(answer_id=uuid.uuid4().hex, question_id=q["id"])
    if q.get("verified"):
        old_b = q["b"]
        q["b"] = round(q["b"] - 0.05 * (h["score"] - h["p"]), 3)
        h["item_delta"] = q["b"] - old_b  # item calibration from all students
    switched_from_correct = not correct and q["answer"] in req.picks[:-1]
    h["switches"] = max(len(req.picks) - 1, 0)
    h["diagnosis"] = None if correct else engine.diagnose(
        h["p"], secs, req.confidence, switched_from_correct, req.timed_out,
        secs / req.time_limit if req.time_limit else 0.0)
    # concept gaps and traps get an easier follow-up in the same chapter; slips and time errors don't
    s["retry"] = None if s["retry"] else (q["chapter"] if h["diagnosis"] in ("concept_gap", "trap") else None)
    s["current"] = None
    s["last_wrong"] = None if correct else (q, req.choice)
    s["last"] = (q, h)

    n = len(L.history)
    s["pred_log"].append(round(engine.predicted_score(L, list(s["chapters"]), s["exam"])["with_strategy"], 4))
    state, reasons = engine.wellbeing(L.history, sum(i >= n - 6 for i in s["skips"]))
    intervention = None
    if state != "ok" and n - s["intervened_at"] >= 5:
        s["intervened_at"], s["boost"] = n, (
            "Confidence builder after a rough patch" if state == "frustrated" else "Fresh topic, easier question: you seem tired")
        intervention = {"state": state, "reasons": reasons, "message":
            "Three misses in a row happens to everyone. Next up: a question from one of your strong chapters to reset."
            if state == "frustrated" else
            "Your focus is dropping (" + "; ".join(reasons) + "). Take a 5-minute break; when you're back I'll switch topic and start easier."}
    prefetch(s)

    coach = None
    if n % 5 == 0 and not intervention:
        seen = sorted(L.theta, key=lambda c: L.readiness(c, s["exam"]))
        coach = (f"Check-in after {n} questions: strongest so far is {seen[-1]} "
                 f"({L.readiness(seen[-1], s['exam']):.0%} ready), weakest is {seen[0]} "
                 f"({L.readiness(seen[0], s['exam']):.0%}). I'll keep pulling you towards {seen[0]}.")
    return {"answer_id": h["answer_id"], "correct": correct, "answer": q["answer"], "explanation": q["explanation"], "secs": round(secs, 1),
            "delta": h["delta"], "misconception": h["misconception"], "guess": secs < engine.GUESS_SECS,
            "readiness_before": before, "readiness_after": L.readiness(q["chapter"], s["exam"]),
            "retry_next": bool(s["retry"]), "coach": coach, "intervention": intervention,
            "diagnosis": h["diagnosis"] and dict(zip(("label", "fix"), engine.ERROR_TYPES[h["diagnosis"]]),
                                                  type=h["diagnosis"])}


@app.post("/api/skip")
@session_locked
def skip(req: Sid):
    s = get_session(req.session_id)
    s["current"] = None
    s["skips"].append(len(s["learner"].history))  # skips feed fatigue detection
    return {"ok": True}


@app.post("/api/explain")
def explain(req: Sid):
    """Personalised take on the last wrong answer; separate call so the result shows instantly."""
    s = get_session(req.session_id)
    if not s.get("last_wrong"):
        return {"explanation": None}
    q, choice = s["last_wrong"]
    if not 0 <= choice < len(q["options"]):
        return {"explanation": None}
    return {"explanation": gemini(
        f"A {audience(s['exam'])} answered this question wrongly.\n"
        f"Question: {q['q']}\nOptions: {q['options']}\nThey chose: {q['options'][choice]}\n"
        f"Correct: {q['options'][q['answer']]}\nReference solution: {q['explanation']}\n"
        "In at most 5 sentences: name the likely misconception behind their choice, then show the correct "
        f"reasoning. {EXPLAIN[s['explain_style']]} Use $...$ LaTeX. Be encouraging, no greeting.")}


@app.post("/api/report")
@session_locked
def report(req: Report):
    """Exclude exactly this answer and hide its question only for this learner."""
    s = get_session(req.session_id)
    old = s["learner"]
    target = next((h for h in old.history if h.get("answer_id") == req.answer_id), None)
    if target is None:
        if any(r["answer_id"] == req.answer_id for r in s["reports"]):
            return {"removed": True}
        raise HTTPException(404, "Answer not found in this session")
    for q in BANK.get(target["chapter"], []):
        if q["id"] == target["question_id"]:
            q["b"] = round(q["b"] - target.get("item_delta", 0), 3)
    s["reported"].add(target["question_id"])
    s["reports"].append({"answer_id": req.answer_id, "question_id": target["question_id"]})
    # Replaying retained evidence also restores unseen-chapter priors and score history.
    learner = engine.Learner(s["chapters"])
    learner.subj = dict(s["initial_subj"])
    predictions = []
    removed_at = old.history.index(target)
    for previous in old.history:
        if previous is target:
            continue
        entry = learner.update(previous["chapter"], previous["b"], previous["correct"],
                               previous["secs"], previous["confidence"], previous["hints"])
        for key in ("answer_id", "question_id", "switches", "diagnosis", "item_delta"):
            if key not in previous:
                continue
            entry[key] = previous[key]
        predictions.append(round(engine.predicted_score(learner, list(s["chapters"]), s["exam"])["with_strategy"], 4))
    s["learner"], s["pred_log"] = learner, predictions
    s["skips"] = [i - (i > removed_at) for i in s["skips"]]
    for key in ("block_mark", "intervened_at"):
        if s[key] > removed_at:
            s[key] -= 1
    s["retry"] = None
    s.pop("boost", None)
    s.pop("last", None)
    s["last_wrong"] = None
    return {"removed": True}


@app.post("/api/focus")
@session_locked
def focus(req: Focus):
    s = get_session(req.session_id)
    if req.focus != "all" and req.focus not in s["chapters"].values() and req.focus not in s["chapters"]:
        raise HTTPException(400, f"Unknown subject or chapter: {req.focus}")
    s["focus"] = req.focus
    return {"focus": req.focus, "pool": len(scope(s))}


@app.post("/api/ask")
def ask(req: Ask):
    s = get_session(req.session_id)
    reply = gemini(
        f"You are a concise tutor for a {audience(s['exam'])}. Answer the doubt in at most 6 sentences, "
        f"at their level, using $...$ LaTeX for maths. {EXPLAIN[s['explain_style']]}\nDoubt: {req.text}")
    return {"reply": reply or "Doubt answering needs GEMINI_API_KEY on the server. Quiz still works without it."}


@app.get("/api/dashboard")
@session_locked
def dashboard(session_id: str):
    s = get_session(session_id)
    L, exam = s["learner"], s["exam"]
    chapters = [{"chapter": c, "subject": subj, "readiness": round(L.readiness(c, exam), 3), "n": L.n.get(c, 0),
                 "flags": L.flags.get(c, 0), "banked": bool(items(s, c))} for c, subj in s["chapters"].items()]
    subjects = {}
    for c in chapters:
        subjects.setdefault(c["subject"], []).append(c["readiness"])
    hist = L.history
    seen = [c for c in chapters if c["n"]]
    return {
        "exam": LEVELS[exam], "focus": s["focus"],
        "subjects": [{"subject": k, "readiness": round(sum(v) / len(v), 3)} for k, v in subjects.items()],
        "chapters": chapters,
        "history": [{"i": i + 1, "chapter": h["chapter"], "b": h["b"], "theta": round(h["theta"], 3),
                     "correct": h["correct"], "secs": h["secs"], "confidence": h["confidence"],
                     "misconception": h["misconception"]} for i, h in enumerate(hist)],
        "weak": sorted(seen, key=lambda c: c["readiness"])[:5],
        "strong": sorted(seen, key=lambda c: -c["readiness"])[:3],
        "stats": {"answered": len(hist), "accuracy": round(sum(h["correct"] for h in hist) / len(hist), 3) if hist else 0,
                  "avg_secs": round(sum(h["secs"] for h in hist) / len(hist), 1) if hist else 0,
                  "misconceptions": sum(h["misconception"] for h in hist), "chapters_touched": len(seen)},
        "errors": [{"type": t, "label": lab, "fix": fix, "n": sum(h.get("diagnosis") == t for h in hist)}
                   for t, (lab, fix) in engine.ERROR_TYPES.items()],
        "calibration": engine.calibration(hist, exam),
        "marking": engine.MARKING.get(exam),
        "predicted": engine.predicted_score(L, list(s["chapters"]), exam),
        "predicted_log": s["pred_log"],
        "gains": engine.best_gains(L, list(s["chapters"]), exam),
        "wellbeing": dict(zip(("state", "reasons"), engine.wellbeing(hist, sum(i >= len(hist) - 6 for i in s["skips"])))),
    }


@app.get("/api/simulate")
def simulate(exam: str = "JEE"):
    return {"exam": LEVELS.get(exam, exam), "results": engine.compare(runs=20, exam=exam)}
'''

# ---------------- questions.json ----------------
FILES['questions.json'] = r'''[
{"id": "phy-kin-1", "chapter": "Kinematics", "b": -1.5, "q": "A car starts from rest and reaches $20\\,\\text{m/s}$ in $5\\,\\text{s}$ with uniform acceleration. Its acceleration is:", "options": ["$2\\,\\text{m/s}^2$", "$4\\,\\text{m/s}^2$", "$5\\,\\text{m/s}^2$", "$100\\,\\text{m/s}^2$"], "answer": 1, "hints": ["Acceleration is the rate of change of velocity.", "Use $a = \\dfrac{v - u}{t}$ with $u = 0$."], "explanation": "$a = (20 - 0)/5 = 4\\,\\text{m/s}^2$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-kin-2", "chapter": "Kinematics", "b": -0.5, "q": "A ball is thrown vertically upward at $20\\,\\text{m/s}$ ($g = 10\\,\\text{m/s}^2$). The maximum height reached is:", "options": ["$10\\,\\text{m}$", "$15\\,\\text{m}$", "$20\\,\\text{m}$", "$40\\,\\text{m}$"], "answer": 2, "hints": ["At the highest point the velocity is zero.", "Use $v^2 = u^2 - 2gh$."], "explanation": "$h = u^2/2g = 400/20 = 20\\,\\text{m}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-kin-3", "chapter": "Kinematics", "b": 0.5, "q": "A projectile is launched at $45^\\circ$ with speed $20\\,\\text{m/s}$ ($g = 10\\,\\text{m/s}^2$). Its horizontal range is:", "options": ["$20\\,\\text{m}$", "$40\\,\\text{m}$", "$40\\sqrt{2}\\,\\text{m}$", "$80\\,\\text{m}$"], "answer": 1, "hints": ["Range formula: $R = \\dfrac{u^2 \\sin 2\\theta}{g}$.", "$\\sin 90^\\circ = 1$."], "explanation": "$R = 400 \\times 1 / 10 = 40\\,\\text{m}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-kin-4", "chapter": "Kinematics", "b": 1.5, "q": "A particle moves along a line with $x = t^3 - 6t^2 + 9t$ (SI units). The distance travelled in the first $4\\,\\text{s}$ is:", "options": ["$4\\,\\text{m}$", "$8\\,\\text{m}$", "$12\\,\\text{m}$", "$16\\,\\text{m}$"], "answer": 2, "hints": ["Distance is not displacement: find where the particle turns around ($v = 0$).", "$v = 3(t-1)(t-3)$, so it turns at $t = 1$ and $t = 3$. Add up $|\\Delta x|$ for each leg."], "explanation": "$x(0)=0,\\ x(1)=4,\\ x(3)=0,\\ x(4)=4$. Distance $= 4 + 4 + 4 = 12\\,\\text{m}$ (displacement is only $4\\,\\text{m}$).", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-1", "chapter": "Laws of Motion", "b": -1.5, "q": "A net force of $10\\,\\text{N}$ acts on a $2\\,\\text{kg}$ body. Its acceleration is:", "options": ["$20\\,\\text{m/s}^2$", "$5\\,\\text{m/s}^2$", "$0.2\\,\\text{m/s}^2$", "$12\\,\\text{m/s}^2$"], "answer": 1, "hints": ["Newton's second law.", "$a = F/m$."], "explanation": "$a = 10/2 = 5\\,\\text{m/s}^2$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-2", "chapter": "Laws of Motion", "b": -0.5, "q": "A $60\\,\\text{kg}$ person stands in a lift accelerating upward at $2\\,\\text{m/s}^2$ ($g = 10\\,\\text{m/s}^2$). The normal force on the person is:", "options": ["$480\\,\\text{N}$", "$600\\,\\text{N}$", "$720\\,\\text{N}$", "$120\\,\\text{N}$"], "answer": 2, "hints": ["Draw the free-body diagram: $N$ up, $mg$ down, net force up.", "$N - mg = ma$."], "explanation": "$N = m(g + a) = 60 \\times 12 = 720\\,\\text{N}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-3", "chapter": "Laws of Motion", "b": 0.5, "q": "Masses $3\\,\\text{kg}$ and $2\\,\\text{kg}$ hang over a light frictionless pulley by a light string ($g = 10\\,\\text{m/s}^2$). The acceleration of the system is:", "options": ["$1\\,\\text{m/s}^2$", "$2\\,\\text{m/s}^2$", "$5\\,\\text{m/s}^2$", "$10\\,\\text{m/s}^2$"], "answer": 1, "hints": ["Treat both masses as one system: driving force is the weight difference.", "$a = \\dfrac{(m_1 - m_2)g}{m_1 + m_2}$."], "explanation": "$a = (3-2)\\times 10/5 = 2\\,\\text{m/s}^2$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-4", "chapter": "Laws of Motion", "b": 1.5, "q": "A $2\\,\\text{kg}$ block rests on a $4\\,\\text{kg}$ block on a frictionless floor. $\\mu = 0.4$ between the blocks. A horizontal force of $18\\,\\text{N}$ pulls the lower block ($g = 10\\,\\text{m/s}^2$). The friction force on the upper block is:", "options": ["$3\\,\\text{N}$", "$6\\,\\text{N}$", "$8\\,\\text{N}$", "$0\\,\\text{N}$"], "answer": 1, "hints": ["First assume they move together and find the common acceleration.", "Check: is the friction needed less than $\\mu m g$? If yes, static friction supplies exactly $ma$."], "explanation": "Together: $a = 18/6 = 3\\,\\text{m/s}^2$. Upper block can get up to $\\mu g = 4\\,\\text{m/s}^2$, so no slipping. Friction $= 2 \\times 3 = 6\\,\\text{N}$ (not $\\mu m g = 8\\,\\text{N}$).", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-1", "chapter": "Some Basic Concepts of Chemistry", "b": -1.5, "q": "Number of moles in $36\\,\\text{g}$ of water is:", "options": ["$1$", "$2$", "$18$", "$0.5$"], "answer": 1, "hints": ["Molar mass of $\\text{H}_2\\text{O}$ is $18\\,\\text{g/mol}$.", "$n = \\text{mass}/M$."], "explanation": "$n = 36/18 = 2$ mol.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-2", "chapter": "Some Basic Concepts of Chemistry", "b": -0.5, "q": "$4\\,\\text{g}$ of NaOH is dissolved to make $500\\,\\text{mL}$ of solution. Its molarity is:", "options": ["$0.1\\,\\text{M}$", "$0.2\\,\\text{M}$", "$0.4\\,\\text{M}$", "$0.8\\,\\text{M}$"], "answer": 1, "hints": ["Molar mass of NaOH is $40\\,\\text{g/mol}$.", "Molarity = moles / volume in litres."], "explanation": "$n = 0.1$ mol, $V = 0.5$ L, $M = 0.2\\,\\text{M}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-3", "chapter": "Some Basic Concepts of Chemistry", "b": 0.5, "q": "Volume of $\\text{O}_2$ at STP ($22.4\\,\\text{L/mol}$) needed to completely burn $1$ mol of $\\text{CH}_4$ is:", "options": ["$11.2\\,\\text{L}$", "$22.4\\,\\text{L}$", "$44.8\\,\\text{L}$", "$67.2\\,\\text{L}$"], "answer": 2, "hints": ["Write and balance the combustion equation.", "$\\text{CH}_4 + 2\\text{O}_2 \\rightarrow \\text{CO}_2 + 2\\text{H}_2\\text{O}$."], "explanation": "2 mol $\\text{O}_2$ per mol $\\text{CH}_4$: $2 \\times 22.4 = 44.8\\,\\text{L}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-4", "chapter": "Some Basic Concepts of Chemistry", "b": 1.5, "q": "$10\\,\\text{g}$ of $\\text{H}_2$ reacts with $64\\,\\text{g}$ of $\\text{O}_2$. Mass of water formed is:", "options": ["$36\\,\\text{g}$", "$72\\,\\text{g}$", "$74\\,\\text{g}$", "$90\\,\\text{g}$"], "answer": 1, "hints": ["Find the limiting reagent: convert both to moles.", "$2\\text{H}_2 + \\text{O}_2 \\rightarrow 2\\text{H}_2\\text{O}$. 2 mol $\\text{O}_2$ needs 4 mol $\\text{H}_2$."], "explanation": "$\\text{H}_2$: 5 mol, $\\text{O}_2$: 2 mol. $\\text{O}_2$ is limiting, gives 4 mol water $= 72\\,\\text{g}$. (74 g ignores the leftover 1 mol $\\text{H}_2$.)", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-1", "chapter": "Chemical Equilibrium", "b": -1.5, "q": "For $\\text{N}_2 + 3\\text{H}_2 \\rightleftharpoons 2\\text{NH}_3$, increasing the pressure shifts the equilibrium:", "options": ["Forward", "Backward", "No shift", "Depends on catalyst"], "answer": 0, "hints": ["Le Chatelier: the system opposes the change.", "Count gas moles on each side."], "explanation": "4 mol gas becomes 2 mol, so higher pressure favours the forward reaction.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-2", "chapter": "Chemical Equilibrium", "b": -0.5, "q": "For $\\text{N}_2 + 3\\text{H}_2 \\rightleftharpoons 2\\text{NH}_3$, the relation between $K_p$ and $K_c$ is:", "options": ["$K_p = K_c(RT)^{2}$", "$K_p = K_c(RT)^{-2}$", "$K_p = K_c(RT)^{-1}$", "$K_p = K_c$"], "answer": 1, "hints": ["$K_p = K_c(RT)^{\\Delta n_g}$.", "$\\Delta n_g$ = gas moles of products minus reactants."], "explanation": "$\\Delta n_g = 2 - 4 = -2$, so $K_p = K_c(RT)^{-2}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-3", "chapter": "Chemical Equilibrium", "b": 0.5, "q": "For $\\text{A} \\rightleftharpoons \\text{B}$, $K_c = 4$. Starting with $1$ mol A in $1$ L, the equilibrium $[\\text{B}]$ is:", "options": ["$0.2\\,\\text{M}$", "$0.25\\,\\text{M}$", "$0.75\\,\\text{M}$", "$0.8\\,\\text{M}$"], "answer": 3, "hints": ["Let $x$ mol of A convert. Then $[\\text{B}] = x$, $[\\text{A}] = 1 - x$.", "Solve $x/(1-x) = 4$."], "explanation": "$x = 4 - 4x \\Rightarrow x = 0.8\\,\\text{M}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-4", "chapter": "Chemical Equilibrium", "b": 1.5, "q": "$\\text{PCl}_5 \\rightleftharpoons \\text{PCl}_3 + \\text{Cl}_2$ has degree of dissociation $\\alpha$ at total pressure $P$. $K_p$ equals:", "options": ["$\\dfrac{\\alpha^2 P}{1-\\alpha^2}$", "$\\dfrac{\\alpha^2 P}{1-\\alpha}$", "$\\dfrac{\\alpha P}{1-\\alpha}$", "$\\dfrac{\\alpha^2 P}{1+\\alpha}$"], "answer": 0, "hints": ["Start with 1 mol $\\text{PCl}_5$: total moles at equilibrium $= 1 + \\alpha$.", "Partial pressure = mole fraction $\\times P$. Then use $(1-\\alpha)(1+\\alpha) = 1-\\alpha^2$."], "explanation": "$K_p = \\dfrac{(\\frac{\\alpha}{1+\\alpha}P)^2}{\\frac{1-\\alpha}{1+\\alpha}P} = \\dfrac{\\alpha^2 P}{1-\\alpha^2}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "mat-quad-1", "chapter": "Quadratic Equations", "b": -1.5, "q": "The roots of $x^2 - 5x + 6 = 0$ are:", "options": ["$1, 6$", "$2, 3$", "$-2, -3$", "$-1, 6$"], "answer": 1, "hints": ["Find two numbers with product 6 and sum 5.", "$(x-2)(x-3)$."], "explanation": "$x^2 - 5x + 6 = (x-2)(x-3)$, roots $2, 3$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-quad-2", "chapter": "Quadratic Equations", "b": -0.5, "q": "If $\\alpha, \\beta$ are roots of $x^2 - 3x + 1 = 0$, then $\\alpha^2 + \\beta^2$ is:", "options": ["$5$", "$7$", "$9$", "$11$"], "answer": 1, "hints": ["$\\alpha + \\beta = 3$, $\\alpha\\beta = 1$.", "$\\alpha^2 + \\beta^2 = (\\alpha+\\beta)^2 - 2\\alpha\\beta$."], "explanation": "$9 - 2 = 7$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-quad-3", "chapter": "Quadratic Equations", "b": 0.5, "q": "$x^2 + kx + 9 = 0$ has equal roots when $k$ is:", "options": ["$\\pm 3$", "$\\pm 6$", "$\\pm 9$", "$6$ only"], "answer": 1, "hints": ["Equal roots means discriminant $= 0$.", "$k^2 - 4 \\cdot 9 = 0$."], "explanation": "$k^2 = 36 \\Rightarrow k = \\pm 6$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-quad-4", "chapter": "Quadratic Equations", "b": 1.5, "q": "Both roots of $x^2 - 2ax + a^2 - 1 = 0$ lie in $(-2, 4)$ exactly when $a \\in$:", "options": ["$(-1, 3)$", "$(-2, 4)$", "$(-1, 4)$", "$(0, 3)$"], "answer": 0, "hints": ["Factor it: the expression is $(x-a)^2 - 1$.", "Roots are $a - 1$ and $a + 1$. Make both lie inside the interval."], "explanation": "$a - 1 > -2$ and $a + 1 < 4$ give $-1 < a < 3$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-1", "chapter": "Probability", "b": -1.5, "q": "Two fair coins are tossed. The probability of at least one head is:", "options": ["$\\frac{1}{4}$", "$\\frac{1}{2}$", "$\\frac{3}{4}$", "$1$"], "answer": 2, "hints": ["Use the complement: P(no head).", "P(TT) $= 1/4$."], "explanation": "$1 - \\frac{1}{4} = \\frac{3}{4}$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-2", "chapter": "Probability", "b": -0.5, "q": "A card is drawn from a standard deck of 52. The probability it is a king or a heart is:", "options": ["$\\frac{17}{52}$", "$\\frac{4}{13}$", "$\\frac{1}{4}$", "$\\frac{1}{52}$"], "answer": 1, "hints": ["$P(A \\cup B) = P(A) + P(B) - P(A \\cap B)$.", "The king of hearts is counted twice."], "explanation": "$\\frac{4 + 13 - 1}{52} = \\frac{16}{52} = \\frac{4}{13}$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-3", "chapter": "Probability", "b": 0.5, "q": "Two fair dice are thrown. The probability that the sum is $7$ is:", "options": ["$\\frac{1}{6}$", "$\\frac{5}{36}$", "$\\frac{7}{36}$", "$\\frac{1}{12}$"], "answer": 0, "hints": ["There are 36 equally likely outcomes.", "List pairs: (1,6), (2,5), ..."], "explanation": "6 favourable pairs: $\\frac{6}{36} = \\frac{1}{6}$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-4", "chapter": "Probability", "b": 1.5, "q": "Bag A has 3 red, 2 black balls. Bag B has 2 red, 3 black. A bag is chosen at random and a red ball is drawn. The probability it came from bag A is:", "options": ["$\\frac{3}{10}$", "$\\frac{1}{2}$", "$\\frac{2}{5}$", "$\\frac{3}{5}$"], "answer": 3, "hints": ["This is Bayes' theorem: P(A | red).", "$P(A|R) = \\dfrac{P(A)P(R|A)}{P(A)P(R|A) + P(B)P(R|B)}$."], "explanation": "$\\dfrac{\\frac12 \\cdot \\frac35}{\\frac12 \\cdot \\frac35 + \\frac12 \\cdot \\frac25} = \\frac{3}{5}$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "bio-cell-1", "chapter": "Cell: The Unit of Life", "b": -1.5, "q": "Which organelle is called the powerhouse of the cell?", "options": ["Ribosome", "Mitochondrion", "Golgi apparatus", "Lysosome"], "answer": 1, "hints": ["Think of where ATP is made by aerobic respiration.", "It has its own DNA and a double membrane."], "explanation": "Mitochondria produce most of the cell's ATP.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-cell-2", "chapter": "Cell: The Unit of Life", "b": -0.5, "q": "Ribosomes of prokaryotic cells are of type:", "options": ["80S", "70S", "60S", "50S"], "answer": 1, "hints": ["Eukaryotic cytoplasmic ribosomes are 80S.", "Prokaryotic ribosomes are smaller: 50S + 30S subunits."], "explanation": "Prokaryotes have 70S ribosomes (50S + 30S; S values do not add).", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-cell-3", "chapter": "Cell: The Unit of Life", "b": 0.3, "q": "Which of these is NOT bounded by a membrane?", "options": ["Lysosome", "Ribosome", "Vacuole", "Golgi apparatus"], "answer": 1, "hints": ["Prokaryotes lack membrane-bound organelles but still have this one.", "It is made of rRNA and protein only."], "explanation": "Ribosomes have no membrane, which is why prokaryotes have them too.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-cell-4", "chapter": "Cell: The Unit of Life", "b": 1.2, "q": "The arrangement of microtubules in a centriole is:", "options": ["9 + 2, doublets", "9 + 0, triplets", "9 + 0, doublets", "9 + 2, triplets"], "answer": 1, "hints": ["Cilia and flagella axonemes are 9 + 2 with doublets. Centrioles differ.", "Centrioles have a cartwheel structure with no central pair."], "explanation": "Centrioles (and basal bodies): 9 peripheral triplets, no central microtubules (9 + 0).", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-1", "chapter": "Principles of Inheritance and Variation", "b": -1.5, "q": "The phenotypic ratio in the F2 generation of Mendel's monohybrid cross is:", "options": ["1 : 2 : 1", "3 : 1", "9 : 3 : 3 : 1", "1 : 1"], "answer": 1, "hints": ["Phenotype, not genotype.", "Dominant trait appears in 3 of every 4."], "explanation": "Phenotype 3 : 1 (genotype 1 : 2 : 1).", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-2", "chapter": "Principles of Inheritance and Variation", "b": -0.5, "q": "The phenotypic ratio in the F2 generation of a dihybrid cross is:", "options": ["3 : 1", "1 : 2 : 1", "9 : 3 : 3 : 1", "1 : 1 : 1 : 1"], "answer": 2, "hints": ["Two genes assorting independently.", "Multiply $(3:1) \\times (3:1)$."], "explanation": "Independent assortment gives 9 : 3 : 3 : 1.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-3", "chapter": "Principles of Inheritance and Variation", "b": 0.3, "q": "In snapdragon, a red-flowered plant crossed with a white-flowered plant gives pink-flowered F1. This is:", "options": ["Codominance", "Incomplete dominance", "Epistasis", "Pleiotropy"], "answer": 1, "hints": ["Neither allele fully masks the other.", "In codominance both traits show separately (like AB blood group); here they blend."], "explanation": "Pink is intermediate: incomplete dominance. F2 gives 1 red : 2 pink : 1 white.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-4", "chapter": "Principles of Inheritance and Variation", "b": 1.2, "q": "A colour-blind man marries a woman with normal vision whose father was colour-blind. The probability that their son is colour-blind is:", "options": ["$0$", "$\\frac{1}{4}$", "$\\frac{1}{2}$", "$1$"], "answer": 2, "hints": ["Colour blindness is X-linked recessive. What is the woman's genotype?", "A son gets his X only from his mother."], "explanation": "The woman is a carrier ($X^cX$). A son gets $X^c$ with probability $\\frac12$; the father's X does not go to sons.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "phy-kin-5", "chapter": "Kinematics", "b": -1.0, "q": "A car covers the first half of a distance at $20\\,\\text{m/s}$ and the second half at $30\\,\\text{m/s}$. Its average speed for the whole journey is:", "options": ["$25\\,\\text{m/s}$", "$24\\,\\text{m/s}$", "$50\\,\\text{m/s}$", "$10\\,\\text{m/s}$"], "answer": 1, "hints": ["Average speed = total distance / total time, not the mean of the two speeds.", "For equal distances, $v_{avg} = \\dfrac{2v_1v_2}{v_1 + v_2}$."], "explanation": "$v_{avg} = \\dfrac{2 \\times 20 \\times 30}{50} = 24\\,\\text{m/s}$. 25 m/s is the trap (simple mean).", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-kin-6", "chapter": "Kinematics", "b": 0.0, "q": "A stone is dropped from rest ($g = 10\\,\\text{m/s}^2$). The distance it falls during the 3rd second is:", "options": ["$45\\,\\text{m}$", "$25\\,\\text{m}$", "$15\\,\\text{m}$", "$30\\,\\text{m}$"], "answer": 1, "hints": ["Distance in the $n$th second is not the same as distance in $n$ seconds.", "$s_n = u + \\dfrac{g}{2}(2n - 1)$."], "explanation": "$s_3 = 0 + 5 \\times 5 = 25\\,\\text{m}$ (45 m is the total after 3 s).", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-kin-7", "chapter": "Kinematics", "b": -0.5, "q": "A car's speed increases uniformly from $10\\,\\text{m/s}$ to $30\\,\\text{m/s}$ over a distance of $100\\,\\text{m}$. Its acceleration is:", "options": ["$2\\,\\text{m/s}^2$", "$4\\,\\text{m/s}^2$", "$8\\,\\text{m/s}^2$", "$0.2\\,\\text{m/s}^2$"], "answer": 1, "hints": ["Time is not given, so use the equation without $t$.", "$v^2 = u^2 + 2as$."], "explanation": "$a = \\dfrac{900 - 100}{2 \\times 100} = 4\\,\\text{m/s}^2$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-kin-8", "chapter": "Kinematics", "b": 0.5, "q": "A boat moves at $5\\,\\text{m/s}$ relative to water and heads straight across a $400\\,\\text{m}$ wide river flowing at $3\\,\\text{m/s}$. How far downstream does it land?", "options": ["$160\\,\\text{m}$", "$240\\,\\text{m}$", "$300\\,\\text{m}$", "$400\\,\\text{m}$"], "answer": 1, "hints": ["The crossing time depends only on the velocity component across the river.", "$t = 400/5$, then drift $= (\\text{river speed}) \\times t$."], "explanation": "$t = 80\\,\\text{s}$, drift $= 3 \\times 80 = 240\\,\\text{m}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-kin-9", "chapter": "Kinematics", "b": 0.9, "q": "Two projectiles are launched with the same speed at $30^\\circ$ and $60^\\circ$ to the horizontal. The ratio of their maximum heights $H_{30}/H_{60}$ is:", "options": ["$\\frac{1}{3}$", "$3$", "$1$", "$\\frac{1}{9}$"], "answer": 0, "hints": ["$H = \\dfrac{u^2 \\sin^2\\theta}{2g}$.", "Compare $\\sin^2 30^\\circ$ with $\\sin^2 60^\\circ$."], "explanation": "$\\dfrac{\\sin^2 30^\\circ}{\\sin^2 60^\\circ} = \\dfrac{1/4}{3/4} = \\dfrac{1}{3}$. (Their ranges are equal.)", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-kin-10", "chapter": "Kinematics", "b": 1.6, "q": "A particle starts from rest at the origin with acceleration $a = 6t - 4$ (SI units). The distance it travels in the first $2\\,\\text{s}$ is:", "options": ["$\\frac{64}{27}\\,\\text{m}$", "$0\\,\\text{m}$", "$\\frac{32}{27}\\,\\text{m}$", "$\\frac{8}{3}\\,\\text{m}$"], "answer": 0, "hints": ["Integrate twice: $v = 3t^2 - 4t$, $x = t^3 - 2t^2$. Does $v$ change sign?", "$v = 0$ at $t = \\frac{4}{3}\\,\\text{s}$. Add the distances before and after."], "explanation": "$x(\\frac43) = -\\frac{32}{27}$, $x(2) = 0$. Distance $= \\frac{32}{27} + \\frac{32}{27} = \\frac{64}{27}\\,\\text{m}$; displacement is 0 (the trap).", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-5", "chapter": "Laws of Motion", "b": -1.0, "q": "A $0.5\\,\\text{kg}$ ball hits a wall at $10\\,\\text{m/s}$ and rebounds with the same speed. The magnitude of the impulse on the ball is:", "options": ["$0\\,\\text{N s}$", "$5\\,\\text{N s}$", "$10\\,\\text{N s}$", "$20\\,\\text{N s}$"], "answer": 2, "hints": ["Impulse = change in momentum. Direction matters.", "$\\Delta p = m(v - (-v)) = 2mv$."], "explanation": "$J = 2 \\times 0.5 \\times 10 = 10\\,\\text{N s}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-6", "chapter": "Laws of Motion", "b": 0.0, "q": "A $5\\,\\text{kg}$ block on a rough floor ($\\mu_k = 0.2$) is pulled by a horizontal force of $20\\,\\text{N}$ ($g = 10\\,\\text{m/s}^2$). Its acceleration is:", "options": ["$4\\,\\text{m/s}^2$", "$2\\,\\text{m/s}^2$", "$0\\,\\text{m/s}^2$", "$6\\,\\text{m/s}^2$"], "answer": 1, "hints": ["Kinetic friction $= \\mu_k mg$.", "$a = (F - \\mu_k mg)/m$."], "explanation": "Friction $= 0.2 \\times 50 = 10\\,\\text{N}$, $a = (20 - 10)/5 = 2\\,\\text{m/s}^2$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-7", "chapter": "Laws of Motion", "b": 0.6, "q": "A block is released on a rough incline of $37^\\circ$ with $\\mu_k = 0.5$ ($g = 10\\,\\text{m/s}^2$, $\\sin 37^\\circ = 0.6$). Its acceleration down the incline is:", "options": ["$6\\,\\text{m/s}^2$", "$2\\,\\text{m/s}^2$", "$4\\,\\text{m/s}^2$", "$0\\,\\text{m/s}^2$"], "answer": 1, "hints": ["Since $\\tan 37^\\circ = 0.75 > 0.5$, the block does slide.", "$a = g(\\sin\\theta - \\mu\\cos\\theta)$."], "explanation": "$a = 10(0.6 - 0.5 \\times 0.8) = 2\\,\\text{m/s}^2$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-8", "chapter": "Laws of Motion", "b": 0.9, "q": "The maximum speed at which a car can take an unbanked turn of radius $40\\,\\text{m}$ with $\\mu_s = 0.4$ ($g = 10\\,\\text{m/s}^2$) is:", "options": ["$4\\sqrt{10}\\,\\text{m/s}$", "$16\\,\\text{m/s}$", "$8\\,\\text{m/s}$", "$2\\sqrt{10}\\,\\text{m/s}$"], "answer": 0, "hints": ["Friction supplies the centripetal force.", "$\\mu mg = mv^2/r \\Rightarrow v = \\sqrt{\\mu g r}$."], "explanation": "$v = \\sqrt{0.4 \\times 10 \\times 40} = \\sqrt{160} = 4\\sqrt{10}\\,\\text{m/s}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-9", "chapter": "Laws of Motion", "b": 1.3, "q": "A $4\\,\\text{kg}$ block on a table ($\\mu_k = 0.25$) is tied by a string over a light frictionless pulley to a hanging $2\\,\\text{kg}$ block ($g = 10\\,\\text{m/s}^2$). The tension in the string is:", "options": ["$\\frac{50}{3}\\,\\text{N}$", "$20\\,\\text{N}$", "$10\\,\\text{N}$", "$\\frac{40}{3}\\,\\text{N}$"], "answer": 0, "hints": ["System: driving force $= 20\\,\\text{N}$, friction $= 0.25 \\times 40 = 10\\,\\text{N}$, total mass $6\\,\\text{kg}$.", "Then for the hanging block: $mg - T = ma$."], "explanation": "$a = 10/6 = \\frac53\\,\\text{m/s}^2$, $T = 2(10 - \\frac53) = \\frac{50}{3}\\,\\text{N}$. (20 N assumes no acceleration.)", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "phy-lom-10", "chapter": "Laws of Motion", "b": 1.5, "q": "A block rests on a smooth wedge of angle $37^\\circ$. With what horizontal acceleration must the wedge move so the block stays at rest relative to it? ($g = 10\\,\\text{m/s}^2$, $\\tan 37^\\circ = 0.75$)", "options": ["$7.5\\,\\text{m/s}^2$", "$6\\,\\text{m/s}^2$", "$8\\,\\text{m/s}^2$", "$\\frac{40}{3}\\,\\text{m/s}^2$"], "answer": 0, "hints": ["Work in the wedge frame: a pseudo force $ma$ acts on the block.", "Along the incline: $ma\\cos\\theta = mg\\sin\\theta$."], "explanation": "$a = g\\tan\\theta = 10 \\times 0.75 = 7.5\\,\\text{m/s}^2$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-5", "chapter": "Some Basic Concepts of Chemistry", "b": -1.0, "q": "The number of molecules in $4.4\\,\\text{g}$ of $\\text{CO}_2$ is ($N_A = 6.022 \\times 10^{23}$):", "options": ["$6.022 \\times 10^{22}$", "$6.022 \\times 10^{23}$", "$1.2 \\times 10^{23}$", "$2.6 \\times 10^{24}$"], "answer": 0, "hints": ["Molar mass of $\\text{CO}_2$ is $44\\,\\text{g/mol}$.", "Molecules $= n \\times N_A$."], "explanation": "$n = 0.1$ mol, so $0.1 \\times 6.022 \\times 10^{23} = 6.022 \\times 10^{22}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-6", "chapter": "Some Basic Concepts of Chemistry", "b": -0.3, "q": "The mass percentage of oxygen in $\\text{H}_2\\text{SO}_4$ is (H = 1, S = 32, O = 16):", "options": ["$65.3\\%$", "$32.7\\%$", "$2.0\\%$", "$16.3\\%$"], "answer": 0, "hints": ["Molar mass of $\\text{H}_2\\text{SO}_4 = 98\\,\\text{g/mol}$.", "Oxygen contributes $4 \\times 16 = 64\\,\\text{g}$."], "explanation": "$\\frac{64}{98} \\times 100 = 65.3\\%$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-7", "chapter": "Some Basic Concepts of Chemistry", "b": 0.3, "q": "A compound contains 40% C, 6.7% H and 53.3% O by mass. Its empirical formula is:", "options": ["$\\text{CHO}$", "$\\text{CH}_2\\text{O}$", "$\\text{C}_2\\text{H}_4\\text{O}$", "$\\text{CH}_2\\text{O}_2$"], "answer": 1, "hints": ["Divide each percentage by the atomic mass.", "Then divide by the smallest value to get the ratio."], "explanation": "C: $40/12 = 3.33$, H: $6.7/1 = 6.7$, O: $53.3/16 = 3.33$. Ratio $1 : 2 : 1$, so $\\text{CH}_2\\text{O}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-8", "chapter": "Some Basic Concepts of Chemistry", "b": 0.6, "q": "$100\\,\\text{mL}$ of $0.5\\,\\text{M}$ HCl is mixed with $200\\,\\text{mL}$ of $0.2\\,\\text{M}$ HCl. The molarity of the mixture is:", "options": ["$0.35\\,\\text{M}$", "$0.3\\,\\text{M}$", "$0.7\\,\\text{M}$", "$0.233\\,\\text{M}$"], "answer": 1, "hints": ["Add the millimoles, not the molarities.", "$M = \\dfrac{M_1V_1 + M_2V_2}{V_1 + V_2}$."], "explanation": "$\\frac{50 + 40}{300} = 0.3\\,\\text{M}$. 0.35 M is the simple average trap.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-9", "chapter": "Some Basic Concepts of Chemistry", "b": 1.0, "q": "$0.2\\,\\text{g}$ of a hydrocarbon on complete combustion gives $0.66\\,\\text{g}$ of $\\text{CO}_2$. The percentage of carbon in it is:", "options": ["$82\\%$", "$75\\%$", "$90\\%$", "$66\\%$"], "answer": 2, "hints": ["All the carbon ends up in $\\text{CO}_2$.", "Mass of C $= \\frac{12}{44} \\times 0.66$."], "explanation": "C $= 0.18\\,\\text{g}$; $\\frac{0.18}{0.2} \\times 100 = 90\\%$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-mole-10", "chapter": "Some Basic Concepts of Chemistry", "b": 1.6, "q": "The volume strength of a $1.5\\,\\text{M}$ $\\text{H}_2\\text{O}_2$ solution is:", "options": ["$11.2$", "$8.4$", "$33.6$", "$16.8$"], "answer": 3, "hints": ["$2\\text{H}_2\\text{O}_2 \\rightarrow 2\\text{H}_2\\text{O} + \\text{O}_2$: 2 mol give 22.4 L $\\text{O}_2$ at STP.", "So volume strength $= 11.2 \\times$ molarity."], "explanation": "$11.2 \\times 1.5 = 16.8$ volume.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-5", "chapter": "Chemical Equilibrium", "b": -1.0, "q": "Adding a catalyst to a system already at equilibrium:", "options": ["Shifts it forward", "Shifts it backward", "Leaves the equilibrium composition unchanged", "Increases $K$"], "answer": 2, "hints": ["A catalyst speeds up the forward and backward reactions equally.", "It changes how fast equilibrium is reached, not where it lies."], "explanation": "A catalyst lowers activation energy for both directions equally, so $K$ and the composition stay the same.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-6", "chapter": "Chemical Equilibrium", "b": -0.2, "q": "For $\\text{A} + \\text{B} \\rightleftharpoons \\text{C}$, $K = 4$. The equilibrium constant for $\\text{C} \\rightleftharpoons \\text{A} + \\text{B}$ is:", "options": ["$4$", "$0.25$", "$-4$", "$2$"], "answer": 1, "hints": ["Reversing a reaction inverts its expression.", "$K' = 1/K$."], "explanation": "$K' = 1/4 = 0.25$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-7", "chapter": "Chemical Equilibrium", "b": 0.3, "q": "For $\\text{A} + \\text{B} \\rightleftharpoons \\text{C}$, $K = 4$. The equilibrium constant for $2\\text{A} + 2\\text{B} \\rightleftharpoons 2\\text{C}$ is:", "options": ["$8$", "$2$", "$16$", "$4$"], "answer": 2, "hints": ["Multiplying an equation by $n$ raises $K$ to the power $n$.", "$K' = K^2$."], "explanation": "$K' = 4^2 = 16$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-8", "chapter": "Chemical Equilibrium", "b": 0.8, "q": "For $\\text{PCl}_5(g) \\rightleftharpoons \\text{PCl}_3(g) + \\text{Cl}_2(g)$ at $300\\,\\text{K}$, $K_p = 24.6\\,\\text{atm}$. Taking $R = 0.082\\,\\text{L atm K}^{-1}\\text{mol}^{-1}$, $K_c$ (in mol/L) is:", "options": ["$1$", "$24.6$", "$0.041$", "$605$"], "answer": 0, "hints": ["$K_p = K_c(RT)^{\\Delta n_g}$. Here $\\Delta n_g = 1$.", "$RT = 0.082 \\times 300 = 24.6$."], "explanation": "$K_c = 24.6 / 24.6 = 1\\,\\text{mol/L}$.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-9", "chapter": "Chemical Equilibrium", "b": 1.2, "q": "For $\\text{N}_2\\text{O}_4 \\rightleftharpoons 2\\text{NO}_2$, $K_c = 0.2$. At some instant $[\\text{N}_2\\text{O}_4] = 0.5\\,\\text{M}$ and $[\\text{NO}_2] = 0.5\\,\\text{M}$. The reaction will:", "options": ["Proceed forward", "Proceed backward", "Already be at equilibrium", "Stop"], "answer": 1, "hints": ["Compute the reaction quotient $Q$ and compare it with $K$.", "$Q = [\\text{NO}_2]^2/[\\text{N}_2\\text{O}_4]$."], "explanation": "$Q = 0.25/0.5 = 0.5 > K = 0.2$, so too much product: the reaction goes backward.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "che-eq-10", "chapter": "Chemical Equilibrium", "b": 1.7, "q": "$1$ mol $\\text{H}_2$ and $1$ mol $\\text{I}_2$ are heated in a $1\\,\\text{L}$ flask. For $\\text{H}_2 + \\text{I}_2 \\rightleftharpoons 2\\text{HI}$, $K_c = 64$. The equilibrium concentration of HI is:", "options": ["$0.8\\,\\text{M}$", "$1.8\\,\\text{M}$", "$1.6\\,\\text{M}$", "$0.4\\,\\text{M}$"], "answer": 2, "hints": ["Let $x$ mol of $\\text{H}_2$ react: $[\\text{HI}] = 2x$, $[\\text{H}_2] = [\\text{I}_2] = 1 - x$.", "Take the square root: $\\dfrac{2x}{1-x} = 8$."], "explanation": "$2x = 8 - 8x \\Rightarrow x = 0.8$, so $[\\text{HI}] = 1.6\\,\\text{M}$. 0.8 M forgets the factor 2.", "source": "Original · JEE Main / NEET past-paper pattern"},
{"id": "mat-quad-5", "chapter": "Quadratic Equations", "b": -1.0, "q": "The sum of the roots of $2x^2 - 8x + 3 = 0$ is:", "options": ["$-4$", "$4$", "$\\frac{3}{2}$", "$8$"], "answer": 1, "hints": ["Sum of roots $= -b/a$.", "Here $a = 2$, $b = -8$."], "explanation": "$-(-8)/2 = 4$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-quad-6", "chapter": "Quadratic Equations", "b": -0.2, "q": "The quadratic equation whose roots are $2$ and $-3$ is:", "options": ["$x^2 - x - 6 = 0$", "$x^2 + x - 6 = 0$", "$x^2 + x + 6 = 0$", "$x^2 - 5x + 6 = 0$"], "answer": 1, "hints": ["$x^2 - (\\text{sum})x + (\\text{product}) = 0$.", "Sum $= -1$, product $= -6$."], "explanation": "$x^2 - (-1)x + (-6) = x^2 + x - 6 = 0$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-quad-7", "chapter": "Quadratic Equations", "b": 0.4, "q": "If $\\alpha, \\beta$ are the roots of $x^2 - 5x + 6 = 0$, then $\\frac{1}{\\alpha} + \\frac{1}{\\beta}$ is:", "options": ["$\\frac{5}{6}$", "$\\frac{6}{5}$", "$5$", "$\\frac{1}{6}$"], "answer": 0, "hints": ["Combine: $\\dfrac{1}{\\alpha} + \\dfrac{1}{\\beta} = \\dfrac{\\alpha + \\beta}{\\alpha\\beta}$.", "$\\alpha + \\beta = 5$, $\\alpha\\beta = 6$."], "explanation": "$\\frac{5}{6}$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-quad-8", "chapter": "Quadratic Equations", "b": 0.8, "q": "The minimum value of $x^2 - 6x + 13$ for real $x$ is:", "options": ["$13$", "$9$", "$4$", "$3$"], "answer": 2, "hints": ["Complete the square.", "$x^2 - 6x + 13 = (x - 3)^2 + 4$."], "explanation": "Minimum $4$ at $x = 3$ (3 is where it occurs, not the value).", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-quad-9", "chapter": "Quadratic Equations", "b": 1.2, "q": "The number of integer values of $k$ for which $x^2 - 2kx + 4 = 0$ has no real roots is:", "options": ["$2$", "$3$", "$4$", "$5$"], "answer": 1, "hints": ["No real roots means discriminant $< 0$.", "$4k^2 - 16 < 0 \\Rightarrow -2 < k < 2$."], "explanation": "$k \\in \\{-1, 0, 1\\}$: 3 values. (Endpoints $\\pm 2$ give equal roots, so they are excluded.)", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-quad-10", "chapter": "Quadratic Equations", "b": 1.8, "q": "If $\\alpha, \\beta$ are roots of $x^2 - (a - 2)x - a + 1 = 0$, the value of $a$ for which $\\alpha^2 + \\beta^2$ is least is:", "options": ["$0$", "$1$", "$2$", "$3$"], "answer": 1, "hints": ["Write $\\alpha^2 + \\beta^2 = (\\alpha + \\beta)^2 - 2\\alpha\\beta$ in terms of $a$.", "You get $a^2 - 2a + 2 = (a - 1)^2 + 1$."], "explanation": "$\\alpha^2 + \\beta^2 = (a-2)^2 + 2(a-1) = (a-1)^2 + 1$, least at $a = 1$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-5", "chapter": "Probability", "b": -1.0, "q": "A fair die is thrown once. The probability of getting a prime number is:", "options": ["$\\frac{1}{3}$", "$\\frac{1}{2}$", "$\\frac{2}{3}$", "$\\frac{1}{6}$"], "answer": 1, "hints": ["List the primes from 1 to 6.", "1 is not prime."], "explanation": "Primes: 2, 3, 5, so $\\frac{3}{6} = \\frac{1}{2}$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-6", "chapter": "Probability", "b": -0.2, "q": "Two cards are drawn one after another without replacement from a deck of 52. The probability that both are aces is:", "options": ["$\\frac{1}{221}$", "$\\frac{1}{169}$", "$\\frac{2}{13}$", "$\\frac{1}{26}$"], "answer": 0, "hints": ["Multiply: P(first ace) $\\times$ P(second ace | first ace).", "$\\frac{4}{52} \\times \\frac{3}{51}$."], "explanation": "$\\frac{4}{52} \\times \\frac{3}{51} = \\frac{1}{221}$. $\\frac{1}{169}$ is the with-replacement trap.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-7", "chapter": "Probability", "b": 0.4, "q": "$A$ and $B$ are independent events with $P(A) = 0.4$ and $P(B) = 0.5$. Then $P(A \\cup B)$ is:", "options": ["$0.9$", "$0.7$", "$0.2$", "$0.5$"], "answer": 1, "hints": ["For independent events $P(A \\cap B) = P(A)P(B)$.", "$P(A \\cup B) = P(A) + P(B) - P(A \\cap B)$."], "explanation": "$0.4 + 0.5 - 0.2 = 0.7$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-8", "chapter": "Probability", "b": 0.8, "q": "Three fair coins are tossed. The probability of getting exactly two heads is:", "options": ["$\\frac{1}{2}$", "$\\frac{1}{4}$", "$\\frac{3}{8}$", "$\\frac{1}{8}$"], "answer": 2, "hints": ["There are $2^3 = 8$ equally likely outcomes.", "Count arrangements of HHT: $\\binom{3}{2}$."], "explanation": "$\\frac{3}{8}$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-9", "chapter": "Probability", "b": 1.2, "q": "A fair die is thrown 4 times. The probability of getting at least one six is:", "options": ["$\\frac{625}{1296}$", "$\\frac{2}{3}$", "$\\frac{1}{6}$", "$\\frac{671}{1296}$"], "answer": 3, "hints": ["Use the complement: P(no six in 4 throws).", "$1 - \\left(\\frac{5}{6}\\right)^4$."], "explanation": "$1 - \\frac{625}{1296} = \\frac{671}{1296}$.", "source": "Original · JEE Main past-paper pattern"},
{"id": "mat-prob-10", "chapter": "Probability", "b": 1.8, "q": "A family has two children. Given that at least one of them is a boy, the probability that both are boys is:", "options": ["$\\frac{1}{2}$", "$\\frac{1}{3}$", "$\\frac{1}{4}$", "$\\frac{2}{3}$"], "answer": 1, "hints": ["Sample space: BB, BG, GB, GG. Which ones have at least one boy?", "Conditional probability: restrict to those outcomes."], "explanation": "Given set {BB, BG, GB}; only BB fits, so $\\frac{1}{3}$. ($\\frac12$ is the common trap.)", "source": "Original · JEE Main past-paper pattern"},
{"id": "bio-cell-5", "chapter": "Cell: The Unit of Life", "b": -1.0, "q": "The cell theory was proposed by:", "options": ["Robert Hooke", "Schleiden and Schwann", "Rudolf Virchow", "Leeuwenhoek"], "answer": 1, "hints": ["One was a botanist, the other a zoologist (1838–39).", "Virchow later added that cells arise from pre-existing cells."], "explanation": "Schleiden and Schwann. Virchow (1855) modified it (cells arise from pre-existing cells).", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-cell-6", "chapter": "Cell: The Unit of Life", "b": -0.4, "q": "Which organelle contains hydrolytic enzymes and digests worn-out cell parts?", "options": ["Golgi apparatus", "Lysosome", "Peroxisome", "Vacuole"], "answer": 1, "hints": ["These enzymes work best at acidic pH.", "It is formed from Golgi vesicles."], "explanation": "Lysosomes contain acid hydrolases (lipases, proteases, carbohydrases).", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-cell-7", "chapter": "Cell: The Unit of Life", "b": 0.0, "q": "Which of these is absent in a prokaryotic cell?", "options": ["Ribosomes", "Cell membrane", "Nuclear membrane", "DNA"], "answer": 2, "hints": ["Prokaryotes have genetic material, but it is not enclosed.", "Their DNA lies in a region called the nucleoid."], "explanation": "Prokaryotes lack a nuclear envelope; DNA is in the nucleoid.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-cell-8", "chapter": "Cell: The Unit of Life", "b": 0.5, "q": "The smooth endoplasmic reticulum is the major site for:", "options": ["Protein synthesis", "Lipid synthesis", "ATP synthesis", "DNA replication"], "answer": 1, "hints": ["Rough ER has ribosomes; smooth ER does not.", "In animal cells it makes steroidal hormones."], "explanation": "SER synthesises lipids (including steroid hormones). Protein synthesis happens on RER ribosomes.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-cell-9", "chapter": "Cell: The Unit of Life", "b": 0.9, "q": "Mesosomes in bacteria are formed by:", "options": ["Infoldings of the plasma membrane", "Extensions of the cell wall", "Fragments of the nucleoid", "Clusters of ribosomes"], "answer": 0, "hints": ["They are membranous structures.", "They help in respiration, secretion and cell wall formation."], "explanation": "Mesosomes are extensions of the plasma membrane into the cell (vesicles, tubules, lamellae).", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-cell-10", "chapter": "Cell: The Unit of Life", "b": 1.4, "q": "The Golgi apparatus is an important site for the formation of:", "options": ["Glycoproteins and glycolipids", "Ribosomal RNA", "ATP", "Microtubules"], "answer": 0, "hints": ["Proteins from the ER are modified in its cisternae.", "It adds carbohydrate groups."], "explanation": "Golgi is the main site of glycoprotein and glycolipid formation. rRNA is made in the nucleolus.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-5", "chapter": "Principles of Inheritance and Variation", "b": -1.0, "q": "A test cross is a cross between an F1 hybrid and:", "options": ["A homozygous dominant parent", "A homozygous recessive parent", "Another F1 hybrid", "An F2 plant"], "answer": 1, "hints": ["It reveals the unknown genotype of a dominant-looking plant.", "The tester shows only the recessive trait."], "explanation": "Crossing with the homozygous recessive parent exposes the F1 genotype.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-6", "chapter": "Principles of Inheritance and Variation", "b": -0.3, "q": "The phenotypic ratio of a monohybrid test cross is:", "options": ["3 : 1", "1 : 1", "1 : 2 : 1", "9 : 3 : 3 : 1"], "answer": 1, "hints": ["$Tt \\times tt$.", "Half the offspring get $T$."], "explanation": "$Tt \\times tt \\rightarrow \\frac12\\,Tt : \\frac12\\,tt$ = 1 : 1.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-7", "chapter": "Principles of Inheritance and Variation", "b": 0.3, "q": "The ABO blood group system in humans is an example of:", "options": ["Incomplete dominance only", "Codominance and multiple alleles", "Epistasis", "Polygenic inheritance"], "answer": 1, "hints": ["How many alleles of gene $I$ exist in the population?", "What happens in a person with blood group AB?"], "explanation": "Three alleles ($I^A, I^B, i$) = multiple alleles; $I^A$ and $I^B$ both express in AB = codominance.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-8", "chapter": "Principles of Inheritance and Variation", "b": 0.7, "q": "The number of different genotypes in the F2 generation of a dihybrid cross is:", "options": ["$4$", "$9$", "$16$", "$3$"], "answer": 1, "hints": ["Each gene alone gives 3 genotypes in F2.", "Two independent genes: multiply."], "explanation": "$3 \\times 3 = 9$ genotypes (4 phenotypes, 16 combinations).", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-9", "chapter": "Principles of Inheritance and Variation", "b": 1.0, "q": "A father with blood group A (genotype $I^Ai$) and a mother with blood group B (genotype $I^Bi$) have a child. The probability that the child has blood group O is:", "options": ["$0$", "$\\frac{1}{4}$", "$\\frac{1}{2}$", "$\\frac{3}{4}$"], "answer": 1, "hints": ["Blood group O needs genotype $ii$.", "Each parent passes $i$ with probability $\\frac12$."], "explanation": "$\\frac12 \\times \\frac12 = \\frac14$.", "source": "Original · NCERT / NEET past-paper pattern"},
{"id": "bio-inh-10", "chapter": "Principles of Inheritance and Variation", "b": 1.5, "q": "A woman who is a carrier of haemophilia marries a normal man. What fraction of all their children is expected to have haemophilia?", "options": ["$\\frac{1}{4}$", "$\\frac{1}{2}$", "$0$", "$1$"], "answer": 0, "hints": ["Haemophilia is X-linked recessive. Daughters get the father's normal X.", "Only sons can be affected: half the sons."], "explanation": "Sons: $\\frac12$ affected; daughters: none affected. Over all children $= \\frac12 \\times \\frac12 = \\frac14$.", "source": "Original · NCERT / NEET past-paper pattern"}
]
'''

# ---------------- questions_imported.json ----------------
FILES['questions_imported.json'] = r'''[
{"id": "jee-2016-paper-1-phy-2", "chapter": "Rotational Motion", "b": 1.8, "q": "A uniform wooden stick of mass $1.6 \\mathrm{~kg}$ and length $l$ rests in an inclined manner on a smooth, vertical wall of height $h(<l)$ such that a small portion of the stick extends beyond the wall. The reaction force of the wall on the stick is perpendicular to the stick. The stick makes an angle of $30^{\\circ}$ with the wall and the bottom of the stick is on a rough floor. The reaction of the wall on the stick is equal in magnitude to the reaction of the floor on the stick. The ratio $h / l$ and the frictional force $f$ at the bottom of the stick are\n\n$\\left(g=10 \\mathrm{~ms} \\mathrm{~s}^{2}\\right)$", "options": ["$\\frac{h}{l}=\\frac{\\sqrt{3}}{16}, f=\\frac{16 \\sqrt{3}}{3} \\mathrm{~N}$", "$\\frac{h}{l}=\\frac{3}{16}, f=\\frac{16 \\sqrt{3}}{3} \\mathrm{~N}$", "$\\frac{h}{l}=\\frac{3 \\sqrt{3}}{16}, f=\\frac{8 \\sqrt{3}}{3} \\mathrm{~N}$", "$\\frac{h}{l}=\\frac{3 \\sqrt{3}}{16}, f=\\frac{16 \\sqrt{3}}{3} \\mathrm{~N}$"], "answer": 3, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (D). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-chem-20", "chapter": "Chemical Thermodynamics", "b": 1.8, "q": "One mole of an ideal gas at $300 \\mathrm{~K}$ in thermal contact with surroundings expands isothermally from $1.0 \\mathrm{~L}$ to $2.0 \\mathrm{~L}$ against a constant pressure of $3.0 \\mathrm{~atm}$. In this process, the change in entropy of surroundings $\\left(\\Delta S_{\\text {surr }}\\right)$ in $\\mathrm{J} \\mathrm{K}^{-1}$ is\n\n(1 $\\mathrm{L} \\operatorname{atm}=101.3 \\mathrm{~J})$", "options": ["5.763", "1.013", "-1.013", "-5.763"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-chem-21", "chapter": "Classification of Elements and Periodicity", "b": 1.8, "q": "The increasing order of atomic radii of the following Group 13 elements is", "options": ["$\\mathrm{Al}<\\mathrm{Ga}<\\mathrm{In}<\\mathrm{Tl}$", "$\\mathrm{Ga}<\\mathrm{Al}<\\mathrm{In}<\\mathrm{Tl}$", "$\\mathrm{Al}<\\mathrm{In}<\\mathrm{Ga}<\\mathrm{Tl}$", "$\\mathrm{Al}<\\mathrm{Ga}<\\mathrm{Tl}<\\mathrm{In}$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-chem-22", "chapter": "Coordination Compounds", "b": 1.8, "q": "Among $[\\mathrm{Ni(CO)}_{4}]$, $[\\mathrm{NiCl}_{4}]^{2-}$, $[\\mathrm{Co(NH_3)_{4}Cl_2}]\\mathrm{Cl}$, $\\mathrm{Na_3}[\\mathrm{CoF_6}]$, $\\mathrm{Na_2O_2}$ and $\\mathrm{CsO_2}$ number of paramagnetic compounds is", "options": ["2", "3", "4", "5"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-chem-23", "chapter": "Polymers", "b": 1.8, "q": "On complete hydrogenation, natural rubber produces", "options": ["ethylene-propylene copolymer", "vulcanised rubber", "polypropylene", "polybutylene"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-math-37", "chapter": "Quadratic Equations", "b": 1.8, "q": "Let $-\\frac{\\pi}{6}<\\theta<-\\frac{\\pi}{12}$. Suppose $\\alpha_{1}$ and $\\beta_{1}$ are the roots of the equation $x^{2}-2 x \\sec \\theta+1=0$ and $\\alpha_{2}$ and $\\beta_{2}$ are the roots of the equation $x^{2}+2 x \\tan \\theta-1=0$. If $\\alpha_{1}>\\beta_{1}$ and $\\alpha_{2}>\\beta_{2}$, then $\\alpha_{1}+\\beta_{2}$ equals", "options": ["$2(\\sec \\theta-\\tan \\theta)$", "$2 \\sec \\theta$", "$-2 \\tan \\theta$", "0"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-math-38", "chapter": "Permutations and Combinations", "b": 1.8, "q": "A debate club consists of 6 girls and 4 boys. A team of 4 members is to be selected from this club including the selection of a captain (from among these 4 members) for the team. If the team has to include at most one boy, then the number of ways of selecting the team is", "options": ["380", "320", "260", "95"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-math-39", "chapter": "Trigonometric Equations", "b": 1.8, "q": "Let $S=\\left\\{x \\in(-\\pi, \\pi): x \\neq 0, \\pm \\frac{\\pi}{2}\\right\\}$. The sum of all distinct solutions of the equation $\\sqrt{3} \\sec x+\\operatorname{cosec} x+2(\\tan x-\\cot x)=0$ in the set $S$ is equal to", "options": ["$-\\frac{7 \\pi}{9}$", "$-\\frac{2 \\pi}{9}$", "0", "$\\frac{5 \\pi}{9}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-math-40", "chapter": "Probability", "b": 1.8, "q": "A computer producing factory has only two plants $T_{1}$ and $T_{2}$. Plant $T_{1}$ produces $20 \\%$ and plant $T_{2}$ produces $80 \\%$ of the total computers produced. $7 \\%$ of computers produced in the factory turn out to be defective. It is known that\n\n$P$ (computer turns out to be defective given that it is produced in plant $T_{1}$ )\n\n$=10 P\\left(\\right.$ computer turns out to be defective given that it is produced in plant $\\left.T_{2}\\right)$,\n\nwhere $P(E)$ denotes the probability of an event $E$. A computer produced in the factory is randomly selected and it does not turn out to be defective. Then the probability that it is produced in plant $T_{2}$ is", "options": ["$\\frac{36}{73}$", "$\\frac{47}{79}$", "$\\frac{78}{93}$", "$\\frac{75}{83}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-1-math-41", "chapter": "Application of Derivatives", "b": 1.8, "q": "The least value of $\\alpha \\in \\mathbb{R}$ for which $4 \\alpha x^{2}+\\frac{1}{x} \\geq 1$, for all $x>0$, is", "options": ["$\\frac{1}{64}$", "$\\frac{1}{32}$", "$\\frac{1}{27}$", "$\\frac{1}{25}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 1.", "source": "JEE Adv 2016 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-phy-1", "chapter": "Nuclei", "b": 1.8, "q": "The electrostatic energy of $Z$ protons uniformly distributed throughout a spherical nucleus of radius $R$ is given by\n\n\\[\nE=\\frac{3}{5} \\frac{Z(Z-1) e^{2}}{4 \\pi \\varepsilon_{0} R}\n\\]\n\nThe measured masses of the neutron, ${ }_{1}^{1} \\mathrm{H},{ }_{7}^{15} \\mathrm{~N}$ and ${ }_{8}^{15} \\mathrm{O}$ are $1.008665 \\mathrm{u}, 1.007825 \\mathrm{u}$, $15.000109 \\mathrm{u}$ and $15.003065 \\mathrm{u}$, respectively. Given that the radii of both the ${ }_{7}^{15} \\mathrm{~N}$ and ${ }_{8}^{15} \\mathrm{O}$ nuclei are same, $1 \\mathrm{u}=931.5 \\mathrm{MeV} / c^{2}$ ( $c$ is the speed of light) and $e^{2} /\\left(4 \\pi \\varepsilon_{0}\\right)=1.44 \\mathrm{MeV} \\mathrm{fm}$. Assuming that the difference between the binding energies of ${ }_{7}^{15} \\mathrm{~N}$ and ${ }_{8}^{15} \\mathrm{O}$ is purely due to the electrostatic energy, the radius of either of the nuclei is\n\n$\\left(1 \\mathrm{fm}=10^{-15} \\mathrm{~m}\\right)$", "options": ["$2.85 \\mathrm{fm}$", "$3.03 \\mathrm{fm}$", "$3.42 \\mathrm{fm}$", "$3.80 \\mathrm{fm}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-phy-2", "chapter": "Nuclei", "b": 1.8, "q": "An accident in a nuclear laboratory resulted in deposition of a certain amount of radioactive material of half-life 18 days inside the laboratory. Tests revealed that the radiation was 64 times more than the permissible level required for safe operation of the laboratory. What is the minimum number of days after which the laboratory can be considered safe for use?", "options": ["64", "90", "108", "120"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-phy-3", "chapter": "Thermodynamics", "b": 1.8, "q": "A gas is enclosed in a cylinder with a movable frictionless piston. Its initial thermodynamic state at pressure $P_{i}=10^{5} \\mathrm{~Pa}$ and volume $V_{i}=10^{-3} \\mathrm{~m}^{3}$ changes to a final state at $P_{f}=(1 / 32) \\times 10^{5} \\mathrm{~Pa}$ and $V_{f}=8 \\times 10^{-3} \\mathrm{~m}^{3}$ in an adiabatic quasi-static process, such that $P^{3} V^{5}=$ constant. Consider another thermodynamic process that brings the system from the same initial state to the same final state in two steps: an isobaric expansion at $P_{i}$ followed by an isochoric (isovolumetric) process at volume $V_{f}$. The amount of heat supplied to the system in the two-step process is approximately", "options": ["$112 \\mathrm{~J}$", "$294 \\mathrm{~J}$", "$588 \\mathrm{~J}$", "$813 \\mathrm{~J}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-phy-4", "chapter": "Thermal Properties of Matter", "b": 1.8, "q": "The ends $\\mathrm{Q}$ and $\\mathrm{R}$ of two thin wires, $\\mathrm{PQ}$ and RS, are soldered (joined) together. Initially each of the wires has a length of $1 \\mathrm{~m}$ at $10^{\\circ} \\mathrm{C}$. Now the end $P$ is maintained at $10^{\\circ} \\mathrm{C}$, while the end $\\mathrm{S}$ is heated and maintained at $400^{\\circ} \\mathrm{C}$. The system is thermally insulated from its surroundings. If the thermal conductivity of wire $\\mathrm{PQ}$ is twice that of the wire $R S$ and the coefficient of linear thermal expansion of $\\mathrm{PQ}$ is $1.2 \\times 10^{-5} \\mathrm{~K}^{-1}$, the change in length of the wire $P Q$ is", "options": ["$0.78 \\mathrm{~mm}$", "$0.90 \\mathrm{~mm}$", "$1.56 \\mathrm{~mm}$", "$2.34 \\mathrm{~mm}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-chem-19", "chapter": "Electrochemistry", "b": 1.8, "q": "For the following electrochemical cell at $298 \\mathrm{~K}$,\n\n$\\mathrm{Pt}(s) \\mid \\mathrm{H}_{2}(g, 1$ bar $)\\left|\\mathrm{H}^{+}(aq, 1 \\mathrm{M}) \\| \\mathrm{M}^{4+}(aq), \\mathrm{M}^{2+}(aq)\\right| \\operatorname{Pt}(s)$\n\n$E_{\\text {cell }}=0.092 \\mathrm{~V}$ when $\\frac{\\left[\\mathrm{M}^{2+}(aq)\\right]}{\\left[M^{4+}(aq)\\right]}=10^{x}$.\n\nGiven : $E_{\\mathrm{M}^{4} / \\mathrm{M}^{2+}}^{0}=0.151 \\mathrm{~V} ; 2.303 \\frac{R T}{F}=0.059 \\mathrm{~V}$\n\nThe value of $x$ is", "options": ["-2", "-1", "1", "2"], "answer": 3, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (D). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-chem-22", "chapter": "Coordination Compounds", "b": 1.8, "q": "The geometries of the ammonia complexes of $\\mathrm{Ni}^{2+}, \\mathrm{Pt}^{2+}$ and $\\mathrm{Zn}^{2+}$, respectively, are", "options": ["octahedral, square planar and tetrahedral", "square planar, octahedral and tetrahedral", "tetrahedral, square planar and octahedral", "octahedral, tetrahedral and square planar"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-math-38", "chapter": "Sequences and Series", "b": 1.8, "q": "Let $b_{i}>1$ for $i=1,2, \\ldots, 101$. Suppose $\\log _{e} b_{1}, \\log _{e} b_{2}, \\ldots, \\log _{e} b_{101}$ are in Arithmetic Progression (A.P.) with the common difference $\\log _{e} 2$. Suppose $a_{1}, a_{2}, \\ldots, a_{101}$ are in A.P. such that $a_{1}=b_{1}$ and $a_{51}=b_{51}$. If $t=b_{1}+b_{2}+\\cdots+b_{51}$ and $s=a_{1}+a_{2}+\\cdots+a_{51}$, then", "options": ["$s>t$ and $a_{101}>b_{101}$", "$s>t$ and $a_{101}<b_{101}$", "$s<t$ and $a_{101}>b_{101}$", "$s<t$ and $a_{101}<b_{101}$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-math-40", "chapter": "Definite Integration", "b": 1.8, "q": "The value of $\\int_{-\\frac{\\pi}{2}}^{\\frac{\\pi}{2}} \\frac{x^{2} \\cos x}{1+e^{x}} d x$ is equal to", "options": ["$\\frac{\\pi^{2}}{4}-2$", "$\\frac{\\pi^{2}}{4}+2$", "$\\pi^{2}-e^{\\frac{\\pi}{2}}$", "$\\pi^{2}+e^{\\frac{\\pi}{2}}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2016-paper-2-math-42", "chapter": "3D Geometry", "b": 1.8, "q": "Let $P$ be the image of the point $(3,1,7)$ with respect to the plane $x-y+z=3$. Then the equation of the plane passing through $P$ and containing the straight line $\\frac{x}{1}=\\frac{y}{2}=\\frac{z}{1}$ is", "options": ["$x+y-3 z=0$", "$3 x+z=0$", "$x-4 y+7 z=0$", "$2 x-y=0$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2016 Paper 2.", "source": "JEE Adv 2016 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-phy-1", "chapter": "Gravitation", "b": 1.8, "q": "Consider a spherical gaseous cloud of mass density $\\rho(r)$ in free space where $r$ is the radial distance from its center. The gaseous cloud is made of particles of equal mass $m$ moving in circular orbits about the common center with the same kinetic energy $K$. The force acting on the particles is their mutual gravitational force. If $\\rho(r)$ is constant in time, the particle number density $n(r)=\\rho(r) / m$ is\n\n[ $G$ is universal gravitational constant]", "options": ["$\\frac{K}{2 \\pi r^{2} m^{2} G}$", "$\\frac{K}{\\pi r^{2} m^{2} G}$", "$\\frac{3 K}{\\pi r^{2} m^{2} G}$", "$\\frac{K}{6 \\pi r^{2} m^{2} G}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-phy-2", "chapter": "Electrostatics", "b": 1.8, "q": "A thin spherical insulating shell of radius $R$ carries a uniformly distributed charge such that the potential at its surface is $V_{0}$. A hole with a small area $\\alpha 4 \\pi R^{2}(\\alpha \\ll 1)$ is made on the shell without affecting the rest of the shell. Which one of the following statements is correct?", "options": ["The potential at the center of the shell is reduced by $2 \\alpha V_{0}$", "The magnitude of electric field at the center of the shell is reduced by $\\frac{\\alpha V_{0}}{2 R}$", "The ratio of the potential at the center of the shell to that of the point at $\\frac{1}{2} R$ from center towards the hole will be $\\frac{1-\\alpha}{1-2 \\alpha}$", "The magnitude of electric field at a point, located on a line passing through the hole and shell's center, on a distance $2 R$ from the center of the spherical shell will be reduced by $\\frac{\\alpha V_{0}}{2 R}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-phy-3", "chapter": "Thermal Properties of Matter", "b": 1.8, "q": "A current carrying wire heats a metal rod. The wire provides a constant power $(P)$ to the rod. The metal rod is enclosed in an insulated container. It is observed that the temperature $(T)$ in the metal rod changes with time $(t)$ as\n\n\\[\nT(t)=T_{0}\\left(1+\\beta t^{\\frac{1}{4}}\\right)\n\\]\n\nwhere $\\beta$ is a constant with appropriate dimension while $T_{0}$ is a constant with dimension of temperature. The heat capacity of the metal is,", "options": ["$\\frac{4 P\\left(T(t)-T_{0}\\right)^{3}}{\\beta^{4} T_{0}^{4}}$", "$\\frac{4 P\\left(T(t)-T_{0}\\right)^{4}}{\\beta^{4} T_{0}^{5}}$", "$\\frac{4 P\\left(T(t)-T_{0}\\right)^{2}}{\\beta^{4} T_{0}^{3}}$", "$\\frac{4 P\\left(T(t)-T_{0}\\right)}{\\beta^{4} T_{0}^{2}}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-phy-4", "chapter": "Nuclei", "b": 1.8, "q": "In a radioactive sample, ${ }_{19}^{40} \\mathrm{~K}$ nuclei either decay into stable ${ }_{20}^{40} \\mathrm{Ca}$ nuclei with decay constant $4.5 \\times 10^{-10}$ per year or into stable ${ }_{18}^{40} \\mathrm{Ar}$ nuclei with decay constant $0.5 \\times 10^{-10}$ per year. Given that in this sample all the stable ${ }_{20}^{40} \\mathrm{Ca}$ and ${ }_{18}^{40} \\mathrm{Ar}$ nuclei are produced by the ${ }_{19}^{40} \\mathrm{~K}$ nuclei only. In time $t \\times 10^{9}$ years, if the ratio of the sum of stable ${ }_{20}^{40} \\mathrm{Ca}$ and ${ }_{18}^{40} \\mathrm{Ar}$ nuclei to the radioactive ${ }_{19}^{40} \\mathrm{~K}$ nuclei is 99 , the value of $t$ will be,\n\n[Given: $\\ln 10=2.3]$", "options": ["1.15", "9.2", "2.3", "4.6"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-chem-19", "chapter": "d- and f-Block Elements", "b": 1.8, "q": "The green colour produced in the borax bead test of a chromium(III) salt is due to", "options": ["$\\mathrm{Cr}\\left(\\mathrm{BO}_{2}\\right)_{3}$", "$\\mathrm{Cr}_{2}\\left(\\mathrm{~B}_{4} \\mathrm{O}_{7}\\right)_{3}$", "$\\mathrm{Cr}_{2} \\mathrm{O}_{3}$", "$\\mathrm{CrB}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-chem-20", "chapter": "Metallurgy", "b": 1.8, "q": "Calamine, malachite, magnetite and cryolite, respectively, are", "options": ["$\\mathrm{ZnSO}_{4}, \\mathrm{CuCO}_{3}, \\mathrm{Fe}_{2} \\mathrm{O}_{3}, \\mathrm{AlF}_{3}$", "$\\mathrm{ZnSO}_{4}, \\mathrm{Cu}(\\mathrm{OH})_{2}, \\mathrm{Fe}_{3} \\mathrm{O}_{4}, \\mathrm{Na}_{3} \\mathrm{AlF}_{6}$", "$\\mathrm{ZnCO}_{3}, \\mathrm{CuCO}_{3} \\cdot \\mathrm{Cu}(\\mathrm{OH})_{2}, \\mathrm{Fe}_{3} \\mathrm{O}_{4}, \\mathrm{Na}_{3} \\mathrm{AlF}_{6}$", "$\\mathrm{ZnCO}_{3}, \\mathrm{CuCO}_{3}, \\mathrm{Fe}_{2} \\mathrm{O}_{3}, \\mathrm{Na}_{3} \\mathrm{AlF}_{6}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-math-37", "chapter": "Complex Numbers", "b": 1.8, "q": "Let $S$ be the set of all complex numbers $Z$ satisfying $|z-2+i| \\geq \\sqrt{5}$. If the complex number $Z_{0}$ is such that $\\frac{1}{\\left|Z_{0}-1\\right|}$ is the maximum of the set $\\left\\{\\frac{1}{|z-1|}: z \\in S\\right\\}$, then the principal argument of $\\frac{4-z_{0}-\\overline{z_{0}}}{Z_{0}-\\overline{z_{0}}+2 i}$ is", "options": ["$-\\frac{\\pi}{2}$", "$\\frac{\\pi}{4}$", "$\\frac{\\pi}{2}$", "$\\frac{3 \\pi}{4}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-math-39", "chapter": "Circles", "b": 1.8, "q": "A line $y=m x+1$ intersects the circle $(x-3)^{2}+(y+2)^{2}=25$ at the points $P$ and $Q$. If the midpoint of the line segment $P Q$ has $x$-coordinate $-\\frac{3}{5}$, then which one of the following options is correct?", "options": ["$-3 \\leq m<-1$", "$2 \\leq m<4$", "$4 \\leq m<6$", "$6 \\leq m<8$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2019-paper-1-math-40", "chapter": "Area Under Curves", "b": 1.8, "q": "The area of the region $\\left\\{(x, y): x y \\leq 8,1 \\leq y \\leq x^{2}\\right\\}$ is", "options": ["$16 \\log _{e} 2-\\frac{14}{3}$", "$8 \\log _{e} 2-\\frac{14}{3}$", "$16 \\log _{e} 2-6$", "$8 \\log _{e} 2-\\frac{7}{3}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2019 Paper 1.", "source": "JEE Adv 2019 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2020-paper-1-chem-20", "chapter": "p-Block Elements", "b": 1.8, "q": "Which of the following liberates $\\mathrm{O}_{2}$ upon hydrolysis?", "options": ["$\\mathrm{Pb}_{3} \\mathrm{O}_{4}$", "$\\mathrm{KO}_{2}$", "$\\mathrm{Na}_{2} \\mathrm{O}_{2}$", "$\\mathrm{Li}_{2} \\mathrm{O}_{2}$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2020 Paper 1.", "source": "JEE Adv 2020 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2020-paper-1-chem-21", "chapter": "Qualitative Analysis", "b": 1.8, "q": "A colorless aqueous solution contains nitrates of two metals, $\\mathbf{X}$ and $\\mathbf{Y}$. When it was added to an aqueous solution of $\\mathrm{NaCl}$, a white precipitate was formed. This precipitate was found to be partly soluble in hot water to give a residue $\\mathbf{P}$ and a solution $\\mathbf{Q}$. The residue $\\mathbf{P}$ was soluble in aq. $\\mathrm{NH}_{3}$ and also in excess sodium thiosulfate. The hot solution $\\mathbf{Q}$ gave a yellow precipitate with KI. The metals $\\mathbf{X}$ and $\\mathbf{Y}$, respectively, are", "options": ["Ag and $\\mathrm{Pb}$", "Ag and Cd", "$\\mathrm{Cd}$ and $\\mathrm{Pb}$", "Cd and Zn"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2020 Paper 1.", "source": "JEE Adv 2020 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2020-paper-1-math-37", "chapter": "Quadratic Equations", "b": 1.8, "q": "Suppose $a, b$ denote the distinct real roots of the quadratic polynomial $x^{2}+20 x-2020$ and suppose $c, d$ denote the distinct complex roots of the quadratic polynomial $x^{2}-20 x+2020$. Then the value of\n\n\\[\na c(a-c)+a d(a-d)+b c(b-c)+b d(b-d)\n\\]\n\nis", "options": ["0", "8000", "8080", "16000"], "answer": 3, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (D). Source: JEE Adv 2020 Paper 1.", "source": "JEE Adv 2020 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2020-paper-1-math-38", "chapter": "Continuity and Differentiability", "b": 1.8, "q": "If the function $f: \\mathbb{R} \\rightarrow \\mathbb{R}$ is defined by $f(x)=|x|(x-\\sin x)$, then which of the following statements is TRUE?", "options": ["$f$ is one-one, but NOT onto", "$f$ is onto, but NOT one-one", "$f$ is BOTH one-one and onto", "$f$ is NEITHER one-one NOR onto"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2020 Paper 1.", "source": "JEE Adv 2020 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2020-paper-1-math-39", "chapter": "Area Under Curves", "b": 1.8, "q": "Let the functions $f: \\mathbb{R} \\rightarrow \\mathbb{R}$ and $g: \\mathbb{R} \\rightarrow \\mathbb{R}$ be defined by\n\n\\[\nf(x)=e^{x-1}-e^{-|x-1|} \\quad \\text { and } \\quad g(x)=\\frac{1}{2}\\left(e^{x-1}+e^{1-x}\\right)\n\\]\n\nThen the area of the region in the first quadrant bounded by the curves $y=f(x), y=g(x)$ and $x=0$ is", "options": ["$(2-\\sqrt{3})+\\frac{1}{2}\\left(e-e^{-1}\\right)$", "$(2+\\sqrt{3})+\\frac{1}{2}\\left(e-e^{-1}\\right)$", "$(2-\\sqrt{3})+\\frac{1}{2}\\left(e+e^{-1}\\right)$", "$(2+\\sqrt{3})+\\frac{1}{2}\\left(e+e^{-1}\\right)$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2020 Paper 1.", "source": "JEE Adv 2020 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2020-paper-1-math-40", "chapter": "Ellipse", "b": 1.8, "q": "Let $a, b$ and $\\lambda$ be positive real numbers. Suppose $P$ is an end point of the latus rectum of the parabola $y^{2}=4 \\lambda x$, and suppose the ellipse $\\frac{x^{2}}{a^{2}}+\\frac{y^{2}}{b^{2}}=1$ passes through the point $P$. If the tangents to the parabola and the ellipse at the point $P$ are perpendicular to each other, then the eccentricity of the ellipse is", "options": ["$\\frac{1}{\\sqrt{2}}$", "$\\frac{1}{2}$", "$\\frac{1}{3}$", "$\\frac{2}{5}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2020 Paper 1.", "source": "JEE Adv 2020 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2020-paper-1-math-41", "chapter": "Probability", "b": 1.8, "q": "Let $C_{1}$ and $C_{2}$ be two biased coins such that the probabilities of getting head in a single toss are $\\frac{2}{3}$ and $\\frac{1}{3}$, respectively. Suppose $\\alpha$ is the number of heads that appear when $C_{1}$ is tossed twice, independently, and suppose $\\beta$ is the number of heads that appear when $C_{2}$ is tossed twice, independently. Then the probability that the roots of the quadratic polynomial $x^{2}-\\alpha x+\\beta$ are real and equal, is", "options": ["$\\frac{40}{81}$", "$\\frac{20}{81}$", "$\\frac{1}{2}$", "$\\frac{1}{4}$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2020 Paper 1.", "source": "JEE Adv 2020 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2020-paper-1-math-42", "chapter": "Application of Derivatives", "b": 1.8, "q": "Consider all rectangles lying in the region\n\n\\[\n\\left\\{(x, y) \\in \\mathbb{R} \\times \\mathbb{R}: 0 \\leq x \\leq \\frac{\\pi}{2} \\text { and } 0 \\leq y \\leq 2 \\sin (2 x)\\right\\}\n\\]\n\nand having one side on the $x$-axis. The area of the rectangle which has the maximum perimeter among all such rectangles, is", "options": ["$\\frac{3 \\pi}{2}$", "$\\pi$", "$\\frac{\\pi}{2 \\sqrt{3}}$", "$\\frac{\\pi \\sqrt{3}}{2}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2020 Paper 1.", "source": "JEE Adv 2020 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-1-phy-4", "chapter": "Nuclei", "b": 1.8, "q": "A heavy nucleus $Q$ of half-life 20 minutes undergoes alpha-decay with probability of $60 \\%$ and beta-decay with probability of $40 \\%$. Initially, the number of $Q$ nuclei is 1000. The number of alpha-decays of $Q$ in the first one hour is", "options": ["50", "75", "350", "525"], "answer": 3, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (D). Source: JEE Adv 2021 Paper 1.", "source": "JEE Adv 2021 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-1-chem-22", "chapter": "Coordination Compounds", "b": 1.8, "q": "The calculated spin only magnetic moments of $\\left[\\mathrm{Cr}\\left(\\mathrm{NH}_{3}\\right)_{6}\\right]^{3+}$ and $\\left[\\mathrm{CuF}_{6}\\right]^{3-}$ in $\\mathrm{BM}$, respectively, are\n\n(Atomic numbers of $\\mathrm{Cr}$ and $\\mathrm{Cu}$ are 24 and 29, respectively)", "options": ["3.87 and 2.84", "4.90 and 1.73", "3.87 and 1.73", "4.90 and 2.84"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2021 Paper 1.", "source": "JEE Adv 2021 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-1-math-37", "chapter": "Circles", "b": 1.8, "q": "Consider a triangle $\\Delta$ whose two sides lie on the $x$-axis and the line $x+y+1=0$. If the orthocenter of $\\Delta$ is $(1,1)$, then the equation of the circle passing through the vertices of the triangle $\\Delta$ is", "options": ["$x^{2}+y^{2}-3 x+y=0$", "$x^{2}+y^{2}+x+3 y=0$", "$x^{2}+y^{2}+2 y-1=0$", "$x^{2}+y^{2}+x+y=0$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2021 Paper 1.", "source": "JEE Adv 2021 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-1-math-38", "chapter": "Area Under Curves", "b": 1.8, "q": "The area of the region\n\n\\[\n\\left\\{(x, y): 0 \\leq x \\leq \\frac{9}{4}, \\quad 0 \\leq y \\leq 1, \\quad x \\geq 3 y, \\quad x+y \\geq 2\\right\\}\n\\]\n\nis", "options": ["$\\frac{11}{32}$", "$\\frac{35}{96}$", "$\\frac{37}{96}$", "$\\frac{13}{32}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2021 Paper 1.", "source": "JEE Adv 2021 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-1-math-39", "chapter": "Probability", "b": 1.8, "q": "Consider three sets $E_{1}=\\{1,2,3\\}, F_{1}=\\{1,3,4\\}$ and $G_{1}=\\{2,3,4,5\\}$. Two elements are chosen at random, without replacement, from the set $E_{1}$, and let $S_{1}$ denote the set of these chosen elements. Let $E_{2}=E_{1}-S_{1}$ and $F_{2}=F_{1} \\cup S_{1}$. Now two elements are chosen at random, without replacement, from the set $F_{2}$ and let $S_{2}$ denote the set of these chosen elements.\n\nLet $G_{2}=G_{1} \\cup S_{2}$. Finally, two elements are chosen at random, without replacement, from the set $G_{2}$ and let $S_{3}$ denote the set of these chosen elements.\n\nLet $E_{3}=E_{2} \\cup S_{3}$. Given that $E_{1}=E_{3}$, let $p$ be the conditional probability of the event $S_{1}=\\{1,2\\}$. Then the value of $p$ is", "options": ["$\\frac{1}{5}$", "$\\frac{3}{5}$", "$\\frac{1}{2}$", "$\\frac{2}{5}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2021 Paper 1.", "source": "JEE Adv 2021 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-1-math-40", "chapter": "Complex Numbers", "b": 1.8, "q": "Let $\\theta_{1}, \\theta_{2}, \\ldots, \\theta_{10}$ be positive valued angles (in radian) such that $\\theta_{1}+\\theta_{2}+\\cdots+\\theta_{10}=2 \\pi$. Define the complex numbers $z_{1}=e^{i \\theta_{1}}, z_{k}=z_{k-1} e^{i \\theta_{k}}$ for $k=2,3, \\ldots, 10$, where $i=\\sqrt{-1}$. Consider the statements $P$ and $Q$ given below:\n\n\\[\n\\begin{aligned}\n\n& P:\\left|z_{2}-z_{1}\\right|+\\left|z_{3}-z_{2}\\right|+\\cdots+\\left|z_{10}-z_{9}\\right|+\\left|z_{1}-z_{10}\\right| \\leq 2 \\pi \\\\\n\n& Q:\\left|z_{2}^{2}-z_{1}^{2}\\right|+\\left|z_{3}^{2}-z_{2}^{2}\\right|+\\cdots+\\left|z_{10}^{2}-z_{9}^{2}\\right|+\\left|z_{1}^{2}-z_{10}^{2}\\right| \\leq 4 \\pi\n\n\\end{aligned}\n\\]\n\nThen,", "options": ["$P$ is TRUE and $Q$ is FALSE", "$Q$ is TRUE and $P$ is FALSE", "both $P$ and $Q$ are TRUE", "both $P$ and $Q$ are FALSE"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2021 Paper 1.", "source": "JEE Adv 2021 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-2-chem-33", "chapter": "Qualitative Analysis", "b": 1.8, "q": "The reaction of $\\mathrm{K}_3\\left[\\mathrm{Fe}(\\mathrm{CN})_6\\right]$ with freshly prepared $\\mathrm{FeSO}_4$ solution produces a dark blue precipitate called Turnbull's blue. Reaction of $\\mathrm{K}_4\\left[\\mathrm{Fe}(\\mathrm{CN})_6\\right]$ with the $\\mathrm{FeSO}_4$ solution in complete absence of air produces a white precipitate $\\mathbf{X}$, which turns blue in air. Mixing the $\\mathrm{FeSO}_4$ solution with $\\mathrm{NaNO}_3$, followed by a slow addition of concentrated $\\mathrm{H}_2 \\mathrm{SO}_4$ through the side of the test tube produces a brown ring. Precipitate $\\mathbf{X}$ is", "options": ["$\\mathrm{Fe}_{4}\\left[\\mathrm{Fe}(\\mathrm{CN})_{6}\\right]_{3}$", "$\\mathrm{Fe}\\left[\\mathrm{Fe}(\\mathrm{CN})_{6}\\right]$", "$\\mathrm{K}_{2} \\mathrm{Fe}\\left[\\mathrm{Fe}(\\mathrm{CN})_{6}\\right]$", "$\\mathrm{KFe}\\left[\\mathrm{Fe}(\\mathrm{CN})_{6}\\right]$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2021 Paper 2.", "source": "JEE Adv 2021 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-2-chem-34", "chapter": "Qualitative Analysis", "b": 1.8, "q": "The reaction of $\\mathrm{K}_3\\left[\\mathrm{Fe}(\\mathrm{CN})_6\\right]$ with freshly prepared $\\mathrm{FeSO}_4$ solution produces a dark blue precipitate called Turnbull's blue. Reaction of $\\mathrm{K}_4\\left[\\mathrm{Fe}(\\mathrm{CN})_6\\right]$ with the $\\mathrm{FeSO}_4$ solution in complete absence of air produces a white precipitate $\\mathbf{X}$, which turns blue in air. Mixing the $\\mathrm{FeSO}_4$ solution with $\\mathrm{NaNO}_3$, followed by a slow addition of concentrated $\\mathrm{H}_2 \\mathrm{SO}_4$ through the side of the test tube produces a brown ring. Among the following, the brown ring is due to the formation of", "options": ["$\\left[\\mathrm{Fe}(\\mathrm{NO})_{2}\\left(\\mathrm{SO}_{4}\\right)_{2}\\right]^{2-}$", "$\\left[\\mathrm{Fe}(\\mathrm{NO})_{2}\\left(\\mathrm{H}_{2} \\mathrm{O}\\right)_{4}\\right]^{3+}$", "$\\left[\\mathrm{Fe}(\\mathrm{NO})_{4}\\left(\\mathrm{SO}_{4}\\right)_{2}\\right]$", "$\\left[\\mathrm{Fe}(\\mathrm{NO})\\left(\\mathrm{H}_{2} \\mathrm{O}\\right)_{5}\\right]^{2+}$"], "answer": 3, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (D). Source: JEE Adv 2021 Paper 2.", "source": "JEE Adv 2021 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-2-math-49", "chapter": "Sequences and Series", "b": 1.8, "q": "Let\n\\[\\mathrm{M}=\\left\\{(\\mathrm{x}, \\mathrm{y}) \\in \\mathbb{R} \\times \\mathbb{R}: \\mathrm{x}^2+\\mathrm{y}^2 \\leq \\mathrm{r}^2\\right\\}\\]\nwhere $\\mathrm{r}>0$. Consider the geometric progression $a_n=\\frac{1}{2^{n-1}}, n=1,2,3, \\ldots$. Let $S_0=0$ and, for $n \\geq 1$, let $S_n$ denote the sum of the first $n$ terms of this progression. For $n \\geq 1$, let $C_n$ denote the circle with center $\\left(S_{n-1}, 0\\right)$ and radius $a_n$, and $D_n$ denote the circle with center $\\left(S_{n-1}, S_{n-1}\\right)$ and radius $a_n$. Consider $M$ with $r=\\frac{1025}{513}$. Let $k$ be the number of all those circles $C_{n}$ that are inside $M$. Let $l$ be the maximum possible number of circles among these $k$ circles such that no two circles intersect. Then", "options": ["$k+2 l=22$", "$2 k+l=26$", "$2 k+3 l=34$", "$3 k+2 l=40$"], "answer": 3, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (D). Source: JEE Adv 2021 Paper 2.", "source": "JEE Adv 2021 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-2-math-50", "chapter": "Circles", "b": 1.8, "q": "Let\n\\[\\mathrm{M}=\\left\\{(\\mathrm{x}, \\mathrm{y}) \\in \\mathbb{R} \\times \\mathbb{R}: \\mathrm{x}^2+\\mathrm{y}^2 \\leq \\mathrm{r}^2\\right\\}\\]\nwhere $\\mathrm{r}>0$. Consider the geometric progression $a_n=\\frac{1}{2^{n-1}}, n=1,2,3, \\ldots$. Let $S_0=0$ and, for $n \\geq 1$, let $S_n$ denote the sum of the first $n$ terms of this progression. For $n \\geq 1$, let $C_n$ denote the circle with center $\\left(S_{n-1}, 0\\right)$ and radius $a_n$, and $D_n$ denote the circle with center $\\left(S_{n-1}, S_{n-1}\\right)$ and radius $a_n$. Consider $M$ with $r=\\frac{\\left(2^{199}-1\\right) \\sqrt{2}}{2^{198}}$. The number of all those circles $D_{n}$ that are inside $M$ is", "options": ["198", "199", "200", "201"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2021 Paper 2.", "source": "JEE Adv 2021 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2021-paper-2-math-51", "chapter": "Definite Integration", "b": 1.8, "q": "Let $\\psi_1:[0, \\infty) \\rightarrow \\mathbb{R}, \\psi_2:[0, \\infty) \\rightarrow \\mathbb{R}, f:[0, \\infty) \\rightarrow \\mathbb{R}$ and $g:[0, \\infty) \\rightarrow \\mathbb{R}$ be functions such that\n\\[\\begin{aligned}\n& f(0)=\\mathrm{g}(0)=0, \\\\\n& \\psi_1(\\mathrm{x})=\\mathrm{e}^{-\\mathrm{x}}+\\mathrm{x}, \\quad \\mathrm{x} \\geq 0, \\\\\n& \\psi_2(\\mathrm{x})=\\mathrm{x}^2-2 \\mathrm{x}-2 \\mathrm{e}^{-\\mathrm{x}}+2, \\mathrm{x} \\geq 0, \\\\\n& f(\\mathrm{x})=\\int_{-\\mathrm{x}}^{\\mathrm{x}}\\left(|\\mathrm{t}|-\\mathrm{t}^2\\right) \\mathrm{e}^{-\\mathrm{t}^2} \\mathrm{dt}, \\mathrm{x}>0\n\\end{aligned}\\]\nand\n\\[g(x)=\\int_0^{x^2} \\sqrt{t} e^{-t} d t, x>0\\]. Which of the following statements is TRUE?", "options": ["$f(\\sqrt{\\ln 3})+g(\\sqrt{\\ln 3})=\\frac{1}{3}$", "For every $x>1$, there exists an $\\alpha \\in(1, x)$ such that $\\psi_{1}(x)=1+\\alpha x$", "For every $x>0$, there exists a $\\beta \\in(0, x)$ such that $\\psi_{2}(x)=2 x\\left(\\psi_{1}(\\beta)-1\\right)$", "$f$ is an increasing function on the interval $\\left[0, \\frac{3}{2}\\right]$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2021 Paper 2.", "source": "JEE Adv 2021 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-1-chem-52", "chapter": "s-Block Elements", "b": 1.8, "q": "LIST-I contains compounds and LIST-II contains reactions\n\n\nLIST-I\n(I) $\\mathrm{H}_2 \\mathrm{O}_2$\n(II) $\\mathrm{Mg}(\\mathrm{OH})_2$\n(III) $\\mathrm{BaCl}_2$\n(IV) $\\mathrm{CaCO}_3$\n\nLIST-II\n(P) $\\mathrm{Mg}\\left(\\mathrm{HCO}_{3}\\right)_{2}+\\mathrm{Ca}(\\mathrm{OH})_{2} \\rightarrow$\n(Q) $\\mathrm{BaO}_{2}+\\mathrm{H}_{2} \\mathrm{SO}_{4} \\rightarrow$\n(R) $\\mathrm{Ca}(\\mathrm{OH})_{2}+\\mathrm{MgCl}_{2} \\rightarrow$\n(S) $\\mathrm{BaO}_{2}+\\mathrm{HCl} \\rightarrow$\n(T) $\\mathrm{Ca}\\left(\\mathrm{HCO}_{3}\\right)_{2}+\\mathrm{Ca}(\\mathrm{OH})_{2} \\rightarrow$\n\nMatch each compound in LIST-I with its formation reaction(s) in LIST-II, and choose the correct option\\", "options": ["I $\\rightarrow$ Q; II $\\rightarrow$ P; III $\\rightarrow \\mathrm{S}$; IV $\\rightarrow$ R", "I $\\rightarrow$ T; II $\\rightarrow$ P; III $\\rightarrow$ Q; IV $\\rightarrow \\mathrm{R}$", "I $\\rightarrow$ T; II $\\rightarrow$ R; III $\\rightarrow$ Q; IV $\\rightarrow$ P", "I $\\rightarrow$ Q; II $\\rightarrow$ R; III $\\rightarrow \\mathrm{S}$; IV $\\rightarrow \\mathrm{P}$"], "answer": 3, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (D). Source: JEE Adv 2022 Paper 1.", "source": "JEE Adv 2022 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-1-chem-53", "chapter": "Coordination Compounds", "b": 1.8, "q": "LIST-I contains metal species and LIST-II contains their properties.\n\nLIST-I\n(I) $\\left[\\mathrm{Cr}(\\mathrm{CN})_{6}\\right]^{4-}$\n(II) $\\left[\\mathrm{RuCl}_{6}\\right]^{2-}$\n(III) $\\left[\\mathrm{Cr}\\left(\\mathrm{H}_{2} \\mathrm{O}\\right)_{6}\\right]^{2+}$\n(IV) $\\left[\\mathrm{Fe}\\left(\\mathrm{H}_{2} \\mathrm{O}\\right)_{6}\\right]^{2+}$\n\nLIST-II\n(P) $t_{2 \\mathrm{~g}}$ orbitals contain 4 electrons\n(Q) $\\mu$ (spin-only $=4.9 \\mathrm{BM}$\n(R) low spin complex ion\n(S) metal ion in $4+$ oxidation state\n(T) $d^{4}$ species\n\n[Given: Atomic number of $\\mathrm{Cr}=24, \\mathrm{Ru}=44, \\mathrm{Fe}=26$ ]\n\nMatch each metal species in LIST-I with their properties in LIST-II, and choose the correct option", "options": ["I $\\rightarrow$ R, T; II $\\rightarrow$ P, S; III $\\rightarrow$ Q, T; IV $\\rightarrow$ P, Q", "I $\\rightarrow$ R, S; II $\\rightarrow$ P, T; III $\\rightarrow$ P, Q; IV $\\rightarrow$ Q, T", "I $\\rightarrow \\mathrm{P}, \\mathrm{R} ; \\mathrm{II} \\rightarrow \\mathrm{R}, \\mathrm{S}$; III $\\rightarrow \\mathrm{R}, \\mathrm{T}$; IV $\\rightarrow \\mathrm{P}, \\mathrm{T}$", "I $\\rightarrow$ Q, T; II $\\rightarrow \\mathrm{S}, \\mathrm{T}$; III $\\rightarrow \\mathrm{P}, \\mathrm{T}$; IV $\\rightarrow \\mathrm{Q}, \\mathrm{R}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2022 Paper 1.", "source": "JEE Adv 2022 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-1-chem-54", "chapter": "Amines", "b": 1.8, "q": "Match the compounds in LIST-I with the observations in LIST-II, and choose the correct option.\n\nLIST-I\n\n(I) Aniline\n(II) $o$-Cresol\n(III) Cysteine\n(IV) Caprolactam\n\nLIST-II\n\n(P) Sodium fusion extract of the compound on boiling with $\\mathrm{FeSO}_{4}$, followed by acidification with conc. $\\mathrm{H}_{2} \\mathrm{SO}_{4}$, gives Prussian blue color.\n(Q) Sodium fusion extract of the compound on treatment with sodium nitroprusside gives blood red color.\n(R) Addition of the compound to a saturated solution of $\\mathrm{NaHCO}_{3}$ results in effervescence.\n(S) The compound reacts with bromine water to give a white precipitate.\n(T) Treating the compound with neutral $\\mathrm{FeCl}_{3}$ solution produces violet color.", "options": ["$\\mathrm{I} \\rightarrow \\mathrm{P}, \\mathrm{Q}$; II $\\rightarrow \\mathrm{S}$; III $\\rightarrow \\mathrm{Q}$, R; IV $\\rightarrow \\mathrm{P}$", "$\\mathrm{I} \\rightarrow \\mathrm{P}$; II $\\rightarrow \\mathrm{R}, \\mathrm{S}$; III $\\rightarrow \\mathrm{R}$; IV $\\rightarrow \\mathrm{Q}, \\mathrm{S}$", "$\\mathrm{I} \\rightarrow \\mathrm{Q}, \\mathrm{S}$; II $\\rightarrow \\mathrm{P}, \\mathrm{T}$; III $\\rightarrow \\mathrm{P}$; IV $\\rightarrow \\mathrm{S}$", "$\\mathrm{I} \\rightarrow \\mathrm{P}, \\mathrm{S}$; II $\\rightarrow \\mathrm{T}$; III $\\rightarrow \\mathrm{Q}, \\mathrm{R}$; IV $\\rightarrow \\mathrm{P}$"], "answer": 3, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (D). Source: JEE Adv 2022 Paper 1.", "source": "JEE Adv 2022 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-2-math-15", "chapter": "Permutations and Combinations", "b": 1.8, "q": "Consider 4 boxes, where each box contains 3 red balls and 2 blue balls. Assume that all 20 balls are distinct. In how many different ways can 10 balls be chosen from these 4 boxes so that from each box at least one red ball and one blue ball are chosen?", "options": ["21816", "85536", "12096", "156816"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2022 Paper 2.", "source": "JEE Adv 2022 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-2-math-17", "chapter": "Probability", "b": 1.8, "q": "Suppose that\n\nBox-I contains 8 red, 3 blue and 5 green balls,\n\nBox-II contains 24 red, 9 blue and 15 green balls,\n\nBox-III contains 1 blue, 12 green and 3 yellow balls,\n\nBox-IV contains 10 green, 16 orange and 6 white balls.\n\nA ball is chosen randomly from Box-I; call this ball $b$. If $b$ is red then a ball is chosen randomly from Box-II, if $b$ is blue then a ball is chosen randomly from Box-III, and if $b$ is green then a ball is chosen randomly from Box-IV. The conditional probability of the event 'one of the chosen balls is white' given that the event 'at least one of the chosen balls is green' has happened, is equal to", "options": ["$\\frac{15}{256}$", "$\\frac{3}{16}$", "$\\frac{5}{52}$", "$\\frac{1}{8}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2022 Paper 2.", "source": "JEE Adv 2022 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-2-math-18", "chapter": "Limits", "b": 1.8, "q": "For positive integer $n$, define\n\n\\[\nf(n)=n+\\frac{16+5 n-3 n^{2}}{4 n+3 n^{2}}+\\frac{32+n-3 n^{2}}{8 n+3 n^{2}}+\\frac{48-3 n-3 n^{2}}{12 n+3 n^{2}}+\\cdots+\\frac{25 n-7 n^{2}}{7 n^{2}}\n\\]\n\nThen, the value of $\\lim _{n \\rightarrow \\infty} f(n)$ is equal to", "options": ["$3+\\frac{4}{3} \\log _{e} 7$", "$4-\\frac{3}{4} \\log _{e}\\left(\\frac{7}{3}\\right)$", "$4-\\frac{4}{3} \\log _{e}\\left(\\frac{7}{3}\\right)$", "$3+\\frac{3}{4} \\log _{e} 7$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2022 Paper 2.", "source": "JEE Adv 2022 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-2-phy-34", "chapter": "Dual Nature of Matter and Radiation", "b": 1.8, "q": "When light of a given wavelength is incident on a metallic surface, the minimum potential needed to stop the emitted photoelectrons is $6.0 \\mathrm{~V}$. This potential drops to $0.6 \\mathrm{~V}$ if another source with wavelength four times that of the first one and intensity half of the first one is used. What are the wavelength of the first source and the work function of the metal, respectively? [Take $\\frac{h c}{e}=1.24 \\times$ $10^{-6} \\mathrm{~J} \\mathrm{mC}^{-1}$.]", "options": ["$1.72 \\times 10^{-7} \\mathrm{~m}, 1.20 \\mathrm{eV}$", "$1.72 \\times 10^{-7} \\mathrm{~m}, 5.60 \\mathrm{eV}$", "$3.78 \\times 10^{-7} \\mathrm{~m}, 5.60 \\mathrm{eV}$", "$3.78 \\times 10^{-7} \\mathrm{~m}, 1.20 \\mathrm{eV}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2022 Paper 2.", "source": "JEE Adv 2022 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-2-chem-51", "chapter": "Solid State", "b": 1.8, "q": "Atom $\\mathrm{X}$ occupies the fcc lattice sites as well as alternate tetrahedral voids of the same lattice. The packing efficiency (in \\%) of the resultant solid is closest to", "options": ["25", "35", "55", "75"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2022 Paper 2.", "source": "JEE Adv 2022 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-2-chem-52", "chapter": "p-Block Elements", "b": 1.8, "q": "The reaction of $\\mathrm{HClO}_{3}$ with $\\mathrm{HCl}$ gives a paramagnetic gas, which upon reaction with $\\mathrm{O}_{3}$ produces", "options": ["$\\mathrm{Cl}_{2} \\mathrm{O}$", "$\\mathrm{ClO}_{2}$", "$\\mathrm{Cl}_{2} \\mathrm{O}_{6}$", "$\\mathrm{Cl}_{2} \\mathrm{O}_{7}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2022 Paper 2.", "source": "JEE Adv 2022 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2022-paper-2-chem-53", "chapter": "p-Block Elements", "b": 1.8, "q": "The reaction of $\\mathrm{Pb}\\left(\\mathrm{NO}_{3}\\right)_{2}$ and $\\mathrm{NaCl}$ in water produces a precipitate that dissolves upon the addition of $\\mathrm{HCl}$ of appropriate concentration. The dissolution of the precipitate is due to the formation of", "options": ["$\\mathrm{PbCl}_{2}$", "$\\mathrm{PbCl}_{4}$", "$\\left[\\mathrm{PbCl}_{4}\\right]^{2-}$", "$\\left[\\mathrm{PbCl}_{6}\\right]^{2-}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2022 Paper 2.", "source": "JEE Adv 2022 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-math-4", "chapter": "Limits", "b": 1.8, "q": "Let $f:(0,1) \\rightarrow \\mathbb{R}$ be the function defined as $f(x)=\\sqrt{n}$ if $x \\in\\left[\\frac{1}{n+1}, \\frac{1}{n}\\right)$ where $n \\in \\mathbb{N}$. Let $g:(0,1) \\rightarrow \\mathbb{R}$ be a function such that $\\int_{x^{2}}^{x} \\sqrt{\\frac{1-t}{t}} d t<g(x)<2 \\sqrt{x}$ for all $x \\in(0,1)$. Then $\\lim _{x \\rightarrow 0} f(x) g(x)$", "options": ["does NOT exist", "is equal to 1", "is equal to 2", "is equal to 3"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-math-5", "chapter": "3D Geometry", "b": 1.8, "q": "Let $Q$ be the cube with the set of vertices $\\left\\{\\left(x_{1}, x_{2}, x_{3}\\right) \\in \\mathbb{R}^{3}: x_{1}, x_{2}, x_{3} \\in\\{0,1\\}\\right\\}$. Let $F$ be the set of all twelve lines containing the diagonals of the six faces of the cube $Q$. Let $S$ be the set of all four lines containing the main diagonals of the cube $Q$; for instance, the line passing through the vertices $(0,0,0)$ and $(1,1,1)$ is in $S$. For lines $\\ell_{1}$ and $\\ell_{2}$, let $d\\left(\\ell_{1}, \\ell_{2}\\right)$ denote the shortest distance between them. Then the maximum value of $d\\left(\\ell_{1}, \\ell_{2}\\right)$, as $\\ell_{1}$ varies over $F$ and $\\ell_{2}$ varies over $S$, is", "options": ["$\\frac{1}{\\sqrt{6}}$", "$\\frac{1}{\\sqrt{8}}$", "$\\frac{1}{\\sqrt{3}}$", "$\\frac{1}{\\sqrt{12}}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-math-6", "chapter": "Probability", "b": 1.8, "q": "Let $X=\\left\\{(x, y) \\in \\mathbb{Z} \\times \\mathbb{Z}: \\frac{x^{2}}{8}+\\frac{y^{2}}{20}<1\\right.$ and $\\left.y^{2}<5 x\\right\\}$. Three distinct points $P, Q$ and $R$ are randomly chosen from $X$. Then the probability that $P, Q$ and $R$ form a triangle whose area is a positive integer, is", "options": ["$\\frac{71}{220}$", "$\\frac{73}{220}$", "$\\frac{79}{220}$", "$\\frac{83}{220}$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-math-7", "chapter": "Parabola", "b": 1.8, "q": "Let $P$ be a point on the parabola $y^{2}=4 a x$, where $a>0$. The normal to the parabola at $P$ meets the $x$-axis at a point $Q$. The area of the triangle $P F Q$, where $F$ is the focus of the parabola, is 120 . If the slope $m$ of the normal and $a$ are both positive integers, then the pair $(a, m)$ is", "options": ["$(2,3)$", "$(1,3)$", "$(2,4)$", "$(3,4)$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-phy-21", "chapter": "Rotational Motion", "b": 1.8, "q": "A bar of mass $M=1.00 \\mathrm{~kg}$ and length $L=0.20 \\mathrm{~m}$ is lying on a horizontal frictionless surface. One end of the bar is pivoted at a point about which it is free to rotate. A small mass $m=0.10 \\mathrm{~kg}$ is moving on the same horizontal surface with $5.00 \\mathrm{~m} \\mathrm{~s}^{-1}$ speed on a path perpendicular to the bar. It hits the bar at a distance $L / 2$ from the pivoted end and returns back on the same path with speed $\\mathrm{v}$. After this elastic collision, the bar rotates with an angular velocity $\\omega$. Which of the following statement is correct?", "options": ["$\\omega=6.98 \\mathrm{rad} \\mathrm{s}^{-1}$ and $\\mathrm{v}=4.30 \\mathrm{~m} \\mathrm{~s}^{-1}$", "$\\omega=3.75 \\mathrm{rad} \\mathrm{s}^{-1}$ and $\\mathrm{v}=4.30 \\mathrm{~m} \\mathrm{~s}^{-1}$", "$\\omega=3.75 \\mathrm{rad} \\mathrm{s}^{-1}$ and $\\mathrm{v}=10.0 \\mathrm{~m} \\mathrm{~s}^{-1}$", "$\\omega=6.80 \\mathrm{rad} \\mathrm{s}^{-1}$ and $\\mathrm{v}=4.10 \\mathrm{~m} \\mathrm{~s}^{-1}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-phy-23", "chapter": "Thermodynamics", "b": 1.8, "q": "One mole of an ideal gas expands adiabatically from an initial state $\\left(T_{\\mathrm{A}}, V_{0}\\right)$ to final state $\\left(T_{\\mathrm{f}}, 5 V_{0}\\right)$. Another mole of the same gas expands isothermally from a different initial state $\\left(T_{\\mathrm{B}}, V_{0}\\right)$ to the same final state $\\left(T_{\\mathrm{f}}, 5 V_{0}\\right)$. The ratio of the specific heats at constant pressure and constant volume of this ideal gas is $\\gamma$. What is the ratio $T_{\\mathrm{A}} / T_{\\mathrm{B}}$ ?", "options": ["$5^{\\gamma-1}$", "$5^{1-\\gamma}$", "$5^{\\gamma}$", "$5^{1+\\gamma}$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-phy-24", "chapter": "Gravitation", "b": 1.8, "q": "Two satellites $\\mathrm{P}$ and $\\mathrm{Q}$ are moving in different circular orbits around the Earth (radius $R$ ). The heights of $\\mathrm{P}$ and $\\mathrm{Q}$ from the Earth surface are $h_{\\mathrm{P}}$ and $h_{\\mathrm{Q}}$, respectively, where $h_{\\mathrm{P}}=R / 3$. The accelerations of $\\mathrm{P}$ and $\\mathrm{Q}$ due to Earth's gravity are $g_{\\mathrm{P}}$ and $g_{\\mathrm{Q}}$, respectively. If $g_{\\mathrm{P}} / g_{\\mathrm{Q}}=36 / 25$, what is the value of $h_{\\mathrm{Q}}$ ?", "options": ["$3 R / 5$", "$R / 6$", "$6 R / 5$", "$5 R / 6$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-chem-39", "chapter": "Electrochemistry", "b": 1.8, "q": "Plotting $1 / \\Lambda_{\\mathrm{m}}$ against $\\mathrm{c} \\Lambda_{\\mathrm{m}}$ for aqueous solutions of a monobasic weak acid (HX) resulted in a straight line with y-axis intercept of $\\mathrm{P}$ and slope of $\\mathrm{S}$. The ratio $\\mathrm{P} / \\mathrm{S}$ is\n\n\n\n$\\left[\\Lambda_{\\mathrm{m}}=\\right.$ molar conductivity\n\n\n\n$\\Lambda_{\\mathrm{m}}^{\\mathrm{o}}=$ limiting molar conductivity\n\n\n\n$\\mathrm{c}=$ molar concentration\n\n\n\n$\\mathrm{K}_{\\mathrm{a}}=$ dissociation constant of $\\mathrm{HX}$ ]", "options": ["$\\mathrm{K}_{\\mathrm{a}} \\Lambda_{\\mathrm{m}}^{\\mathrm{o}}$", "$\\mathrm{K}_{\\mathrm{a}} \\Lambda_{\\mathrm{m}}^{\\mathrm{o}} / 2$", "$2 \\mathrm{~K}_{\\mathrm{a}} \\Lambda_{\\mathrm{m}}^{\\mathrm{o}}$", "$1 /\\left(\\mathrm{K}_{\\mathrm{a}} \\Lambda_{\\mathrm{m}}^{\\mathrm{o}}\\right)$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-1-chem-40", "chapter": "Ionic Equilibrium", "b": 1.8, "q": "On decreasing the $p \\mathrm{H}$ from 7 to 2, the solubility of a sparingly soluble salt (MX) of a weak acid (HX) increased from $10^{-4} \\mathrm{~mol} \\mathrm{~L}^{-1}$ to $10^{-3} \\mathrm{~mol} \\mathrm{~L}^{-1}$. The $p\\mathrm{~K}_{\\mathrm{a}}$ of $\\mathrm{HX}$ is", "options": ["3", "4", "5", "2"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2023 Paper 1.", "source": "JEE Adv 2023 Paper 1 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-2-math-1", "chapter": "Application of Derivatives", "b": 1.8, "q": "Let $f:[1, \\infty) \\rightarrow \\mathbb{R}$ be a differentiable function such that $f(1)=\\frac{1}{3}$ and $3 \\int_{1}^{x} f(t) d t=x f(x)-\\frac{x^{3}}{3}, x \\in[1, \\infty)$. Let $e$ denote the base of the natural logarithm. Then the value of $f(e)$ is", "options": ["$\\frac{e^{2}+4}{3}$", "$\\frac{\\log _{e} 4+e}{3}$", "$\\frac{4 e^{2}}{3}$", "$\\frac{e^{2}-4}{3}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2023 Paper 2.", "source": "JEE Adv 2023 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-2-math-2", "chapter": "Probability", "b": 1.8, "q": "Consider an experiment of tossing a coin repeatedly until the outcomes of two consecutive tosses are same. If the probability of a random toss resulting in head is $\\frac{1}{3}$, then the probability that the experiment stops with head is", "options": ["$\\frac{1}{3}$", "$\\frac{5}{21}$", "$\\frac{4}{21}$", "$\\frac{2}{7}$"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2023 Paper 2.", "source": "JEE Adv 2023 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-2-math-3", "chapter": "Inverse Trigonometric Functions", "b": 1.8, "q": "For any $y \\in \\mathbb{R}$, let $\\cot ^{-1}(y) \\in(0, \\pi)$ and $\\tan ^{-1}(y) \\in\\left(-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right)$. Then the sum of all the solutions of the equation $\\tan ^{-1}\\left(\\frac{6 y}{9-y^{2}}\\right)+\\cot ^{-1}\\left(\\frac{9-y^{2}}{6 y}\\right)=\\frac{2 \\pi}{3}$ for $0<|y|<3$, is equal to", "options": ["$2 \\sqrt{3}-3$", "$3-2 \\sqrt{3}$", "$4 \\sqrt{3}-6$", "$6-4 \\sqrt{3}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2023 Paper 2.", "source": "JEE Adv 2023 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-2-math-4", "chapter": "Vectors", "b": 1.8, "q": "Let the position vectors of the points $P, Q, R$ and $S$ be $\\vec{a}=\\hat{i}+2 \\hat{j}-5 \\hat{k}, \\vec{b}=3 \\hat{i}+6 \\hat{j}+3 \\hat{k}$, $\\vec{c}=\\frac{17}{5} \\hat{i}+\\frac{16}{5} \\hat{j}+7 \\hat{k}$ and $\\vec{d}=2 \\hat{i}+\\hat{j}+\\hat{k}$, respectively. Then which of the following statements is true?", "options": ["The points $P, Q, R$ and $S$ are NOT coplanar", "$\\frac{\\vec{b}+2 \\vec{d}}{3}$ is the position vector of a point which divides $P R$ internally in the ratio $5: 4$", "$\\frac{\\vec{b}+2 \\vec{d}}{3}$ is the position vector of a point which divides $P R$ externally in the ratio $5: 4$", "The square of the magnitude of the vector $\\vec{b} \\times \\vec{d}$ is 95"], "answer": 1, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (B). Source: JEE Adv 2023 Paper 2.", "source": "JEE Adv 2023 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-2-phy-19", "chapter": "Units and Measurements", "b": 1.8, "q": "Young's modulus of elasticity $Y$ is expressed in terms of three derived quantities, namely, the gravitational constant $G$, Planck's constant $h$ and the speed of light $c$, as $Y=c^{\\alpha} h^{\\beta} G^{\\gamma}$. Which of the following is the correct option?", "options": ["$\\alpha=7, \\beta=-1, \\gamma=-2$", "$\\alpha=-7, \\beta=-1, \\gamma=-2$", "$\\alpha=7, \\beta=-1, \\gamma=2$", "$\\alpha=-7, \\beta=1, \\gamma=-2$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 2.", "source": "JEE Adv 2023 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-2-phy-20", "chapter": "Kinematics", "b": 1.8, "q": "A particle of mass $m$ is moving in the $x y$-plane such that its velocity at a point $(x, y)$ is given as $\\overrightarrow{\\mathrm{v}}=\\alpha(y \\hat{x}+2 x \\hat{y})$, where $\\alpha$ is a non-zero constant. What is the force $\\vec{F}$ acting on the particle?", "options": ["$\\vec{F}=2 m \\alpha^{2}(x \\hat{x}+y \\hat{y})$", "$\\vec{F}=m \\alpha^{2}(y \\hat{x}+2 x \\hat{y})$", "$\\vec{F}=2 m \\alpha^{2}(y \\hat{x}+x \\hat{y})$", "$\\vec{F}=m \\alpha^{2}(x \\hat{x}+2 y \\hat{y})$"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 2.", "source": "JEE Adv 2023 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-2-phy-21", "chapter": "Kinetic Theory of Gases", "b": 1.8, "q": "An ideal gas is in thermodynamic equilibrium. The number of degrees of freedom of a molecule of the gas is $n$. The internal energy of one mole of the gas is $U_{n}$ and the speed of sound in the gas is $\\mathrm{v}_{n}$. At a fixed temperature and pressure, which of the following is the correct option?", "options": ["$v_{3}<v_{6}$ and $U_{3}>U_{6}$", "$v_{5}>v_{3}$ and $U_{3}>U_{5}$", "$v_{5}>v_{7}$ and $U_{5}<U_{7}$", "$v_{6}<v_{7}$ and $U_{6}<U_{7}$"], "answer": 2, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (C). Source: JEE Adv 2023 Paper 2.", "source": "JEE Adv 2023 Paper 2 (official key, via JEEBench)", "key_source": "official"},
{"id": "jee-2023-paper-2-chem-36", "chapter": "Surface Chemistry", "b": 1.8, "q": "Consider the following statements related to colloids.\n\n\n\n(I) Lyophobic colloids are not formed by simple mixing of dispersed phase and dispersion medium.\n\n\n\n(II) For emulsions, both the dispersed phase and the dispersion medium are liquid.\n\n\n\n(III) Micelles are produced by dissolving a surfactant in any solvent at any temperature.\n\n\n\n(IV) Tyndall effect can be observed from a colloidal solution with dispersed phase having the same refractive index as that of the dispersion medium.\n\n\n\nThe option with the correct set of statements is", "options": ["(I) and (II)", "(II) and (III)", "(III) and (IV)", "(II) and (IV)"], "answer": 0, "hints": ["Identify the core principle the question is testing.", "Check each option against a limiting or special case."], "explanation": "Official JEE Advanced answer: (A). Source: JEE Adv 2023 Paper 2.", "source": "JEE Adv 2023 Paper 2 (official key, via JEEBench)", "key_source": "official"}
]
'''

# ---------------- index.html ----------------
FILES['index.html'] = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PrepPilot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --bg: #f5f4ef; --panel: #ffffff; --ink: #17191e; --muted: #5f6571; --line: #e3e1d9;
  --chip: #eeece5; --accent: #d9481f; --accent-ink: #ffffff; --good: #1d8348; --bad: #c0392b;
  --warn: #a8690f; --bot: #ffffff; --user: #17191e; --user-ink: #ffffff; --sel: #fde9e1;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --panel: #171a21; --ink: #e8e9ed; --muted: #9ba2ae; --line: #2a2f39;
    --chip: #222731; --accent: #ff6a3d; --accent-ink: #14100e; --good: #3ccf7a; --bad: #ff6b5e;
    --warn: #f0b44c; --bot: #1b1f27; --user: #e8e9ed; --user-ink: #111317; --sel: #3a2219;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; overflow: hidden; }
body { background: var(--bg); color: var(--ink); font: 15px/1.5 "Instrument Sans", system-ui, sans-serif; display: flex; flex-direction: column; }
button, select, input { font: inherit; color: inherit; }
button { cursor: pointer; }
h1, h2, h3 { font-family: "Bricolage Grotesque", "Instrument Sans", sans-serif; margin: 0; }

header { display: flex; align-items: center; gap: 12px; padding: 10px 16px; border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: wrap; }
header h1 { font-size: 20px; letter-spacing: -0.02em; }
header h1 span { color: var(--accent); }
select { background: var(--chip); border: 0; border-radius: 10px; padding: 7px 10px; font-size: 14px; font-weight: 600; max-width: 260px; }
a.home { text-decoration: none; color: inherit; }
.user { position: relative; }
.avatar { width: 34px; height: 34px; border-radius: 50%; border: 0; background: var(--accent); color: var(--accent-ink);
          font-weight: 700; cursor: pointer; }
.usermenu { position: absolute; right: 0; top: 42px; z-index: 30; min-width: 220px; display: flex; flex-direction: column; gap: 4px;
            background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px; box-shadow: 0 12px 32px rgba(0,0,0,.18); }
.usermenu[hidden] { display: none; }
.usermenu .btn { margin-top: 8px; text-align: center; text-decoration: none; color: inherit; }
.pill { font-size: 12px; padding: 4px 10px; border-radius: 99px; background: var(--chip); color: var(--muted); margin-left: auto; }
.pill.on { color: var(--good); }
#sidebtn { display: none; }
.pill.clock { font-variant-numeric: tabular-nums; color: var(--ink); font-weight: 600; }
.pill.clock.due { background: var(--sel); color: var(--accent); }
#breakov { position: fixed; inset: 0; z-index: 50; display: grid; place-items: center; padding: 16px;
           background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(6px); }
#breakov[hidden] { display: none; }
.brkcard { width: min(440px, 100%); background: var(--panel); border: 1px solid var(--line); border-radius: 20px;
           padding: 28px 24px; text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,.25); }
.brkicon { font-size: 40px; }
.brkcard h2 { font-size: 26px; margin: 4px 0 6px; }
.brkleft { font-family: "Bricolage Grotesque", sans-serif; font-size: 64px; font-weight: 700; font-variant-numeric: tabular-nums; margin: 8px 0; }
.brktrack { height: 8px; border-radius: 99px; background: var(--chip); overflow: hidden; }
#brkbar { height: 100%; width: 0; background: var(--good); transition: width .25s linear; }
#brktips { text-align: left; font-size: 14px; margin: 14px 0 6px; padding-left: 20px; }
#brktips li { margin: 4px 0; }
.timed { font-size: 13px; font-weight: 600; display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }
.pill.pred { margin-left: auto; color: var(--ink); font-weight: 600; }
.pill.pred:not([hidden]) + .pill { margin-left: 0; }
.tbar { height: 6px; border-radius: 99px; background: var(--chip); overflow: hidden; margin: 2px 0 4px; }
.tbar div { height: 100%; background: var(--good); transition: width .25s linear; }
.tbar div.low { background: var(--bad); }
.tleft { font-size: 12px; color: var(--muted); }
.diag { display: flex; flex-direction: column; gap: 2px; margin: 4px 0 8px; padding: 8px 10px; border-radius: 8px; border-left: 4px solid var(--warn); background: var(--chip); font-size: 14px; }
.diag.concept_gap { border-color: var(--bad); } .diag.trap { border-color: var(--accent); } .diag.time_pressure { border-color: #6c8cff; }
.msg.care { border-left-color: #6c8cff; }
.predict .big { font-size: 34px; }
.predict div:not(.big) { font-size: 13px; color: var(--muted); }
.predict .up { color: var(--good); }
.chartbox.small { height: 120px; }
.focusrow { display: flex; flex-direction: column; gap: 2px; font-size: 13px; padding: 8px 10px; border-radius: 8px; background: var(--chip); }
.focusrow span { color: var(--muted); }
.fill.err-concept_gap { background: var(--bad); } .fill.err-careless { background: var(--warn); }
.fill.err-trap { background: var(--accent); } .fill.err-time_pressure { background: #6c8cff; }
table.calib { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.calib th { text-align: left; color: var(--muted); font-weight: 600; padding: 4px; }
.calib td { padding: 5px 4px; border-top: 1px solid var(--line); vertical-align: top; }

main { flex: 1; display: grid; grid-template-columns: 270px minmax(0, 1fr) 420px; grid-template-rows: minmax(0, 1fr); min-height: 0; overflow: hidden; }
#side { overflow-y: auto; border-right: 1px solid var(--line); background: var(--panel); padding: 12px 10px 24px; }
#side .all { width: 100%; text-align: left; }
#side details { margin-top: 10px; }
#side summary { cursor: pointer; display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 8px; font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 15px; list-style: none; }
#side summary::-webkit-details-marker { display: none; }
#side summary::before { content: "▸"; color: var(--muted); font-size: 12px; transition: transform .15s; }
#side details[open] summary::before { transform: rotate(90deg); }
#side summary .pct { margin-left: auto; font-family: "Instrument Sans", sans-serif; font-size: 12px; color: var(--muted); font-weight: 600; }
#side summary .focusbtn { border: 0; background: var(--chip); border-radius: 6px; font-size: 11px; padding: 2px 6px; color: var(--muted); }
.chap { width: 100%; display: flex; align-items: center; gap: 8px; text-align: left; border: 0; background: none; padding: 5px 8px 5px 22px; border-radius: 8px; font-size: 13.5px; line-height: 1.3; }
.chap:hover { background: var(--chip); }
.chap.on, #side summary.on, #side .all.on { background: var(--sel); }
.dot { width: 9px; height: 9px; border-radius: 50%; flex: none; background: var(--line); }
.chap .n { margin-left: auto; font-size: 11px; color: var(--muted); }
#chat { display: flex; flex-direction: column; min-height: 0; border-right: 1px solid var(--line); }
#log { flex: 1; overflow-y: auto; padding: 20px 16px; display: flex; flex-direction: column; gap: 12px; }
.msg { max-width: 720px; padding: 12px 14px; border-radius: 14px; background: var(--bot); border: 1px solid var(--line); white-space: pre-wrap; overflow-wrap: anywhere; }
.msg.user { align-self: flex-end; background: var(--user); color: var(--user-ink); border-color: transparent; }
.msg.muted { color: var(--muted); font-style: italic; }
.msg.coach { border-left: 4px solid var(--accent); }
.card { white-space: normal; width: 100%; }
.meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.tag { font-size: 11.5px; font-weight: 600; padding: 2px 8px; border-radius: 99px; background: var(--chip); color: var(--muted); }
.tag.Easy { color: var(--good); } .tag.Medium { color: var(--warn); } .tag.Advanced { color: var(--bad); }
.tag.ai { color: var(--accent); }
.qtext { font-size: 16px; margin-bottom: 12px; }
.opts { display: grid; gap: 8px; }
.opt { text-align: left; border: 1px solid var(--line); background: var(--panel); border-radius: 10px; padding: 10px 12px; display: flex; gap: 10px; align-items: baseline; transition: border-color .15s, background .15s; }
.opt:hover:not(:disabled) { border-color: var(--accent); }
.opt b { font-family: "Bricolage Grotesque", sans-serif; color: var(--muted); }
.opt.picked { border-color: var(--accent); background: var(--sel); }
.opt.right { border-color: var(--good); background: color-mix(in srgb, var(--good) 14%, var(--panel)); }
.opt.wrong { border-color: var(--bad); background: color-mix(in srgb, var(--bad) 14%, var(--panel)); }
.opt:disabled { cursor: default; }
.card button:disabled:not(.right):not(.wrong):not(.picked) { opacity: .55; }
.conf { display: none; margin-top: 12px; align-items: center; gap: 8px; flex-wrap: wrap; }
.conf.show { display: flex; }
.conf span { font-size: 13px; color: var(--muted); }
.btn { border: 1px solid var(--line); background: var(--panel); border-radius: 99px; padding: 6px 14px; font-size: 13px; font-weight: 600; }
.btn:hover { border-color: var(--accent); }
.btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.actions { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
.hintline { margin-top: 10px; padding: 8px 10px; border-radius: 8px; background: var(--chip); font-size: 14px; }
.result h3 { font-size: 16px; margin-bottom: 6px; }
.result .ok { color: var(--good); } .result .no { color: var(--bad); }
.engine { margin-top: 10px; font-size: 13px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.up { color: var(--good); font-weight: 600; } .down { color: var(--bad); font-weight: 600; }

form { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--line); background: var(--panel); }
form input { flex: 1; min-width: 0; border: 1px solid var(--line); background: var(--bg); border-radius: 99px; padding: 10px 16px; }

aside { display: flex; flex-direction: column; min-height: 0; background: var(--panel); }
.tabs { display: flex; border-bottom: 1px solid var(--line); }
.tabs button { flex: 1; border: 0; background: none; padding: 12px 0; font-weight: 600; color: var(--muted); border-bottom: 2px solid transparent; }
.tabs button.on { color: var(--ink); border-color: var(--accent); }
.panel { display: none; overflow-y: auto; padding: 16px; flex: 1; }
.panel.on { display: block; }
.panel h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 18px 0 8px; }
.panel h2:first-child { margin-top: 0; }
.big { font-family: "Bricolage Grotesque", sans-serif; font-size: 22px; font-weight: 700; line-height: 1.2; }
.reason { margin: 8px 0 4px; font-size: 14px; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.stat { background: var(--chip); border-radius: 10px; padding: 8px 10px; }
.stat small { display: block; font-size: 11px; color: var(--muted); }
.stat strong { font-family: "Bricolage Grotesque", sans-serif; font-size: 18px; }
.bar { display: grid; grid-template-columns: 110px 1fr 44px; gap: 8px; align-items: center; font-size: 13px; margin: 5px 0; }
.bar .track { height: 8px; border-radius: 99px; background: var(--chip); overflow: hidden; }
.bar .fill { height: 100%; border-radius: 99px; background: var(--accent); transition: width .4s; }
.bar .fill.neg { background: var(--muted); }
.cands li { display: flex; justify-content: space-between; font-size: 13px; padding: 3px 0; border-bottom: 1px dashed var(--line); }
.cands li:first-child { font-weight: 600; }
ol.cands { padding-left: 0; list-style: none; margin: 0; }
.map h3 { font-size: 13px; margin: 10px 0 6px; }
.tiles { display: flex; flex-wrap: wrap; gap: 4px; }
.tile { width: 22px; height: 22px; border-radius: 5px; border: 0; padding: 0; }
.tile.prior { opacity: .35; }
.tile.focus { outline: 2px solid var(--ink); outline-offset: 1px; }
.tile.flag { box-shadow: inset 0 0 0 2px var(--bad); }
.legend { font-size: 12px; color: var(--muted); margin-top: 6px; }
.weak li { font-size: 13px; margin: 4px 0; }
.empty { color: var(--muted); font-size: 14px; }
.chartbox { position: relative; height: 220px; }
.headline { font-family: "Bricolage Grotesque", sans-serif; font-size: 18px; margin: 12px 0; }

@media (max-width: 1200px) {
  main { grid-template-columns: minmax(0, 1fr) 400px; }
  #sidebtn { display: inline-block; }
  #side { display: none; position: fixed; z-index: 10; top: 56px; left: 0; bottom: 0; width: min(320px, 88vw); box-shadow: 4px 0 24px rgba(0,0,0,.25); }
  body.side-open #side { display: block; }
}
@media (max-width: 900px) {
  html, body { height: auto; overflow: visible; }
  main { grid-template-columns: 1fr; grid-template-rows: 75vh auto; overflow: visible; }
  #chat { border-right: 0; border-bottom: 1px solid var(--line); }
  aside { max-height: none; }
  .pill { margin-left: 0; }
}
</style>
</head>
<body>
<header>
  <h1><a href="/" class="home">Prep<span>Pilot</span></a></h1>
  <select id="level" aria-label="Class or exam">
    <optgroup label="School">
      <option>Class 1</option><option>Class 2</option><option>Class 3</option><option>Class 4</option>
      <option>Class 5</option><option>Class 6</option><option>Class 7</option><option>Class 8</option>
      <option>Class 9</option><option>Class 10</option><option>Class 11</option><option>Class 12</option>
    </optgroup>
    <optgroup label="Competitive exams">
      <option value="JEE">JEE Advanced</option><option value="NEET">NEET</option>
    </optgroup>
  </select>
  <button class="btn" id="sidebtn" aria-controls="side">☰ Chapters</button>
  <label class="timed" title="Exam mode: each question has a countdown"><input type="checkbox" id="timed"> ⏱ Timed</label>
  <span class="pill clock" id="clock" title="40-minute focus blocks, then a mandatory break">🎯 Focus</span>
  <button class="btn" id="breaknow" title="Start your break early">☕ Break</button>
  <span class="pill pred" id="pred" hidden></span>
  <span class="pill" id="ai">AI: checking</span>
  <div class="user">
    <button class="avatar" id="userbtn" aria-haspopup="menu" aria-expanded="false" aria-label="Account">?</button>
    <div class="usermenu" id="usermenu" role="menu" hidden>
      <strong id="uname"></strong>
      <span id="uemail" class="legend"></span>
      <a class="btn" href="/study" role="menuitem">Subjects &amp; chapters</a>
      <a class="btn" href="/setup" role="menuitem">⚙ Preferences</a>
      <button class="btn" id="logout" role="menuitem">Log out</button>
    </div>
  </div>
</header>
<main>
  <nav id="side" aria-label="Subjects and chapters"></nav>
  <section id="chat">
    <div id="log" aria-live="polite"></div>
    <form id="ask">
      <input id="input" autocomplete="off" placeholder="Ask a doubt, or type /hint  /skip  /focus Physics  /dashboard">
      <button class="btn primary">Send</button>
    </form>
  </section>
  <aside>
    <div class="tabs">
      <button data-tab="why" class="on">Why this question</button>
      <button data-tab="dash">Dashboard</button>
      <button data-tab="sim">Simulation</button>
    </div>
    <div class="panel on" id="why"><p class="empty">The engine's reasoning for each question shows here.</p></div>
    <div class="panel" id="dash"><p class="empty">Answer a question to see your learner model.</p></div>
    <div class="panel" id="sim">
      <h2>Adaptive vs static question bank</h2>
      <p class="reason">Simulated students with hidden true ability in 6 chapters. Each question teaches a little, most when its difficulty matches the student. We count questions until every chapter is exam-ready.</p>
      <button class="btn primary" id="runsim">Run simulation (20 runs per profile)</button>
      <div class="headline" id="simhead"></div>
      <div class="chartbox"><canvas id="simchart"></canvas></div>
    </div>
  </aside>
</main>
<script>
const $ = s => document.querySelector(s);
const KATEX = { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }], throwOnError: false };
const S = { sid: null, exam: "JEE", examName: "", q: null, picked: null, busy: false, charts: {}, focus: "all" };
const pct = x => Math.round(x * 100) + "%";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) math(e, text);
  return e;
}
function math(e, text) {  // textContent first, so AI output can never inject HTML
  e.textContent = text;
  if (window.renderMathInElement) renderMathInElement(e, KATEX);
  return e;
}
async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  let r;
  try { r = await fetch(path, opts); }
  catch (e) { throw Object.assign(new Error("Can't reach the PrepPilot server. Check it's running, then try again."), { offline: true }); }
  const d = await r.json().catch(() => ({}));
  if (r.status === 404 && path !== "/api/me")  // server restarted: in-memory session is gone
    throw Object.assign(new Error("Your session expired because the server restarted."), { expired: true });
  if (r.status === 423) throw Object.assign(new Error("Break time"), { brk: d.detail });  // mandatory break
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d;
}
function problem(e, canRetry = false) {
  // friendly error with a one-click way out instead of a dead end
  const m = say("⚠ " + e.message);
  const fix = e.expired ? ["Start a fresh session", () => start(S.exam)]
            : (e.offline && canRetry) ? ["Try again", () => next()] : null;
  if (fix) {
    const b = el("button", "btn primary", fix[0]);
    b.onclick = () => { b.disabled = true; fix[1](); };
    m.append(el("br"), b);
  }
}

function push(node) {
  const log = $("#log");
  log.append(node);
  log.scrollTo({ top: log.scrollHeight, behavior: "smooth" });  // scroll the chat only, never the page
  return node;
}
function say(text, who = "bot", extra = "") { return push(el("div", `msg ${who} ${extra}`, text)); }

async function start(exam, fresh = false) {
  clearInterval(S.brkTimer);
  $("#breakov").hidden = true;
  S.blockEnd = 0;
  S.exam = exam; S.q = null; S.focus = "all"; S.busy = false;
  $("#log").innerHTML = "";
  $("#why").innerHTML = '<p class="empty">The engine\'s reasoning for each question shows here.</p>';
  $("#dash").innerHTML = '<p class="empty">Answer a question to see your learner model.</p>';
  $("#simhead").textContent = "";
  if (S.charts.sim) { S.charts.sim.destroy(); S.charts.sim = null; }
  const pr = S.prefs || {};
  const resumeKey = "pp-session:" + JSON.stringify([S.user?.email || "guest", exam, DEMO, pr]);
  let resumeId = "";
  try { resumeId = fresh ? "" : localStorage.getItem(resumeKey) || ""; } catch (e) {}
  const d = await api("/api/session", { resume_id: resumeId, level: exam, demo: DEMO,
                                        // setup choices; subjects only apply to the level they were picked for
                                        subjects: pr.level === exam ? pr.subjects || [] : [],
                                        start_ability: pr.start_ability ?? 0, block_minutes: pr.block_minutes || 40,
                                        explain_style: pr.explain_style || "steps" });
  try { localStorage.setItem(resumeKey, d.session_id); } catch (e) {}
  setClock(d.clock);
  S.sid = d.session_id; S.examName = d.exam; S.ai = d.ai; S.timeLimit = d.time_limit;
  clearInterval(S.timer);
  $("#pred").hidden = true;
  try { localStorage.setItem("pp-level", exam); } catch (e) {}
  $("#ai").textContent = d.ai ? "AI: Gemini on" : "AI: off (bank only)";
  $("#ai").classList.toggle("on", d.ai);
  buildSide(d.subjects);
  const names = d.subjects.map(s => s.subject);
  const first = S.user && !S.user.guest ? ` ${S.user.name.split(" ")[0]}` : "";
  say(`Hi${first}! I'm PrepPilot, your ${d.exam} practice partner.\n` +
      `Syllabus: ${d.chapters} chapters across ${names.join(", ")}. ` +
      (d.banked_chapters ? `Verified questions in ${d.banked_chapters} chapter${d.banked_chapters === 1 ? "" : "s"}` : "No verified questions at this level yet") +
      (d.ai ? `${d.banked_chapters ? "," : ";"} AI-generated (self-checked) questions everywhere else.` : ". Set GEMINI_API_KEY to unlock every chapter.") +
      `\nPick a subject or chapter on the left to focus, or let me choose. After each answer, tell me how sure you were.`);
  const requestedFocus = new URL(location.href).searchParams.get("focus");
  if (requestedFocus && fresh) {
    await api("/api/focus", {session_id: S.sid, focus: requestedFocus});
  }
  S.focus = requestedFocus && fresh ? requestedFocus : d.focus || "all";
  markFocus();
  await refreshDash();
  if (d.active) {
    renderQuestion(d.active.question, d.active.why, d.active.elapsed);
    renderWhy(d.active.why);
  } else next();
}

function buildSide(subjects) {
  const side = $("#side");
  side.innerHTML = "";
  const all = el("button", "btn all on", "All subjects (let PrepPilot choose)");
  all.dataset.focus = "all";
  all.onclick = () => setFocus("all");
  side.append(all);
  subjects.forEach(({ subject, chapters }) => {
    const det = el("details");
    det.open = subjects.length <= 2;
    const sum = el("summary");
    sum.dataset.focus = subject;
    const fb = el("button", "focusbtn", "Practise");
    fb.title = `Practise only ${subject}`;
    fb.onclick = e => { e.preventDefault(); setFocus(subject); };
    sum.append(el("span", "", subject), el("span", "pct", ""), fb);
    det.append(sum);
    chapters.forEach(c => {
      const b = el("button", "chap");
      b.dataset.focus = c;
      b.append(el("span", "dot"), el("span", "", c), el("span", "n", ""));
      b.onclick = () => setFocus(c);
      det.append(b);
    });
    side.append(det);
  });
}

function updateSide(d) {
  const byName = Object.fromEntries(d.chapters.map(c => [c.chapter, c]));
  document.querySelectorAll("#side .chap").forEach(b => {
    const c = byName[b.dataset.focus];
    if (!c) return;
    const dot = b.querySelector(".dot");
    dot.style.background = `hsl(${Math.round(c.readiness * 130)}, 62%, 46%)`;
    dot.style.opacity = c.n ? 1 : 0.35;
    b.querySelector(".n").textContent = c.n ? `${pct(c.readiness)}${c.flags ? " ⚠" : ""}` : "";
    b.title = `${c.chapter}: ${pct(c.readiness)} ready${c.n ? `, ${c.n} answered` : " (estimate)"}`;
  });
  const subj = Object.fromEntries(d.subjects.map(s => [s.subject, s.readiness]));
  document.querySelectorAll("#side summary").forEach(s => s.querySelector(".pct").textContent = pct(subj[s.dataset.focus] ?? 0));
}

function markFocus() {
  document.querySelectorAll("#side [data-focus]").forEach(b => b.classList.toggle("on", b.dataset.focus === S.focus));
}

async function next() {
  if (S.busy) return;
  S.busy = true;
  const sid = S.sid;  // the class may be switched while this request is in flight
  const wait = say("Picking your next question…", "bot", "muted");
  try {
    const d = await api("/api/next", { session_id: sid });
    wait.remove();
    if (sid !== S.sid) return;
    setClock(d.clock);
    renderQuestion(d.question, d.why);
    renderWhy(d.why);
  } catch (e) {
    wait.remove();
    if (sid !== S.sid) return;
    if (e.brk) showBreak(e.brk); else problem(e, true);
  }
  finally { if (sid === S.sid) S.busy = false; }
}

// ---------- study rhythm: 40-min focus blocks, mandatory 10–15 min breaks ----------
const DEMO = new URLSearchParams(location.search).has("demo");  // ?demo=1: 2-min blocks, 30-s breaks
const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function setClock(c) {
  S.blockEnd = Date.now() + c.block_left * 1000;
  S.breakEnd = Date.now() + c.break_left * 1000;
  S.blockSecs = c.block_secs;
  try { localStorage.setItem("pp-clock", JSON.stringify({ block_start: c.block_start, break_until: c.break_until, demo: DEMO })); } catch (e) {}
}

const TIPS = ["Stand up and stretch your back and shoulders.", "Drink a glass of water.",
  "Look at something 20 feet away for 20 seconds: your eyes need it.", "Walk around for a couple of minutes.",
  "Skip social media: it doesn't rest your brain.", "Take 5 slow, deep breaths."];

function showBreak(b) {
  S.q = null; clearInterval(S.timer);
  document.querySelectorAll("#log .card button").forEach(x => x.disabled = true);
  setClock(b);
  const ov = $("#breakov");
  ov.hidden = false;
  $("#brkwhy").textContent = b.long
    ? "You showed signs of fatigue in this block, so this break is longer."
    : b.reason === "Break taken early" ? "Good call. Rested brains learn faster." : `${mmss(b.block_secs)} of focused practice done.`;
  $("#brkstats").textContent = b.block_answered
    ? `This block: ${b.block_answered} questions, ${pct(b.block_accuracy)} correct.` : "";
  $("#brktips").innerHTML = "";
  [...TIPS].sort(() => Math.random() - .5).slice(0, 3).forEach(t => $("#brktips").append(el("li", "", t)));
  const total = Math.max(1, b.break_left);
  clearInterval(S.brkTimer);
  S.brkTimer = setInterval(() => {
    const left = Math.max(0, (S.breakEnd - Date.now()) / 1000);
    $("#brkleft").textContent = mmss(Math.ceil(left));
    $("#brkbar").style.width = (100 * (1 - left / total)) + "%";
    if (left <= 0) {
      clearInterval(S.brkTimer);
      ov.hidden = true;
      say(`Break over. New ${mmss(S.blockSecs)} focus block starts now.`, "bot", "coach");
      next();
    }
  }, 250);
}

setInterval(() => {  // header clock; when the block ends, the break starts after the current question
  if (!S.blockEnd || !$("#breakov").hidden) return;
  const left = (S.blockEnd - Date.now()) / 1000;
  $("#clock").textContent = left > 0 ? `🎯 Focus ${mmss(left)}` : S.q ? "☕ Break after this question" : "☕ Break time";
  $("#clock").classList.toggle("due", left <= 0);
  if (left <= 0 && !S.q && !S.busy && S.sid) next();
}, 1000);

function renderQuestion(q, why, elapsed = 0) {
  S.q = q; S.picked = null;
  const card = el("div", "msg card");
  const meta = el("div", "meta");
  meta.append(el("span", "tag", why.subject), el("span", "tag", q.chapter), el("span", "tag " + why.level, why.level));
  meta.append(q.official ? el("span", "tag Easy", `✓ Real past paper · ${q.source.split(" (")[0]}`)
    : q.verified ? el("span", "tag Easy", "✓ Verified key · past-paper pattern")
    : el("span", "tag ai", "AI-generated · self-checked"));
  if (why.retry) meta.append(el("span", "tag", "Follow-up after a miss"));
  const opts = el("div", "opts");
  q.options.forEach((o, i) => {
    const b = el("button", "opt");
    b.append(el("b", "", "ABCD"[i]), el("span", "", o));
    b.onclick = () => pick(card, i);
    opts.append(b);
  });
  const conf = el("div", "conf");
  conf.append(el("span", "", "How sure are you?"));
  [["sure", "🟢 Sure"], ["unsure", "🟡 Unsure"], ["guess", "🔴 Guessing"]].forEach(([v, label]) => {
    const b = el("button", "btn", label);
    b.onclick = () => submit(card, v);
    conf.append(b);
  });
  const acts = el("div", "actions");
  const hb = el("button", "btn", "💡 Hint (lowers credit)");
  hb.onclick = () => hint(card);
  const sk = el("button", "btn", "Skip");
  sk.onclick = async () => {
    if (S.q !== q) return;
    S.q = null; clearInterval(S.timer);
    card.querySelectorAll("button").forEach(b => b.disabled = true);
    await api("/api/skip", { session_id: S.sid }).catch(() => {});  // skips feed fatigue detection
    next();
  };
  acts.append(hb, sk);
  card.append(meta);
  clearInterval(S.timer);
  S.picks = []; S.limit = 0;
  if ($("#timed").checked) {  // exam mode: per-question countdown, auto-submits at zero
    S.limit = S.timeLimit;
    const bar = el("div", "tbar"), fill = el("div"), left = el("span", "tleft", `${S.limit}s`);
    bar.append(fill); card.append(bar, left);
    const end = Date.now() + Math.max(0, S.limit - elapsed) * 1000;
    S.timer = setInterval(() => {
      const s = Math.max(0, (end - Date.now()) / 1000);
      fill.style.width = (100 * s / S.limit) + "%";
      fill.classList.toggle("low", s < S.limit * 0.2);
      left.textContent = `${Math.ceil(s)}s left`;
      if (s <= 0) { clearInterval(S.timer); if (S.q === q) submit(card, S.picked === null ? "guess" : "unsure", true); }
    }, 250);
  }
  card.append(el("div", "qtext", q.q), opts, conf, acts);
  push(card);
}

function pick(card, i) {
  if (!S.q) return;
  S.picked = i;
  S.picks.push(i);
  card.querySelectorAll(".opt").forEach((b, j) => b.classList.toggle("picked", j === i));
  card.querySelector(".conf").classList.add("show");
}

async function hint(card) {
  if (!S.q) return;
  const d = await api("/api/hint", { session_id: S.sid });
  card.querySelector(".actions").before(el("div", "hintline", d.hint ? `Hint ${d.used}: ${d.hint}` : "No more hints. Give it your best shot!"));
}

async function submit(card, confidence, timedOut = false) {
  if (!S.q || (S.picked === null && !timedOut)) return;
  const choice = S.picked ?? -1, q = S.q; S.q = null;
  clearInterval(S.timer);
  card.querySelectorAll("button").forEach(b => b.disabled = true);
  card.querySelector(".conf").classList.remove("show");
  let d;
  try {
    d = await api("/api/answer", { session_id: S.sid, choice, confidence, picks: S.picks,
                                   time_limit: S.limit, timed_out: timedOut });
  } catch (e) {
    if (e.offline) {  // keep the question so the student can just press their confidence again
      S.q = q;
      card.querySelectorAll("button").forEach(b => b.disabled = false);
      card.querySelector(".conf").classList.add("show");
    }
    return problem(e);
  }
  const opts = card.querySelectorAll(".opt");
  opts[d.answer].classList.add("right");
  if (!d.correct && choice >= 0) opts[choice].classList.add("wrong");

  const r = el("div", "msg card result");
  r.append(el("h3", d.correct ? "ok" : "no", d.correct ? "✅ Correct"
    : timedOut && choice < 0 ? `⏱ Time's up. Answer: ${"ABCD"[d.answer]}` : `❌ Not quite. Answer: ${"ABCD"[d.answer]}`));
  if (d.diagnosis) {
    const dg = el("div", "diag " + d.diagnosis.type);
    dg.append(el("strong", "", d.diagnosis.label), el("span", "", d.diagnosis.fix));
    r.append(dg);
  }
  r.append(el("div", "", d.explanation));
  const eng = el("div", "engine");
  const ch = card.querySelector(".tag:nth-child(2)").textContent;
  const delta = el("span", d.delta >= 0 ? "up" : "down", `${ch}: ${pct(d.readiness_before)} → ${pct(d.readiness_after)} ready`);
  eng.append(delta, el("span", "", `· ${d.secs}s`));
  if (d.guess) eng.append(el("span", "tag", "Too fast: treated as a guess"));
  if (d.misconception) eng.append(el("span", "tag Advanced", "Confident but wrong: misconception flagged"));
  if (d.retry_next) eng.append(el("span", "tag", "Easier follow-up coming"));
  r.append(eng);
  const acts = el("div", "actions");
  const nb = el("button", "btn primary", "Next question →");
  nb.onclick = () => { nb.disabled = true; next(); };
  const rb = el("button", "btn", "⚑ Question is wrong");
  const answeredSession = S.sid;
  rb.onclick = async () => {
    rb.disabled = true;
    try {
      await api("/api/report", { session_id: answeredSession, answer_id: d.answer_id });
      rb.textContent = "Reported. This answer no longer counts.";
      if (answeredSession === S.sid) await refreshDash();
    } catch (e) { rb.disabled = false; problem(e); }
  };
  acts.append(nb, rb);
  r.append(acts);
  push(r);
  if (d.coach) say(d.coach, "bot", "coach");
  if (d.intervention) {
    const m = say((d.intervention.state === "frustrated" ? "💙 " : "☕ ") + d.intervention.message, "bot", "coach care");
    if (d.intervention.state === "tired") {
      const bb = el("button", "btn primary", "Take my break now");
      bb.onclick = () => { bb.disabled = true; api("/api/break", { session_id: S.sid }).catch(e => e.brk && showBreak(e.brk)); };
      m.append(el("br"), bb);
    }
  }
  refreshDash();
  if (!d.correct && S.ai) {
    const box = el("div", "hintline", "Looking at why you picked that…");
    acts.before(box);
    api("/api/explain", { session_id: S.sid })
      .then(x => x.explanation ? math(box, "🧠 " + x.explanation) : box.remove())
      .catch(() => box.remove());
  }
}

function renderWhy(w) {
  const p = $("#why");
  p.innerHTML = "";
  p.append(el("h2", "", "Chosen chapter"), el("div", "big", w.chapter));
  const parts = w.score_parts;
  const top = Object.entries(parts).filter(([k]) => k !== "repeat_penalty").sort((a, b) => b[1] - a[1])[0][0];
  let reason = w.boost ? `${w.boost}: an easier question (about ${pct(w.p)} chance) from one of your stronger chapters, to rebuild momentum.`
    : w.retry ? `Your last ${w.chapter} answer looked like ${w.last_error ? `a "${w.last_error.toLowerCase()}"` : "a gap"}, so this is an easier follow-up on the same concept.`
    : top === "misconception" ? `You were confidently wrong here before, so it gets extra priority.`
    : w.answered === 0 ? `You haven't practised this chapter yet. The starting estimate (${pct(w.readiness)} ready) comes from your ${w.subject} average, so I'm testing it.`
    : `This is one of your weakest areas for ${S.examName} right now (${pct(w.readiness)} ready).`;
  if (!w.boost) reason += ` Difficulty is set so you have about a ${pct(w.p)} chance of getting it right: hard enough to learn from, easy enough not to guess.`;
  p.append(el("p", "reason", reason));

  const g = el("div", "grid");
  [["Readiness", pct(w.readiness)], ["Ability θ", w.ability.toFixed(2)], ["P(correct)", pct(w.p)],
   ["Target b", w.target_b.toFixed(2)], ["Question b", w.b.toFixed(2)], ["Answered here", w.answered]]
    .forEach(([k, v]) => { const s = el("div", "stat"); s.append(el("small", "", k), el("strong", "", String(v))); g.append(s); });
  p.append(g);

  p.append(el("h2", "", "Priority score breakdown"));
  const labels = { weakness: "Weakness", explore: "Exploration", misconception: "Misconception", repeat_penalty: "Recently asked" };
  Object.entries(parts).forEach(([k, v]) => {
    const row = el("div", "bar");
    const track = el("div", "track"), fill = el("div", "fill" + (v < 0 ? " neg" : ""));
    fill.style.width = Math.min(100, Math.abs(v) * 100) + "%";
    track.append(fill);
    row.append(el("span", "", labels[k]), track, el("span", "", v.toFixed(2)));
    p.append(row);
  });

  p.append(el("h2", "", `Top candidates (of ${w.pool} chapters)`));
  const ol = el("ol", "cands");
  w.candidates.forEach(c => { const li = el("li"); li.append(el("span", "", c.chapter), el("span", "", c.score.toFixed(2))); ol.append(li); });
  p.append(ol);
  if (!w.retry && w.chapter !== w.candidates[0].chapter)
    p.append(el("p", "legend", "Chosen from chapters tied within 0.05 of the top score, preferring one with a question ready."));
  p.append(el("p", "legend", `Source: ${w.source}. Level: ${w.level}.`));
}

async function refreshDash() {
  const d = await api(`/api/dashboard?session_id=${S.sid}`);
  updateSide(d);
  const p = $("#dash");
  p.innerHTML = "";
  const pr = d.predicted, mk = d.marking;
  $("#pred").textContent = `Predicted: ${pct(pr.with_strategy)}${mk ? " of max marks" : ""}`;
  $("#pred").hidden = false;

  // predicted score, changing as the student answers
  const ps = el("div", "predict");
  ps.append(el("div", "big", pct(pr.with_strategy)),
    el("div", "", mk ? `predicted score (% of max marks) on a ${d.exam} paper, marking +${mk[0]} / −${mk[1]}`
                     : `predicted score on a ${d.exam} test`));
  if (pr.strategy_gain > 0.005)
    ps.append(el("div", "up", `+${pct(pr.strategy_gain)} just from skipping questions you're likely to get wrong`));
  p.append(el("h2", "", "Predicted score"), ps);
  if (d.predicted_log.length > 1) {
    const box = el("div", "chartbox small"), cv = el("canvas");
    box.append(cv); p.append(box);
    if (S.charts.pred) S.charts.pred.destroy();
    if (window.Chart) S.charts.pred = new Chart(cv, {
      type: "line",
      data: { labels: d.predicted_log.map((_, i) => i + 1),
              datasets: [{ data: d.predicted_log.map(x => Math.round(x * 1000) / 10), borderColor: css("--accent"), tension: .3, pointRadius: 0 }] },
      options: { maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } },
                 scales: { x: { ticks: { color: css("--muted") }, grid: { display: false } },
                           y: { ticks: { color: css("--muted"), callback: v => v + "%" }, grid: { color: css("--line") } } } },
    });
  }

  // focus / fatigue
  const wb = d.wellbeing;
  const fm = el("div", "focusrow " + wb.state);
  fm.append(el("strong", "", { ok: "🟢 Focused", tired: "🟠 Getting tired", frustrated: "🔴 Frustrated" }[wb.state]),
            el("span", "", wb.reasons.length ? wb.reasons.join("; ") : "No signs of fatigue: steady speed and accuracy."));
  p.append(el("h2", "", "Focus"), fm);

  // why the student is wrong
  const wrong = d.errors.reduce((a, e) => a + e.n, 0);
  p.append(el("h2", "", `Why you get questions wrong${wrong ? ` (${wrong} misses)` : ""}`));
  if (!wrong) p.append(el("p", "empty", "No wrong answers yet."));
  d.errors.filter(e => e.n).sort((a, b) => b.n - a.n).forEach(e => {
    const row = el("div", "bar"), track = el("div", "track"), fill = el("div", "fill err-" + e.type);
    fill.style.width = (100 * e.n / wrong) + "%"; track.append(fill);
    row.append(el("span", "", e.label), track, el("span", "", String(e.n)));
    p.append(row);
  });
  const top = d.errors.filter(e => e.n).sort((a, b) => b.n - a.n)[0];
  if (top) p.append(el("p", "legend", `Fix for your top pattern (${top.label}): ${top.fix}`));

  // confidence calibration + negative-marking strategy
  p.append(el("h2", "", "Confidence vs reality (exam strategy)"));
  if (!d.calibration.length) p.append(el("p", "empty", "Rate your confidence on each answer to see this."));
  else {
    const t = el("table", "calib");
    const hr = el("tr");
    ["You said", "Right", mk ? "Marks / question" : "Accuracy", "Strategy"].forEach(h => hr.append(el("th", "", h)));
    t.append(hr);
    const name = { sure: "🟢 Sure", unsure: "🟡 Unsure", guess: "🔴 Guessing" };
    d.calibration.forEach(c => {
      const tr = el("tr");
      tr.append(el("td", "", `${name[c.confidence]} (${c.n})`), el("td", "", pct(c.accuracy)),
                el("td", c.expected >= 0 ? "up" : "down", mk ? (c.expected >= 0 ? "+" : "") + c.expected.toFixed(2) : pct(c.accuracy)),
                el("td", "", c.advice));
      t.append(tr);
    });
    p.append(t);
    if (mk) p.append(el("p", "legend", `Break-even: attempting pays only if you're right more than ${pct(mk[1] / (mk[0] + mk[1]))} of the time.`));
  }

  // where study time pays most
  p.append(el("h2", "", "Highest-payoff chapters to study next"));
  const ul2 = el("ol", "cands");
  d.gains.forEach(gn => { const li = el("li"); li.append(el("span", "", `${gn.chapter} (${pct(gn.readiness)} ready)`), el("span", "up", `+${gn.gain.toFixed(2)}`)); ul2.append(li); });
  p.append(ul2, el("p", "legend", mk ? "Extra expected marks per question if your ability here rises one step." : "Extra expected accuracy per question if your ability here rises one step."));

  const g = el("div", "grid");
  [["Answered", d.stats.answered], ["Accuracy", pct(d.stats.accuracy)], ["Avg time", d.stats.avg_secs + "s"],
   ["Misconceptions", d.stats.misconceptions], ["Chapters touched", d.stats.chapters_touched], ["Exam", d.exam]]
    .forEach(([k, v]) => { const s = el("div", "stat"); s.append(el("small", "", k), el("strong", "", String(v))); g.append(s); });
  p.append(el("h2", "", "Session"), g);

  p.append(el("h2", "", `Subject readiness for ${d.exam}`));
  d.subjects.forEach(s => {
    const row = el("div", "bar"), track = el("div", "track"), fill = el("div", "fill");
    fill.style.width = pct(s.readiness); track.append(fill);
    row.append(el("span", "", s.subject), track, el("span", "", pct(s.readiness)));
    p.append(row);
  });

  p.append(el("h2", "", "Ability vs question difficulty"));
  const box = el("div", "chartbox"), cv = el("canvas");
  box.append(cv); p.append(box);
  if (S.charts.theta) S.charts.theta.destroy();
  if (window.Chart) S.charts.theta = new Chart(cv, {
    type: "line",
    data: {
      labels: d.history.map(h => h.i),
      datasets: [
        { label: "Your ability θ (chapter answered)", data: d.history.map(h => h.theta), borderColor: css("--accent"), tension: .3, pointRadius: 0 },
        { label: "Question difficulty b", data: d.history.map(h => h.b), showLine: false, pointRadius: 5,
          pointBackgroundColor: d.history.map(h => h.correct ? css("--good") : css("--bad")), borderColor: "transparent" },
      ],
    },
    options: {
      maintainAspectRatio: false, animation: false,
      plugins: { legend: { labels: { color: css("--muted"), boxWidth: 12 } },
                 tooltip: { callbacks: { afterLabel: c => d.history[c.dataIndex].chapter } } },
      scales: { x: { ticks: { color: css("--muted") }, grid: { display: false } },
                y: { ticks: { color: css("--muted") }, grid: { color: css("--line") } } },
    },
  });

  p.append(el("h2", "", "Weak areas"));
  if (!d.weak.length) p.append(el("p", "empty", "Nothing yet."));
  const ul = el("ul", "weak");
  d.weak.forEach(c => ul.append(el("li", "", `${c.chapter} · ${pct(c.readiness)} ready · ${c.n} answered${c.flags ? ` · ⚠ ${c.flags} misconception` : ""}`)));
  p.append(ul);

  p.append(el("h2", "", "Full syllabus map (click a tile to focus)"));
  const map = el("div", "map");
  const bySubj = {};
  d.chapters.forEach(c => (bySubj[c.subject] ||= []).push(c));
  Object.entries(bySubj).forEach(([s, chs]) => {
    map.append(el("h3", "", s));
    const tiles = el("div", "tiles");
    chs.forEach(c => {
      const t = el("button", "tile" + (c.n ? "" : " prior") + (c.flags ? " flag" : "") + (S.focus === c.chapter ? " focus" : ""));
      t.style.background = `hsl(${Math.round(c.readiness * 130)}, 62%, 46%)`;
      t.title = `${c.chapter}: ${pct(c.readiness)} ready${c.n ? `, ${c.n} answered` : " (estimate from subject prior)"}`;
      t.setAttribute("aria-label", t.title);
      t.onclick = () => setFocus(c.chapter);
      tiles.append(t);
    });
    map.append(tiles);
  });
  map.append(el("p", "legend", "Red = not ready, green = exam-ready. Faded = not practised yet (estimated from subject). Red ring = misconception."));
  p.append(map);
}

async function setFocus(f) {
  try {
    const d = await api("/api/focus", { session_id: S.sid, focus: f });
    S.focus = f;
    markFocus();
    document.body.classList.remove("side-open");
    say(f === "all" ? `Focus: whole syllabus (${d.pool} chapters).` : `Focus: ${f} (${d.pool} chapter${d.pool === 1 ? "" : "s"}).`);
    // swap an unanswered question for one from the new focus
    if (S.q) { S.q = null; clearInterval(S.timer); [...document.querySelectorAll(".msg.card:not(.result)")].pop().querySelectorAll("button").forEach(b => b.disabled = true); }
    next();
  } catch (e) { say("⚠ " + e.message); }
}

function tab(name) {
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("on", p.id === name));
}

$("#ask").onsubmit = async e => {
  e.preventDefault();
  const text = $("#input").value.trim();
  if (!text) return;
  $("#input").value = "";
  say(text, "user");
  const [cmd, ...rest] = text.split(" ");
  const arg = rest.join(" ").trim();
  const card = [...document.querySelectorAll(".msg.card:not(.result)")].pop();
  if (cmd === "/hint") return S.q ? hint(card) : say("No active question.");
  if (cmd === "/skip" || cmd === "/next") return S.q ? card.querySelector(".actions .btn:last-child").click() : next();
  if (cmd === "/focus") return setFocus(arg || "all");
  if (cmd === "/dashboard") return tab("dash");
  if (cmd === "/why") return tab("why");
  if (cmd === "/sim") return tab("sim");
  const wait = say("Thinking…", "bot", "muted");
  try { const d = await api("/api/ask", { session_id: S.sid, text }); wait.remove(); say(d.reply); }
  catch (err) { wait.remove(); say("⚠ " + err.message); }
};

document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => tab(b.dataset.tab));
$("#level").onchange = e => start(e.target.value);
$("#sidebtn").onclick = () => document.body.classList.toggle("side-open");

$("#runsim").onclick = async () => {
  const btn = $("#runsim");
  btn.disabled = true; btn.textContent = "Simulating…";
  const d = await api(`/api/simulate?exam=${S.exam}`);
  const avg = Math.round(d.results.reduce((a, r) => a + r.saved, 0) / d.results.length);
  $("#simhead").textContent = `Adaptive reaches ${d.exam} readiness in ${avg}% fewer questions on average.`;
  if (S.charts.sim) S.charts.sim.destroy();
  S.charts.sim = new Chart($("#simchart"), {
    type: "bar",
    data: {
      labels: d.results.map(r => r.profile),
      datasets: [
        { label: "Adaptive (PrepPilot)", data: d.results.map(r => r.adaptive), backgroundColor: css("--accent"), borderRadius: 6 },
        { label: "Static bank", data: d.results.map(r => r.static), backgroundColor: css("--muted"), borderRadius: 6 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: css("--muted"), boxWidth: 12 } } },
      scales: { x: { ticks: { color: css("--muted") }, grid: { display: false } },
                y: { title: { display: true, text: "Questions to readiness", color: css("--muted") }, ticks: { color: css("--muted") }, grid: { color: css("--line") } } },
    },
  });
  btn.disabled = false; btn.textContent = "Run again";
};

// ---------- account: signed-in user or guest; anyone else goes to /login ----------
function signOut() {
  const token = localStorage.getItem("pp-token");
  if (token) fetch(`/api/logout?token=${encodeURIComponent(token)}`, { method: "POST" }).catch(() => {});
  try { ["pp-user", "pp-token", "pp-prefs"].forEach(k => localStorage.removeItem(k)); } catch (e) {}
  location.href = "/login";
}
try { S.user = JSON.parse(localStorage.getItem("pp-user") || "null"); } catch (e) { S.user = null; }
try { S.prefs = JSON.parse(localStorage.getItem("pp-prefs") || "null"); } catch (e) { S.prefs = null; }
if (!S.user) location.replace("/login");
else if (!S.prefs) { S.user = null; location.replace("/setup"); }  // first visit: set up first
else {
  $("#timed").checked = !!S.prefs.timed;
  $("#userbtn").textContent = (S.user.name || "?").trim()[0].toUpperCase();
  $("#uname").textContent = S.user.name;
  $("#uemail").textContent = S.user.guest ? "Guest: progress isn't saved" : S.user.email;
  $("#logout").textContent = S.user.guest ? "Sign in / create account" : "Log out";
  const token = localStorage.getItem("pp-token");
  if (token) fetch(`/api/me?token=${encodeURIComponent(token)}`).then(r => { if (r.status === 401) signOut(); });
}
$("#userbtn").onclick = e => {
  e.stopPropagation();
  const m = $("#usermenu");
  m.hidden = !m.hidden;
  $("#userbtn").setAttribute("aria-expanded", String(!m.hidden));
};
document.addEventListener("click", e => { if (!e.target.closest(".user")) $("#usermenu").hidden = true; });
$("#logout").onclick = signOut;

window.addEventListener("load", () => {
  if (!S.user) return;
  let lv = "JEE";
  try { lv = localStorage.getItem("pp-level") || lv; } catch (e) {}
  if (![...$("#level").options].some(o => o.value === lv)) lv = "JEE";
  $("#level").value = lv;
  const url = new URL(location.href);
  const fresh = url.searchParams.has("new");
  url.searchParams.delete("new");
  start(lv, fresh).then(() => history.replaceState(null, "", url)).catch(e => problem(e));
});
$("#breaknow").onclick = () => S.sid && api("/api/break", { session_id: S.sid }).catch(e => e.brk && showBreak(e.brk));
</script>
<div id="breakov" hidden role="dialog" aria-modal="true" aria-labelledby="brktitle">
  <div class="brkcard">
    <div class="brkicon">☕</div>
    <h2 id="brktitle">Break time</h2>
    <p id="brkwhy"></p>
    <div id="brkleft" class="brkleft">10:00</div>
    <div class="brktrack"><div id="brkbar"></div></div>
    <p id="brkstats" class="legend"></p>
    <ul id="brktips"></ul>
    <p class="legend">Questions unlock automatically when the break ends.</p>
  </div>
</div>
</body>
</html>
'''

# ---------------- study.html ----------------
FILES['study.html'] = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Your study space · PrepPilot</title>
<link rel="stylesheet" href="/site.css">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700;800&family=Instrument+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root { --bg:#f5f6fa; --panel:#fff; --ink:#202637; --muted:#687084; --line:#e2e6ee; --accent:#5746cd; --sel:#efedfc; --accent-ink:#fff; }
body { min-height:100vh; }
.topbar { height:76px; background:#19232d; color:#fff; display:flex; align-items:center; justify-content:space-between; padding:0 32px; gap:20px; }
.topbar .brand span { color:#b9afff; }.topbar .brand { white-space:nowrap; }
.top-level { border:1px solid #53606b; padding:7px 16px; border-radius:10px; margin-right:auto; font-size:14px; }
.profile { display:flex;align-items:center;gap:12px;font-size:14px; }.avatar { width:38px;height:38px;display:grid;place-items:center;background:#d9d4fa;color:#372c7c;border-radius:50%;font-weight:700; }
.layout { display:grid;grid-template-columns:230px minmax(0,1fr);min-height:calc(100vh - 76px); }
aside { background:var(--panel);border-right:1px solid var(--line);padding:32px 18px; }
.nav-caption { text-transform:uppercase;font-size:11px;letter-spacing:.12em;color:var(--muted);padding:0 16px;margin:0 0 14px; }
.side-link { display:flex;align-items:center;gap:12px;text-decoration:none;padding:13px 16px;border-radius:9px;margin-bottom:8px;font-size:15px; }
.side-link.active { background:var(--sel);color:var(--accent);font-weight:700; }.side-link:hover { background:var(--sel); }
.side-note { border-top:1px solid var(--line);margin:30px 16px 0;padding-top:22px;font-size:13px;color:var(--muted); }
.side-note strong { display:block;color:var(--ink);margin-bottom:5px; }
main { padding:28px 40px 60px;max-width:1440px;width:100%;margin:auto; }
.breadcrumb { color:var(--muted);font-size:13px;margin:0 0 22px; }.breadcrumb span { color:var(--ink); }
.hero { position:relative;background:#26343f;color:#fff;border-radius:18px;padding:32px 36px;overflow:hidden;display:flex;justify-content:space-between;align-items:center;gap:20px; }
.hero::after { content:'θ';position:absolute;right:120px;top:-95px;font:300px Georgia,serif;color:#ffffff06;pointer-events:none; }
.overline { font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:#bfbae4;margin:0 0 10px; }
h1 { font-size:clamp(26px,3vw,38px);line-height:1.15;margin:0 0 12px; }.hero p:last-child { margin:0;color:#c6ced4;font-size:14px; }
.focus-badge { z-index:1;flex-shrink:0;border:1px solid #66727d;border-radius:14px;padding:15px 20px;text-align:center; }.focus-badge b { display:block;font-size:25px;font-family:'Bricolage Grotesque',sans-serif; }.focus-badge span { font-size:11px;color:#c6ced4; }
.section-head { display:flex;justify-content:space-between;align-items:center;gap:16px;margin:30px 0 16px; }.section-head h2 { font-size:21px;margin:0; }.section-head p { color:var(--muted);font-size:13px;margin:4px 0 0; }
.subjects { display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:16px; }
.subject-card { position:relative;display:flex;text-align:left;align-items:flex-start;flex-direction:column;gap:6px;background:var(--panel);border:1px solid var(--line);padding:22px;border-radius:14px;cursor:pointer;transition:border-color .15s,transform .15s; }
.subject-card:hover { transform:translateY(-2px);border-color:var(--accent); }.subject-card[aria-pressed=true] { border:2px solid var(--accent);padding:21px;background:var(--sel); }
.subject-icon { background:var(--tint);color:var(--color);width:48px;height:48px;display:grid;place-items:center;border-radius:12px;font:25px Georgia,serif;margin-bottom:14px; }
.subject-card b { font-size:17px; }.subject-card small { color:var(--muted);font-size:12px; }.subject-card .arrow { position:absolute;right:20px;top:24px;color:var(--muted);font-size:20px; }
.chapter-section { background:var(--panel);border:1px solid var(--line);border-radius:16px;margin-top:28px;padding:24px; }
.chapter-section .section-head { margin:0 0 20px; }.search { border:1px solid var(--line);background:var(--bg);border-radius:9px;padding:10px 14px;width:230px;max-width:100%;font-size:14px; }
.chapters { display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px; }
.chapter { display:flex;align-items:center;gap:14px;border:1px solid var(--line);padding:18px;border-radius:12px; }
.chapter-number { width:32px;flex-shrink:0;color:var(--muted);font-size:13px;font-variant-numeric:tabular-nums; }.chapter-text { flex:1;min-width:0; }.chapter h3 { font-family:inherit;font-size:14px;line-height:1.4;margin:0 0 6px;font-weight:600; }.chapter p { margin:0;color:var(--muted);font-size:11px; }.chapter .btn { font-size:12px;padding:8px 12px;white-space:nowrap; }
.btn:disabled { opacity:.5;cursor:not-allowed; }.note { color:var(--muted);font-size:12px;margin:18px 0 0; }.status { padding:18px;color:var(--muted); }.status[hidden], [hidden] { display:none!important; }
button:focus-visible,a:focus-visible { outline:3px solid #a59aee;outline-offset:3px; }
@media(prefers-color-scheme:dark) { :root { --bg:#11151e;--panel:#1b2130;--ink:#e9edf7;--muted:#a2adc1;--line:#30394b;--accent:#b4a7ff;--sel:#302b4d;--accent-ink:#161221; }.hero{background:#232d3c;} }
@media(max-width:1000px) { .layout{grid-template-columns:185px minmax(0,1fr);}main{padding:24px;}.chapters{grid-template-columns:1fr;}.subject-card{padding:18px;} }
@media(max-width:640px) { .topbar{padding:14px 18px;height:auto;flex-wrap:wrap;gap:10px;}.topbar .brand{font-size:20px;}.top-level{order:3;width:100%;font-size:12px;}.profile{margin-left:auto;}.profile #greeting{display:none;}.layout{display:block;}aside{padding:10px 16px;display:flex;gap:8px;border-right:0;border-bottom:1px solid var(--line);}.side-link{padding:8px 10px;margin:0;font-size:12px;}.nav-caption,.side-note{display:none;}main{padding:20px 16px;}.hero{padding:24px;align-items:flex-start;}.focus-badge{padding:10px;}.focus-badge b{font-size:20px;}.subjects{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}.subject-card{padding:16px;}.subject-card[aria-pressed=true]{padding:15px;}.chapter-section{padding:16px;}.chapter-section .section-head{align-items:stretch;flex-direction:column;}.search{width:100%;}.chapter{padding:14px;gap:8px;}.chapter-number{width:22px;}.chapter h3{font-size:13px;} }
@media(prefers-reduced-motion:reduce) { *{scroll-behavior:auto!important;transition:none!important;} }
</style>
</head>
<body>
<header class="topbar"><a href="/study" class="brand">Prep<span>Pilot</span></a><span class="top-level" id="levelname">Your study space</span><div class="profile"><span id="greeting"></span><span class="avatar" id="avatar">P</span></div></header>
<div class="layout">
<aside aria-label="Study navigation"><div><p class="nav-caption">Your learning</p><a class="side-link active" href="/study" aria-current="page">▦ &nbsp; Study space</a><a class="side-link" href="/setup">⚙ &nbsp; Preferences</a><a class="side-link" id="resume" href="/app" hidden>↗ &nbsp; Resume quiz</a><div class="side-note"><strong>One chapter at a time.</strong>Choose where to begin. PrepPilot adapts the questions as you practise.</div></div></aside>
<main>
<p class="breadcrumb">Study space &nbsp; / &nbsp; <span id="crumb">Your syllabus</span></p>
<section class="hero"><div><p class="overline">Your personal study space</p><h1 id="welcome">What will you learn today?</h1><p>Pick a subject. Find your chapter. Make a little progress.</p></div><div class="focus-badge"><b id="minutes">40 min</b><span>Your next focus block</span></div></section>
<div class="section-head"><div><h2>Your subjects</h2><p id="settings">Prepared around your preferences.</p></div><a href="/setup" class="btn">Edit setup</a></div>
<div id="status" class="status" role="status">Loading your subjects…</div>
<div id="subjects" class="subjects" aria-label="Choose a subject"></div>
<section class="chapter-section" id="chapter-section" hidden aria-labelledby="chaptertitle">
<div class="section-head"><div><h2 id="chaptertitle">Chapters</h2><p id="chaptercount"></p></div><input id="search" class="search" type="search" aria-label="Search chapters" placeholder="Search a chapter…"></div>
<div class="chapters" id="chapters"></div><p class="note">Your focus timer starts when you click Start quiz. You can finish the current question before a break begins.</p>
</section>
</main></div>
<script>
const $ = s => document.querySelector(s);
function el(tag,cls,text) { const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e; }
let user, prefs, catalog, selected;
try { user=JSON.parse(localStorage.getItem('pp-user')||'null');prefs=JSON.parse(localStorage.getItem('pp-prefs')||'null'); } catch(e) {}
const symbols={Physics:['φ','#e7f0ff','#3264ac'],Chemistry:['⚗','#f0eafb','#8653a9'],Maths:['∑','#fff0df','#a65b16'],Biology:['❧','#e2f4e9','#28744b'],Science:['⚛','#e7f0ff','#3264ac'],English:['Aa','#f0eafb','#8653a9']};
function renderChapters() {
 const box=$('#chapters');box.replaceChildren();
 const term=$('#search').value.trim().toLowerCase();
 const matches=selected.chapters.filter(ch=>ch.chapter.toLowerCase().includes(term));
 $('#chaptercount').textContent=matches.length+' chapters · Choose where your quiz begins';
 for(const ch of matches) {
  const card=el('article','chapter');
  card.append(el('span','chapter-number',String(selected.chapters.indexOf(ch)+1).padStart(2,'0')));
  const text=el('div','chapter-text');text.append(el('h3','',ch.chapter),el('p','',ch.questions?ch.questions+' bank questions':catalog.ai?'AI practice available':'AI needed · currently unavailable'));
  const start=el('button','btn', 'Start quiz →');start.disabled=!ch.questions&&!catalog.ai;
  start.setAttribute('aria-label','Start quiz: '+ch.chapter);
  start.onclick=()=>{ const query=new URLSearchParams({new:'1',focus:ch.chapter});location.href='/app?'+query; };
  card.append(text,start);box.append(card);
 }
 if(!matches.length)box.append(el('p','status','No chapters match your search.'));
}
function choose(subject) {
 selected=subject;$('#search').value='';$('#chaptertitle').textContent=subject.subject+' chapters';$('#chapter-section').hidden=false;
 document.querySelectorAll('.subject-card').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.subject===subject.subject)));
 renderChapters();
}
async function load() {
 if(!user){location.replace('/login');return;}if(!prefs){location.replace('/setup');return;}
 $('#greeting').textContent='Hi, '+user.name.split(' ')[0];$('#avatar').textContent=user.name[0].toUpperCase();
 $('#minutes').textContent=(prefs.block_minutes||40)+' min';
 const level=prefs.level||'JEE';
 const response=await fetch('/api/catalog?'+new URLSearchParams({level}));if(!response.ok)throw Error('Could not load your syllabus. Please try again.');
 catalog=await response.json();$('#levelname').textContent=catalog.name;$('#crumb').textContent=catalog.name;
 $('#settings').textContent=catalog.name+' · '+({'-1':'Just starting','0':'Comfortable','1':'Confident'}[String(prefs.start_ability??0)]||'Adaptive practice');
 const subjects=catalog.subjects.filter(s=>!prefs.subjects?.length||prefs.subjects.includes(s.subject));
 for(const subject of subjects) {
  const [symbol,tint,color]=symbols[subject.subject]||['▤','#e7eef6','#536980'];
  const button=el('button','subject-card');button.dataset.subject=subject.subject;button.setAttribute('aria-pressed','false');
  button.style.setProperty('--tint',tint);button.style.setProperty('--color',color);
  button.append(el('span','subject-icon',symbol),el('span','arrow','↗'),el('b','',subject.subject),el('small','',subject.chapters.length+' chapters'));
  button.onclick=()=>choose(subject);$('#subjects').append(button);
 }
 if(subjects.length)choose(subjects[0]);else throw Error('Choose your subjects in Preferences to get started.');
 const key='pp-session:'+JSON.stringify([user.email||'guest',level,false,prefs]);$('#resume').hidden=!localStorage.getItem(key);
 $('#status').hidden=true;
}
$('#search').oninput=renderChapters;
load().catch(e=>{$('#status').hidden=false;$('#status').textContent=e.message;});
</script>
</body></html>
'''

# ---------------- setup.html ----------------
FILES['setup.html'] = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Set up your practice · PrepPilot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/site.css">
<style>
main { padding: 40px 0 120px; }
.intro { max-width: 640px; margin-bottom: 28px; }
.intro h1 { font-size: clamp(30px, 4vw, 42px); line-height: 1.1; margin: 12px 0 8px; }
.intro p { color: var(--muted); margin: 0; font-size: 17px; }

.step { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 24px; margin-bottom: 16px; }
.step-head { display: flex; gap: 14px; align-items: baseline; margin-bottom: 4px; }
.num { flex: none; width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: 14px;
       background: var(--chip); color: var(--muted); }
.step.done .num { background: var(--accent); color: var(--accent-ink); }
.step h2 { font-size: 21px; margin: 0; }
.step .sub { color: var(--muted); font-size: 14.5px; margin: 2px 0 16px 42px; }
.body { margin-left: 42px; }

.group-label { font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 14px 0 8px; }
.group-label:first-child { margin-top: 0; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { border: 1px solid var(--line); background: var(--bg); border-radius: 12px; padding: 10px 16px; font-weight: 600; cursor: pointer;
        transition: border-color .15s, background .15s; }
.chip:hover { border-color: var(--accent); }
.chip[aria-pressed="true"] { border-color: var(--accent); background: var(--sel); color: var(--ink); }
.chip small { display: block; font-weight: 500; color: var(--muted); font-size: 12px; }

.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.card { text-align: left; border: 1px solid var(--line); background: var(--bg); border-radius: 16px; padding: 16px; cursor: pointer;
        transition: border-color .15s, background .15s; }
.card:hover { border-color: var(--accent); }
.card[aria-pressed="true"] { border-color: var(--accent); background: var(--sel); }
.card b { display: block; font-size: 16px; margin-bottom: 4px; }
.card span { color: var(--muted); font-size: 14px; }
.card .emoji { font-size: 22px; display: block; margin-bottom: 6px; }

.toggle { display: flex; gap: 14px; align-items: center; cursor: pointer; }
.toggle input { width: 20px; height: 20px; accent-color: var(--accent); }
.toggle b { display: block; }
.toggle span { color: var(--muted); font-size: 14px; }

.bar { position: fixed; left: 0; right: 0; bottom: 0; background: color-mix(in srgb, var(--panel) 92%, transparent);
       backdrop-filter: blur(10px); border-top: 1px solid var(--line); }
.bar .wrap { display: flex; align-items: center; gap: 16px; padding: 14px 0; }
.summary { color: var(--muted); font-size: 14px; flex: 1; min-width: 0; }
.summary b { color: var(--ink); }
.btn[disabled] { opacity: .5; cursor: not-allowed; transform: none; }

@media (max-width: 760px) {
  .cards { grid-template-columns: 1fr; }
  .body, .step .sub { margin-left: 0; }
  .step { padding: 18px; }
  .summary { display: none; }
  .bar .btn { width: 100%; }
}
</style>
</head>
<body>
<nav class="nav">
  <div class="wrap">
    <a class="brand" href="/">Prep<span>Pilot</span></a>
    <span class="muted" id="hello" style="margin-left:auto;font-size:15px"></span>
  </div>
</nav>

<main class="wrap">
  <div class="intro">
    <span class="eyebrow">Set up · takes 30 seconds</span>
    <h1>Let's tailor PrepPilot to you.</h1>
    <p>Pick what you're studying and how you like to learn. You can change all of this later from your account menu.</p>
  </div>

  <section class="step" id="s-level">
    <div class="step-head"><span class="num">1</span><h2>What are you preparing for?</h2></div>
    <p class="sub">Questions, chapters and the "exam-ready" bar all follow this.</p>
    <div class="body" id="levels"></div>
  </section>

  <section class="step" id="s-subjects">
    <div class="step-head"><span class="num">2</span><h2>Which subjects?</h2></div>
    <p class="sub">Pick one or more. PrepPilot only asks from these.</p>
    <div class="body chips" id="subjects"></div>
  </section>

  <section class="step" id="s-start">
    <div class="step-head"><span class="num">3</span><h2>Where should we start?</h2></div>
    <p class="sub">Sets the first questions' difficulty. After a few answers the engine adjusts on its own either way.</p>
    <div class="body cards" data-pref="start_ability">
      <button class="card" data-value="-1"><span class="emoji">🌱</span><b>Just starting</b><span>Begin with easier questions and build up.</span></button>
      <button class="card" data-value="0"><span class="emoji">📘</span><b>Comfortable</b><span>I know the basics. Start at a standard level.</span></button>
      <button class="card" data-value="1"><span class="emoji">🚀</span><b>Confident</b><span>Challenge me from the first question.</span></button>
    </div>
  </section>

  <section class="step" id="s-rhythm">
    <div class="step-head"><span class="num">4</span><h2>How long should each focus block be?</h2></div>
    <p class="sub">Your focus timer starts when you start your chapter quiz. After your selected time, take a break; it is longer if you seem tired.</p>
    <div class="body cards" data-pref="block_minutes">
      <button class="card" data-value="25"><span class="emoji">⏱️</span><b>25 minutes</b><span>Short sprints · 6–9 min breaks</span></button>
      <button class="card" data-value="40"><span class="emoji">🎯</span><b>40 minutes</b><span>Recommended · 10–15 min breaks</span></button>
      <button class="card" data-value="60"><span class="emoji">🏔️</span><b>60 minutes</b><span>Exam stamina · 15–22 min breaks</span></button>
    </div>
  </section>

  <section class="step" id="s-style">
    <div class="step-head"><span class="num">5</span><h2>How should mistakes be explained?</h2></div>
    <p class="sub">Used for AI explanations of wrong answers and for your doubts.</p>
    <div class="body cards" data-pref="explain_style">
      <button class="card" data-value="steps"><span class="emoji">🪜</span><b>Step by step</b><span>Every step, numbered.</span></button>
      <button class="card" data-value="short"><span class="emoji">⚡</span><b>Short and direct</b><span>Just the key idea.</span></button>
      <button class="card" data-value="analogy"><span class="emoji">🧩</span><b>With analogies</b><span>An everyday example first.</span></button>
    </div>
  </section>

  <section class="step done" id="s-timed">
    <div class="step-head"><span class="num">6</span><h2>Exam conditions</h2></div>
    <p class="sub">You can switch this on and off any time from the ⏱ Timed switch.</p>
    <label class="body toggle"><input type="checkbox" id="timed">
      <div><b>Start in timed mode</b><span>Each question gets a countdown, like the real exam.</span></div></label>
  </section>
</main>

<div class="bar">
  <div class="wrap">
    <div class="summary" id="summary"></div>
    <button class="btn primary big" id="go" disabled>Choose a chapter →</button>
  </div>
</div>

<script>
const $ = s => document.querySelector(s);
let user = null;
try { user = JSON.parse(localStorage.getItem("pp-user") || "null"); } catch (e) {}
if (!user) location.replace("/login");

let saved = {};
try { saved = JSON.parse(localStorage.getItem("pp-prefs") || "{}"); } catch (e) {}
const P = { level: saved.level || (user && user.level) || "", subjects: saved.subjects || [], start_ability: saved.start_ability ?? 0,
            block_minutes: saved.block_minutes || 40, explain_style: saved.explain_style || "steps", timed: !!saved.timed };
let LEVELS = [];

if (user) $("#hello").textContent = user.guest ? "Guest" : `Hi, ${user.name.split(" ")[0]}`;

function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

function renderLevels() {
  const groups = [["School · Primary", [1, 2, 3, 4, 5]], ["School · Middle", [6, 7, 8]], ["School · Secondary", [9, 10]],
                  ["School · Senior secondary", [11, 12]], ["Competitive exams", ["JEE", "NEET"]]];
  const box = $("#levels");
  box.innerHTML = "";
  groups.forEach(([label, items]) => {
    box.append(el("div", "group-label", label));
    const chips = el("div", "chips");
    items.forEach(it => {
      const key = typeof it === "number" ? `Class ${it}` : it;
      const lv = LEVELS.find(l => l.level === key);
      const b = el("button", "chip", lv.name);
      b.append(el("small", "", lv.subjects.map(s => s.subject).join(" · ")));
      b.setAttribute("aria-pressed", String(P.level === key));
      b.onclick = () => {
        if (P.level !== key) P.subjects = [];
        P.level = key; renderLevels(); renderSubjects(); update();
      };
      chips.append(b);
    });
    box.append(chips);
  });
}

function renderSubjects() {
  const box = $("#subjects");
  box.innerHTML = "";
  const lv = LEVELS.find(l => l.level === P.level);
  if (!lv) { box.append(el("span", "muted", "Pick a class or exam first.")); return; }
  const valid = lv.subjects.map(s => s.subject);
  P.subjects = P.subjects.filter(s => valid.includes(s));
  if (!P.subjects.length) P.subjects = [...valid];  // default: everything
  lv.subjects.forEach(({ subject, chapters }) => {
    const b = el("button", "chip", subject);
    b.append(el("small", "", `${chapters} chapters`));
    b.setAttribute("aria-pressed", String(P.subjects.includes(subject)));
    b.onclick = () => {
      P.subjects = P.subjects.includes(subject) ? P.subjects.filter(s => s !== subject) : [...P.subjects, subject];
      renderSubjects(); update();
    };
    box.append(b);
  });
}

document.querySelectorAll(".cards[data-pref]").forEach(group => {
  const key = group.dataset.pref;
  const sync = () => group.querySelectorAll(".card").forEach(c => c.setAttribute("aria-pressed", String(String(P[key]) === c.dataset.value)));
  group.querySelectorAll(".card").forEach(c => c.onclick = () => {
    P[key] = isNaN(c.dataset.value) ? c.dataset.value : Number(c.dataset.value);
    sync(); update();
  });
  sync();
});
$("#timed").checked = P.timed;
$("#timed").onchange = e => { P.timed = e.target.checked; update(); };

function update() {
  const lv = LEVELS.find(l => l.level === P.level);
  $("#s-level").classList.toggle("done", !!lv);
  $("#s-subjects").classList.toggle("done", P.subjects.length > 0);
  ["s-start", "s-rhythm", "s-style"].forEach(id => $("#" + id).classList.add("done"));
  const ok = !!lv && P.subjects.length > 0;
  $("#go").disabled = !ok;
  const start = { "-1": "just starting", "0": "comfortable", "1": "confident" }[String(P.start_ability)];
  $("#summary").innerHTML = "";
  if (ok) $("#summary").append(
    el("b", "", lv.name), ` · ${P.subjects.join(", ")} · ${start} · ${P.block_minutes}-min blocks${P.timed ? " · timed" : ""}`);
  else $("#summary").textContent = !lv ? "Pick a class or exam to continue." : "Pick at least one subject.";
}

$("#go").onclick = () => {
  try {
    localStorage.setItem("pp-prefs", JSON.stringify(P));
    localStorage.setItem("pp-level", P.level);
  } catch (e) {}
  location.href = "/study";
};

fetch("/api/levels").then(r => r.json()).then(d => {
  LEVELS = d;
  if (!LEVELS.some(l => l.level === P.level)) P.level = "";
  renderLevels(); renderSubjects(); update();
});
</script>
</body>
</html>
'''

# ---------------- landing.html ----------------
FILES['landing.html'] = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PrepPilot · Adaptive practice for JEE, NEET and Class 1–12</title>
<meta name="description" content="An AI study companion that adapts every question to you, diagnoses why you get things wrong, and coaches your exam strategy.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/site.css">
<style>
/* hero */
.hero { padding: 72px 0 56px; position: relative; overflow: hidden; }
.hero::before { content: ""; position: absolute; inset: -40% -10% auto auto; width: 700px; height: 700px; border-radius: 50%;
                background: radial-gradient(circle, var(--glow), transparent 65%); pointer-events: none; }
.hero .wrap { display: grid; grid-template-columns: 1.05fr .95fr; gap: 56px; align-items: center; position: relative; }
.hero h1 { font-size: clamp(38px, 5.4vw, 64px); line-height: 1.04; margin: 18px 0 18px; font-weight: 800; }
.hero h1 em { font-style: normal; color: var(--accent); }
.hero p.lead { font-size: 19px; color: var(--muted); max-width: 540px; margin: 0 0 28px; }
.cta { display: flex; gap: 12px; flex-wrap: wrap; }
.trust { margin-top: 26px; display: flex; gap: 18px; flex-wrap: wrap; font-size: 14px; color: var(--muted); }
.trust b { color: var(--ink); }

/* product mock */
.mock { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 18px;
        box-shadow: 0 30px 80px rgba(0,0,0,.12), 0 2px 6px rgba(0,0,0,.05); transform: rotate(1deg); }
.mock .bar { display: flex; gap: 6px; margin-bottom: 14px; }
.mock .bar i { width: 10px; height: 10px; border-radius: 50%; background: var(--line); }
.mq { border: 1px solid var(--line); border-radius: 14px; padding: 14px; }
.tags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
.tag { font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 99px; background: var(--chip); color: var(--muted); }
.tag.ok { color: var(--good); } .tag.hot { color: var(--accent); }
.mq p { margin: 0 0 10px; font-size: 15px; }
.opt { border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; margin-top: 6px; font-size: 14px; }
.opt.right { border-color: var(--good); background: color-mix(in srgb, var(--good) 12%, var(--panel)); }
.opt.wrong { border-color: var(--bad); background: color-mix(in srgb, var(--bad) 12%, var(--panel)); }
.diag { margin-top: 12px; border: 1px solid color-mix(in srgb, var(--accent) 40%, var(--line)); background: var(--sel); border-radius: 8px; padding: 8px 12px; font-size: 13.5px; }
.diag b { display: block; }
.mini { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }
.mini div { background: var(--chip); border-radius: 10px; padding: 10px 12px; font-size: 12px; color: var(--muted); }
.mini strong { display: block; font-family: "Bricolage Grotesque", sans-serif; font-size: 20px; color: var(--ink); }
.mini .up { color: var(--good); }

/* sections */
section { padding: 72px 0; scroll-margin-top: 64px; }
.sec-head { max-width: 680px; margin-bottom: 40px; }
.sec-head h2 { font-size: clamp(30px, 3.6vw, 42px); line-height: 1.1; margin: 12px 0 10px; }
.sec-head p { color: var(--muted); font-size: 18px; margin: 0; }

.stats { border-block: 1px solid var(--line); padding: 28px 0; background: var(--panel); }
.stats .wrap { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.stat strong { display: block; font-family: "Bricolage Grotesque", sans-serif; font-size: 34px; line-height: 1.1; }
.stat span { color: var(--muted); font-size: 14px; }

.features { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.feature { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 24px; transition: transform .2s, border-color .2s; }
.feature:hover { transform: translateY(-3px); border-color: color-mix(in srgb, var(--accent) 50%, var(--line)); }
.feature h3 { font-size: 20px; margin: 0 0 8px; display: flex; gap: 10px; align-items: baseline; }
.feature h3 .ico { font-size: 20px; }
.feature p { margin: 0; color: var(--muted); font-size: 15px; }

.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; counter-reset: s; }
.step { padding: 24px; border-radius: 18px; border: 1px dashed var(--line); }
.step::before { counter-increment: s; content: counter(s); display: grid; place-items: center; width: 36px; height: 36px;
                border-radius: 50%; background: var(--accent); color: var(--accent-ink); font-weight: 700; margin-bottom: 14px; }
.step h3 { font-size: 19px; margin: 0 0 6px; }
.step p { margin: 0; color: var(--muted); font-size: 15px; }

.proof { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; align-items: center; background: var(--panel);
         border: 1px solid var(--line); border-radius: 24px; padding: 40px; }
.bars .row { display: grid; grid-template-columns: 160px 1fr 52px; gap: 12px; align-items: center; font-size: 14px; margin: 12px 0; }
.bars .track { height: 14px; border-radius: 99px; background: var(--chip); overflow: hidden; }
.bars .fill { height: 100%; border-radius: 99px; background: var(--muted); }
.bars .fill.ours { background: var(--accent); }
.big-num { font-family: "Bricolage Grotesque", sans-serif; font-size: 64px; font-weight: 800; color: var(--accent); line-height: 1; }

.faq details { border-bottom: 1px solid var(--line); padding: 18px 0; }
.faq summary { cursor: pointer; font-weight: 600; font-size: 17px; }
.faq p { color: var(--muted); margin: 10px 0 0; }

.final { text-align: center; padding: 64px 24px; border-radius: 24px; color: #fff;
         background: radial-gradient(circle at 20% 0%, #c2410c, #9a3412 50%, #6b2109); }
.final h2 { font-size: clamp(30px, 4vw, 46px); margin: 0 0 10px; }
.final p { opacity: .9; margin: 0 0 26px; font-size: 18px; }
.final .btn { background: #fff; color: #17191e; border-color: #fff; }

@media (max-width: 900px) {
  .hero .wrap, .proof { grid-template-columns: 1fr; }
  .features, .steps { grid-template-columns: 1fr 1fr; }
  .mock { transform: none; }
}
@media (max-width: 620px) {
  .hero { padding-top: 40px; }
  .features, .steps, .stats .wrap { grid-template-columns: 1fr 1fr; }
  .features, .steps { grid-template-columns: 1fr; }
  .bars .row { grid-template-columns: 110px 1fr 44px; }
  .proof { padding: 24px; }
  section { padding: 52px 0; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; scroll-behavior: auto !important; } }
</style>
</head>
<body>
<nav class="nav">
  <div class="wrap">
    <a class="brand" href="/">Prep<span>Pilot</span></a>
    <div class="nav-links">
      <a href="#features">Features</a>
      <a href="#how">How it works</a>
      <a href="#proof">Results</a>
      <a href="#faq">FAQ</a>
    </div>
    <a class="btn" href="/login">Sign in</a>
    <a class="btn primary" href="/login?mode=signup">Get started</a>
  </div>
</nav>

<header class="hero">
  <div class="wrap">
    <div>
      <span class="eyebrow">● Adaptive AI study companion</span>
      <h1>Practice that understands <em>how you fail</em>, not just how often.</h1>
      <p class="lead">PrepPilot picks every question for you, diagnoses why you got it wrong, and coaches your exam strategy. From Class 1 to JEE Advanced and NEET.</p>
      <div class="cta">
        <a class="btn primary big" href="/login?mode=signup">Start practising free →</a>
        <a class="btn big" href="#how">See how it works</a>
      </div>
      <div class="trust"><span><b>74</b> real JEE Advanced questions</span><span><b>100%</b> answer keys verified</span><span><b>No</b> credit card</span></div>
    </div>

    <div class="mock" aria-hidden="true">
      <div class="bar"><i></i><i></i><i></i></div>
      <div class="mq">
        <div class="tags"><span class="tag">Physics</span><span class="tag">Kinematics</span><span class="tag ok">✓ Verified key</span></div>
        <p>A car covers the first half of a distance at 20 m/s and the second half at 30 m/s. Average speed?</p>
        <div class="opt wrong">A&nbsp; 25 m/s</div>
        <div class="opt right">B&nbsp; 24 m/s</div>
        <div class="diag"><b>🪤 Fell for a trap option</b>You were sure, but 25 m/s is the simple mean. For equal distances use the harmonic mean.</div>
      </div>
      <div class="mini">
        <div>Readiness<strong>62%</strong></div>
        <div>Predicted<strong>41%</strong></div>
        <div>Next<strong class="up">Easier</strong></div>
      </div>
    </div>
  </div>
</header>

<div class="stats">
  <div class="wrap">
    <div class="stat"><strong>14</strong><span>levels: Class 1–12, JEE Advanced, NEET</span></div>
    <div class="stat"><strong>400+</strong><span>chapters mapped across every subject</span></div>
    <div class="stat"><strong>154</strong><span>verified bank questions + AI for the rest</span></div>
    <div class="stat"><strong>44%</strong><span>fewer questions to readiness (simulation)</span></div>
  </div>
</div>

<section id="features">
  <div class="wrap">
    <div class="sec-head">
      <span class="eyebrow">Features</span>
      <h2>Not another static question bank.</h2>
      <p>Most apps give every student the same questions. PrepPilot models how you learn, and changes what you see after every single answer.</p>
    </div>
    <div class="features">
      <div class="feature"><h3><span class="ico" aria-hidden="true">🎯</span>Adapts every question</h3><p>An IRT ability model per chapter picks the topic and difficulty where you learn fastest: about a 60% chance of getting it right.</p></div>
      <div class="feature"><h3><span class="ico" aria-hidden="true">🔍</span>Knows why you're wrong</h3><p>Concept gap, careless slip, trap option or time pressure. Each mistake type gets its own fix, not just "try again".</p></div>
      <div class="feature"><h3><span class="ico" aria-hidden="true">⚖️</span>Exam strategy coach</h3><p>Tracks how often you're right when you feel sure versus guessing, and tells you when negative marking makes skipping smarter.</p></div>
      <div class="feature"><h3><span class="ico" aria-hidden="true">📈</span>Live predicted score</h3><p>See your expected marks move with every answer, and which chapters would add the most marks per hour of study.</p></div>
      <div class="feature"><h3><span class="ico" aria-hidden="true">☕</span>Focus blocks and breaks</h3><p>40-minute focus blocks with enforced breaks. It notices fatigue and frustration and eases off before you burn out.</p></div>
      <div class="feature"><h3><span class="ico" aria-hidden="true">💡</span>Explains every decision</h3><p>A "Why this question?" panel shows exactly what the engine saw and why it chose this question for you.</p></div>
    </div>
  </div>
</section>

<section id="how">
  <div class="wrap">
    <div class="sec-head">
      <span class="eyebrow">How it works</span>
      <h2>Three steps. Zero setup.</h2>
    </div>
    <div class="steps">
      <div class="step"><h3>Pick your class or exam</h3><p>Class 1 to 12, JEE Advanced or NEET. Practise everything, or focus on one subject or chapter from the sidebar.</p></div>
      <div class="step"><h3>Answer and rate your confidence</h3><p>Sure, unsure or guessing. Your speed, answer changes and hints all feed the model, not just right or wrong.</p></div>
      <div class="step"><h3>Watch it adapt</h3><p>Weak chapters come back, easy ones fade, and the dashboard shows your strengths, error patterns and predicted score.</p></div>
    </div>
  </div>
</section>

<section id="proof">
  <div class="wrap">
    <div class="proof">
      <div>
        <span class="eyebrow">Results</span>
        <h2 style="font-size:clamp(28px,3.4vw,40px);margin:12px 0 10px;line-height:1.1">Reach exam readiness with fewer questions.</h2>
        <p class="muted">We ran simulated students with hidden ability in 6 chapters through PrepPilot and through a fixed question bank. Same students, same goal: every chapter exam-ready.</p>
        <div class="big-num">up to 44%</div>
        <p class="muted" style="margin-top:6px">fewer questions needed (21–44% across student profiles).</p>
      </div>
      <div class="bars" aria-label="Questions to readiness, adaptive vs static">
        <p class="muted" style="margin:0 0 6px;font-size:14px">Questions to readiness · lower is better</p>
        <div class="row"><span>Uneven student</span><div class="track"><div class="fill ours" style="width:56%"></div></div><b>235</b></div>
        <div class="row"><span class="muted">Static bank</span><div class="track"><div class="fill" style="width:100%"></div></div><span>421</span></div>
        <div class="row"><span>Weak all-round</span><div class="track"><div class="fill ours" style="width:74%"></div></div><b>312</b></div>
        <div class="row"><span class="muted">Static bank</span><div class="track"><div class="fill" style="width:100%"></div></div><span>422</span></div>
        <div class="row"><span>Nearly ready</span><div class="track"><div class="fill ours" style="width:79%"></div></div><b>140</b></div>
        <div class="row"><span class="muted">Static bank</span><div class="track"><div class="fill" style="width:100%"></div></div><span>177</span></div>
      </div>
    </div>
  </div>
</section>

<section id="faq">
  <div class="wrap faq" style="max-width:780px">
    <div class="sec-head"><span class="eyebrow">FAQ</span><h2>Questions, answered.</h2></div>
    <details><summary>Where do the questions come from?</summary><p>Three sources, clearly tagged: real JEE Advanced questions (2016–2023) with official answer keys via the JEEBench dataset, original questions modelled on JEE/NEET past-paper patterns with machine-checked answer keys, and AI-generated questions that must pass an independent second solve.</p></details>
    <details><summary>What if a question is wrong?</summary><p>Report it with one click. It's removed instantly and the answer won't count against you.</p></details>
    <details><summary>Does it work without the AI?</summary><p>Yes. The adaptive engine runs locally. AI is used for extra questions, personalised explanations and doubt answering.</p></details>
    <details><summary>Why can't I skip the break?</summary><p>Because studying tired wastes time. Breaks are 10 minutes after each 40-minute block, or 15 if you showed signs of fatigue.</p></details>
  </div>
</section>

<section>
  <div class="wrap">
    <div class="final">
      <h2>Study smarter starting today.</h2>
      <p>Free. Takes 20 seconds to start.</p>
      <a class="btn big" href="/login?mode=signup">Create your account →</a>
    </div>
  </div>
</section>

<footer>
  <div class="wrap">
    <span><b class="brand" style="font-size:16px">Prep<span style="color:var(--accent)">Pilot</span></b> · Built for PS 2: Adaptive AI Study Companion</span>
    <span>JEE Advanced questions: JEEBench (MIT), Arora et al., EMNLP 2023</span>
  </div>
</footer>
</body>
</html>
'''

# ---------------- login.html ----------------
FILES['login.html'] = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in · PrepPilot</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/site.css">
<style>
body { min-height: 100vh; display: grid; grid-template-columns: 1fr 1fr; }
.side { position: relative; padding: 40px; display: flex; flex-direction: column; justify-content: space-between; color: #fff;
        background: radial-gradient(circle at 15% 10%, #c2410c, #9a3412 45%, #4a1606); overflow: hidden; }
.side .brand { color: #fff; } .side .brand span { color: #ffd2c2; }
.side h2 { font-size: clamp(30px, 3.4vw, 44px); line-height: 1.1; margin: 0 0 14px; max-width: 460px; }
.side p { opacity: .9; max-width: 440px; margin: 0; font-size: 17px; }
.side ul { list-style: none; padding: 0; margin: 28px 0 0; display: grid; gap: 12px; font-size: 15px; }
.side li::before { content: "✓"; display: inline-grid; place-items: center; width: 22px; height: 22px; margin-right: 10px;
                   border-radius: 50%; background: rgba(255,255,255,.18); font-size: 12px; }
.quote { background: rgba(0,0,0,.18); border: 1px solid rgba(255,255,255,.15); border-radius: 16px; padding: 18px; font-size: 15px; }
.quote b { display: block; margin-top: 8px; font-size: 13px; opacity: .8; font-weight: 600; }

.main { display: grid; place-items: center; padding: 32px 16px; }
.box { width: min(400px, 100%); }
.box h1 { font-size: 32px; margin: 0 0 6px; }
.tabs { display: grid; grid-template-columns: 1fr 1fr; background: var(--chip); border-radius: 12px; padding: 4px; margin: 24px 0; }
.tabs button { border: 0; background: none; padding: 9px; border-radius: 9px; font-weight: 600; cursor: pointer; color: var(--muted); }
.tabs button.on { background: var(--panel); color: var(--ink); box-shadow: 0 1px 3px rgba(0,0,0,.12); }
label { display: block; font-size: 14px; font-weight: 600; margin: 14px 0 6px; }
input, select { width: 100%; padding: 12px 14px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); }
input:focus, select:focus { border-color: var(--accent); outline: none; box-shadow: 0 0 0 3px var(--sel); }
.pw { position: relative; }
.pw button { position: absolute; right: 8px; top: 50%; transform: translateY(-50%); border: 0; background: none; color: var(--muted);
             font-size: 13px; font-weight: 600; cursor: pointer; padding: 6px; }
.hint { font-size: 12.5px; color: var(--muted); margin-top: 6px; }
.err { min-height: 22px; color: var(--bad); font-size: 14px; margin: 12px 0 4px; }
.submit { width: 100%; margin-top: 4px; }
.or { display: flex; align-items: center; gap: 12px; color: var(--muted); font-size: 13px; margin: 20px 0; }
.or::before, .or::after { content: ""; flex: 1; height: 1px; background: var(--line); }
.guest { width: 100%; }
.fine { font-size: 12.5px; color: var(--muted); text-align: center; margin-top: 18px; }
.only-signup { display: none; }
form.signup .only-signup { display: block; }
@media (max-width: 860px) {
  body { grid-template-columns: 1fr; }
  .side { padding: 28px 20px; }
  .side ul, .quote { display: none; }
}
</style>
</head>
<body>
<aside class="side">
  <a class="brand" href="/">Prep<span>Pilot</span></a>
  <div>
    <h2>Your study partner that learns how you learn.</h2>
    <p>Every answer teaches PrepPilot something about you: what you know, why you slip, and when you need a break.</p>
    <ul>
      <li>Adaptive questions from Class 1 to JEE Advanced and NEET</li>
      <li>Diagnosis of every mistake, with a fix</li>
      <li>Live predicted score and exam strategy</li>
    </ul>
  </div>
  <div class="quote">"Most apps tell you that you're wrong. This one tells you <i>why</i>, and what to do next."<b>The idea behind PrepPilot</b></div>
</aside>

<main class="main">
  <div class="box">
    <h1 id="title">Welcome back</h1>
    <p class="muted" id="subtitle">Sign in to continue your practice.</p>
    <div class="tabs" role="tablist">
      <button type="button" role="tab" data-mode="signin" class="on">Sign in</button>
      <button type="button" role="tab" data-mode="signup">Create account</button>
    </div>
    <form id="form" novalidate>
      <div class="only-signup">
        <label for="name">Full name</label>
        <input id="name" autocomplete="name" placeholder="Aarav Sharma">
      </div>
      <label for="email">Email</label>
      <input id="email" type="email" autocomplete="email" placeholder="you@example.com" required>
      <label for="password">Password</label>
      <div class="pw">
        <input id="password" type="password" autocomplete="current-password" placeholder="••••••••" required minlength="8">
        <button type="button" id="show" aria-label="Show password">Show</button>
      </div>
      <div class="hint only-signup">At least 8 characters.</div>
      <div class="only-signup">
        <label for="level">What are you preparing for?</label>
        <select id="level">
          <optgroup label="School">
            <option>Class 1</option><option>Class 2</option><option>Class 3</option><option>Class 4</option>
            <option>Class 5</option><option>Class 6</option><option>Class 7</option><option>Class 8</option>
            <option>Class 9</option><option selected>Class 10</option><option>Class 11</option><option>Class 12</option>
          </optgroup>
          <optgroup label="Competitive exams">
            <option value="JEE">JEE Advanced</option><option value="NEET">NEET</option>
          </optgroup>
        </select>
      </div>
      <div class="err" id="err" role="alert"></div>
      <button class="btn primary big submit" id="go">Sign in</button>
    </form>
    <div class="or">or</div>
    <button class="btn big guest" id="guest" type="button">Continue as guest</button>
    <p class="fine">Guest progress isn't saved to an account. <a href="/">Back to home</a></p>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
let mode = new URLSearchParams(location.search).get("mode") === "signup" ? "signup" : "signin";

function setMode(m) {
  mode = m;
  $("#form").classList.toggle("signup", m === "signup");
  document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.mode === m));
  $("#title").textContent = m === "signup" ? "Create your account" : "Welcome back";
  $("#subtitle").textContent = m === "signup" ? "Free forever. Takes 20 seconds." : "Sign in to continue your practice.";
  $("#go").textContent = m === "signup" ? "Create account" : "Sign in";
  $("#password").autocomplete = m === "signup" ? "new-password" : "current-password";
  $("#err").textContent = "";
}

function enter(user, token) {
  try {
    localStorage.setItem("pp-user", JSON.stringify(user));
    if (token) localStorage.setItem("pp-token", token); else localStorage.removeItem("pp-token");
    if (token && user.level) localStorage.setItem("pp-level", user.level);  // start on the level they chose
  } catch (e) {}
  location.href = "/setup";  // choose class, difficulty and focus time before each new visit
}

$("#form").onsubmit = async e => {
  e.preventDefault();
  const email = $("#email").value.trim(), password = $("#password").value, name = $("#name").value.trim();
  if (mode === "signup" && !name) return $("#err").textContent = "Enter your name.";
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return $("#err").textContent = "Enter a valid email address.";
  if (password.length < 8) return $("#err").textContent = "Password must be at least 8 characters.";
  $("#go").disabled = true; $("#err").textContent = "";
  try {
    const body = mode === "signup" ? { name, email, password, level: $("#level").value } : { email, password };
    const r = await fetch(mode === "signup" ? "/api/signup" : "/api/login",
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Something went wrong.");
    enter(d.user, d.token);
  } catch (err) { $("#err").textContent = err.message; }
  $("#go").disabled = false;
};

$("#show").onclick = () => {
  const pw = $("#password"), hidden = pw.type === "password";
  pw.type = hidden ? "text" : "password";
  $("#show").textContent = hidden ? "Hide" : "Show";
  $("#show").setAttribute("aria-label", hidden ? "Hide password" : "Show password");
};
$("#guest").onclick = () => enter({ name: "Guest", guest: true, email: "guest-" + Date.now() + "-" + Math.random().toString(36).slice(2) });
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => setMode(b.dataset.mode));
setMode(mode);
</script>
</body>
</html>
'''

# ---------------- site.css ----------------
FILES['site.css'] = r'''/* Shared look for the landing and login pages; same tokens as the app. */
:root {
  --bg: #f5f4ef; --panel: #ffffff; --ink: #17191e; --muted: #5f6571; --line: #e3e1d9;
  --chip: #eeece5; --accent: #d9481f; --accent-ink: #ffffff; --good: #1d8348; --bad: #c0392b;
  --warn: #a8690f; --sel: #fde9e1; --glow: rgba(217, 72, 31, .14);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115; --panel: #171a21; --ink: #e8e9ed; --muted: #9ba2ae; --line: #2a2f39;
    --chip: #222731; --accent: #ff6a3d; --accent-ink: #14100e; --good: #3ccf7a; --bad: #ff6b5e;
    --warn: #f0b44c; --sel: #3a2219; --glow: rgba(255, 106, 61, .16);
  }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.6 "Instrument Sans", system-ui, sans-serif; }
h1, h2, h3, .brand { font-family: "Bricolage Grotesque", "Instrument Sans", sans-serif; letter-spacing: -0.02em; }
a { color: inherit; }
button, input, select { font: inherit; color: inherit; }
.wrap { width: min(1140px, 100% - 32px); margin-inline: auto; }

/* nav */
.nav { position: sticky; top: 0; z-index: 10; background: color-mix(in srgb, var(--bg) 85%, transparent);
       backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
.nav .wrap { display: flex; align-items: center; gap: 24px; height: 64px; }
.brand { font-size: 22px; font-weight: 700; text-decoration: none; }
.brand span { color: var(--accent); }
.nav-links { display: flex; gap: 22px; margin-left: auto; font-size: 15px; }
.nav-links a { text-decoration: none; color: var(--muted); }
.nav-links a:hover { color: var(--ink); }

/* buttons */
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px; border: 1px solid var(--line);
       background: var(--panel); border-radius: 12px; padding: 10px 18px; font-weight: 600; font-size: 15px;
       text-decoration: none; cursor: pointer; transition: transform .15s, border-color .15s, box-shadow .15s; }
.btn:hover { border-color: var(--accent); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); box-shadow: 0 2px 6px rgba(0, 0, 0, .15); }
.btn.primary:hover { transform: translateY(-1px); }
.btn.big { padding: 14px 24px; font-size: 16px; border-radius: 14px; }
.btn:focus-visible, input:focus-visible, select:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.eyebrow { display: inline-flex; gap: 8px; align-items: center; font-size: 13px; font-weight: 600; color: var(--accent);
           background: var(--sel); padding: 5px 12px; border-radius: 99px; }
.muted { color: var(--muted); }

footer { border-top: 1px solid var(--line); padding: 28px 0; font-size: 14px; color: var(--muted); }
footer .wrap { display: flex; gap: 16px; flex-wrap: wrap; justify-content: space-between; }

@media (max-width: 760px) {
  .nav-links { display: none; }
  .nav .wrap { gap: 12px; }
  .nav .btn { margin-left: auto; padding: 8px 14px; }
}
'''


def run():
    parser = argparse.ArgumentParser(description="PrepPilot adaptive quiz prototype")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("Install dependencies: python -m pip install fastapi==0.162.0 uvicorn==0.54.0")
    with tempfile.TemporaryDirectory(prefix="preppilot-") as temporary:
        root = Path(temporary)
        for name, content in FILES.items():
            (root / name).write_text(content, encoding="utf-8")
        sys.path.insert(0, str(root))
        import main
        # Keep account data beside this launcher, outside the temporary source folder.
        main.USERS_FILE = Path(__file__).resolve().parent / "quizbot_users.json"
        print(f"Open http://localhost:{args.port}")
        uvicorn.run(main.app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
