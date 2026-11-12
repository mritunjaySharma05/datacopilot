"""Answerability features: numeric signals for "can this question be answered from this data?".

`extract_features(question, schema, context)` returns a dict with one value per name in
FEATURE_NAMES. `schema` is the profiler's {table: [{"name", "type", "samples"}]}. The optional
`context` (see `build_context`) adds what 3 sample values per column cannot: the dataset's
year range and the vocabulary of values in its text columns.

The classifier itself is trained in Module 4; this module only computes features.
"""
import difflib
import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------ word lists

STOPWORDS = frozenset("""
a an the of in on at to for from by with and or not no is are was were be been being do does did
has have had it its this that these those there their them they he she his her as into than then
so if all any each every per vs versus about over under between among across during since until
me my i you your we our us can could would will just only also more less very
""".split())

# Words that describe the analysis, not the data: neither matched nor "unmatched".
ANALYSIS_WORDS = frozenset("""
how many much what which who whom whose when where why total sum average avg mean median count
number numbers most least highest lowest largest smallest biggest top bottom first last best worst
max maximum min minimum percent percentage share rate ratio compared compare against difference
each every per year years season seasons month months day days time times list show give find
data dataset record records value values recorded ever overall all-time including excluding
counting only among least within between rank ranked order ordered group grouped distinct different
increase increased decrease decreased grew growth change changed drop dropped rise fell more
contain contains contained placed held made won win wins scored score lost
""".split())

# Calendar words: generic time references, not data values or entities.
TIME_WORDS = frozenset("""
monday tuesday wednesday thursday friday saturday sunday january february march april may june
july august september october november december week weekday weekend hour hours morning evening
""".split())

# A 4-digit number followed by one of these is a quantity, not a year ("2000 balls").
UNIT_WORDS = frozenset("""
ball balls run runs hectare hectares ha tonne tonnes ton tons kg km review reviews match matches
stop stops race races item items order orders product products customer customers wicket wickets
point points lap laps record records row rows rupees reais
""".split())

# Dataset / domain names that are not values in the data.
DOMAIN_WORDS = frozenset("ipl f1 formula olist india indian brazil brazilian cricket".split())

WRITE_VERBS = frozenset("""
delete remove drop insert add update change set rename create truncate empty clear alter modify
replace overwrite erase wipe
""".split())

OPINION_WORDS = frozenset("""
best worst greatest good bad better worse beautiful nicest coolest interesting exciting boring
overrated underrated fair unfair trustworthy loyal dramatic stylish favourite favorite promising
important healthier healthy should ought recommend worth happy happier
""".split())

# Phrases that point to facts outside any table (ownership, people's roles, "now", the world).
EXTERNAL_CUES = (
    "who is", "who owns", "ceo", "owner", "principal", "coach", "minister", "current", "currently",
    "today", "this year", "net worth", "population", "capital of", "born", "exchange rate",
    "inflation", "gdp", "market share", "brand value", "prize money", "social media", "followers",
    "broadcast", "tv", "sponsor", "regulation", "government", "scheme", "designed", "usually",
    "ticket", "salary", "cost to build",
)

# Roman-script Hindi function words (none of them are also common English words).
HINGLISH_MARKERS = frozenset("""
kitne kitna kitni kaun kaunsa kaunsi kis kise kisne kya kab kahan mein hai hain tha thi sabse
zyada jyada kam wale wali wala aur saal har ne ko ka ki ke se gaya gayi gaye hue hua bataye batao
abhi bhi jo unme jinke jin liye karne karke chhod tak beech pehle baad
""".split())

YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b(?![,.]\d)")
YEAR_WITH_NEXT_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b(?![,.]\d)\s*([A-Za-z]*)")
NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

FUZZY_CUTOFF = 0.85
MIN_FUZZY_LEN = 5

FEATURE_NAMES = (
    "n_words", "n_content_words", "schema_overlap", "value_overlap", "unmatched_content_words",
    "unmatched_share", "has_write_verb", "starts_with_write_verb", "mentions_table_word",
    "n_opinion_words", "n_external_cues", "n_years", "year_out_of_range", "years_outside_by",
    "n_numbers", "n_entities", "n_unmatched_entities", "qtype_how_many", "qtype_which_what",
    "qtype_who", "qtype_when_where", "qtype_yes_no", "qtype_imperative", "hinglish_share",
    "ends_with_question_mark",
)


# ------------------------------------------------------------------ context

@dataclass(frozen=True)
class DatasetContext:
    """Facts about a dataset that the schema samples alone don't give."""

    year_min: int | None = None
    year_max: int | None = None
    value_tokens: frozenset = field(default_factory=frozenset)


def find_years(question: str) -> list[int]:
    """Years mentioned in the text; '2000 balls' or '1,500' are quantities, not years."""
    return [int(y) for y, nxt in YEAR_WITH_NEXT_RE.findall(question) if nxt.lower() not in UNIT_WORDS]


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens without possessive 's."""
    return [re.sub(r"['’]s$", "", w.lower()).strip("'’-") for w in WORD_RE.findall(text)]


def stem(word: str) -> str:
    """Very light plural / suffix stripping so 'states' matches 'state'."""
    for suffix, keep in (("ies", "y"), ("s", "")):
        if len(word) > 4 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)] + keep
    return word


def schema_tokens(schema: dict[str, list[dict]]) -> set[str]:
    """Stemmed tokens of all table and column names ('customer_state' -> customer, state)."""
    names = list(schema) + [c["name"] for cols in schema.values() for c in cols]
    return {stem(t) for n in names for t in re.split(r"[^a-z0-9]+", n.lower()) if t}


def sample_tokens(schema: dict[str, list[dict]]) -> set[str]:
    """Tokens from the schema's sample values (fallback when no context is given)."""
    return {stem(t) for cols in schema.values() for c in cols for s in c["samples"]
            if isinstance(s, str) for t in tokenize(s)}


_TEXT_TYPES = ("CHAR", "TEXT", "STRING")
_DATE_TYPES = ("DATE", "TIME")
MAX_DISTINCT_VALUES = 20_000
MIN_YEAR_SHARE = 0.001  # a year needs >= 0.1% of a column's rows to count as covered


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def column_years(conn, table: str, column: str, min_share: float = MIN_YEAR_SHARE) -> list[int]:
    """Years present in a date/year column, ignoring rare outlier years (< min_share of rows).

    E.g. Olist has 4 of 112k shipping_limit_date values in 2020; they should not stretch
    the dataset's year range to 2020.
    """
    c, t = _q(column), _q(table)
    rows = conn.execute(f"SELECT substr(CAST({c} AS TEXT), 1, 4) AS y, COUNT(*) FROM {t} "
                        f"WHERE {c} IS NOT NULL GROUP BY y").fetchall()
    total = sum(n for _, n in rows)
    return [int(y) for y, n in rows
            if y and str(y).isdigit() and 1800 <= int(y) <= 2100 and n >= min_share * total]


def build_context(conn, schema: dict[str, list[dict]], max_distinct: int = MAX_DISTINCT_VALUES) -> DatasetContext:
    """Year range and value vocabulary of a database (DuckDB or sqlite3 connection).

    Years come from date/time columns and integer columns named like *year* / season
    (rare outlier years ignored, see `column_years`).
    Values come from text columns with at most `max_distinct` distinct values (ids and
    free text are skipped).
    """
    years: list[int] = []
    tokens: set[str] = set()
    for table, cols in schema.items():
        for col in cols:
            name, ctype = col["name"], str(col["type"]).upper()
            c, t = _q(name), _q(table)
            is_year_int = "INT" in ctype and ("year" in name.lower() or name.lower() == "season")
            if is_year_int or any(k in ctype for k in _DATE_TYPES):
                years += column_years(conn, table, name)
            elif any(k in ctype for k in _TEXT_TYPES) and not name.lower().endswith("id"):
                n = conn.execute(f"SELECT COUNT(DISTINCT {c}) FROM {t}").fetchone()[0]
                if n and n <= max_distinct:
                    for (value,) in conn.execute(f"SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL").fetchall():
                        tokens.update(stem(w) for w in tokenize(str(value)))
    return DatasetContext(min(years) if years else None, max(years) if years else None, frozenset(tokens))


# ------------------------------------------------------------------ matching helpers

def _matches(word: str, vocab: set[str] | frozenset, vocab_list: list[str] | None = None) -> bool:
    """Exact (stemmed) match, or a close spelling for longer words."""
    w = stem(word)
    if w in vocab:
        return True
    if len(w) >= MIN_FUZZY_LEN and vocab_list is not None:
        return bool(difflib.get_close_matches(w, vocab_list, n=1, cutoff=FUZZY_CUTOFF))
    return False


def entities(question: str) -> list[list[str]]:
    """Runs of capitalized / all-caps words (skipping the question's first word): 'Kerala Blasters'."""
    words = [re.sub(r"['’]s$", "", w).strip("'’-") for w in WORD_RE.findall(question)]
    runs, current = [], []
    for i, w in enumerate(words):
        capital = w[:1].isupper() and (i > 0 or w.lower() not in STOPWORDS | ANALYSIS_WORDS | WRITE_VERBS)
        if capital and i == 0 and w.lower() in OPINION_WORDS | HINGLISH_MARKERS:
            capital = False
        if capital:
            current.append(w.lower())
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [r for r in runs if not all(t in DOMAIN_WORDS | TIME_WORDS for t in r)]


def question_type(question: str) -> dict[str, int]:
    """One-hot question type (English and Hinglish cues)."""
    q = question.lower().strip()
    first = (tokenize(q) or [""])[0]
    types = {
        "qtype_how_many": bool(re.search(r"\bhow (many|much)\b|\bkit(ne|na|ni)\b", q)),
        "qtype_who": bool(re.search(r"^\s*who\b|\bwho (is|was|are|won|scored|owns|set|led)\b", q)),
        "qtype_when_where": bool(re.search(r"^\s*(when|where|on what date|in which year)\b|\b(kab|kahan)\b", q)),
        "qtype_yes_no": first in {"is", "are", "was", "were", "do", "does", "did", "should", "can", "could", "would"},
        "qtype_imperative": first in WRITE_VERBS | {"list", "show", "give", "find", "tell", "get"},
    }
    types["qtype_which_what"] = not any(types.values()) and bool(
        re.search(r"\b(which|what|kaun|kaunsa|kaunsi|kis|kya)\b", q))
    return {k: int(v) for k, v in types.items()}


# ------------------------------------------------------------------ main entry point

def extract_features(question: str, schema: dict[str, list[dict]],
                     context: DatasetContext | None = None) -> dict[str, float]:
    """Numeric answerability features for one question against one dataset's schema."""
    context = context or DatasetContext(value_tokens=frozenset(sample_tokens(schema)))
    names = schema_tokens(schema)
    names_list, values_list = sorted(names), sorted(context.value_tokens)
    words = tokenize(question)
    lower = question.lower()

    content = [w for w in words if w not in STOPWORDS and w not in ANALYSIS_WORDS and w not in TIME_WORDS
               and w not in DOMAIN_WORDS and w not in HINGLISH_MARKERS and len(w) > 1]
    in_schema = [w for w in content if _matches(w, names, names_list)]
    in_values = [w for w in content if _matches(w, context.value_tokens, values_list)]
    unmatched = [w for w in content if w not in in_schema and w not in in_values]

    years = find_years(question)
    outside = 0
    if years and context.year_min is not None:
        outside = max(max(context.year_min - y, y - context.year_max, 0) for y in years)
    numbers = [n for n in NUMBER_RE.findall(question) if not (YEAR_RE.fullmatch(n) and int(n) in years)]

    ents = entities(question)
    known = names | set(context.value_tokens)
    unmatched_ents = [e for e in ents if not all(_matches(t, known, values_list) for t in e if t not in DOMAIN_WORDS)]

    n_words = len(words)
    features = {
        "n_words": n_words,
        "n_content_words": len(content),
        "schema_overlap": len(in_schema) / len(content) if content else 0.0,
        "value_overlap": len(in_values) / len(content) if content else 0.0,
        "unmatched_content_words": len(unmatched),
        "unmatched_share": len(unmatched) / len(content) if content else 0.0,
        "has_write_verb": int(any(w in WRITE_VERBS for w in words)),
        "starts_with_write_verb": int(bool(words) and words[0] in WRITE_VERBS),
        "mentions_table_word": int(bool(re.search(r"\b(table|column|row|rows|record|records)\b", lower))
                                   and any(w in WRITE_VERBS for w in words)),
        "n_opinion_words": sum(w in OPINION_WORDS for w in words),
        "n_external_cues": sum(bool(re.search(rf"\b{re.escape(c)}\b", lower)) for c in EXTERNAL_CUES),
        "n_years": len(years),
        "year_out_of_range": int(outside > 0),
        "years_outside_by": outside,
        "n_numbers": len(numbers),
        "n_entities": len(ents),
        "n_unmatched_entities": len(unmatched_ents),
        **question_type(question),
        "hinglish_share": sum(w in HINGLISH_MARKERS for w in words) / n_words if n_words else 0.0,
        "ends_with_question_mark": int(question.strip().endswith("?")),
    }
    return {name: float(features[name]) for name in FEATURE_NAMES}
