# Module 6 plan: notes and known issues

Started 2026-11-23. Module 6 milestone: end-to-end integration, charts, auto insights
(`docs/planning/milestones.md`). This file collects issues found earlier that Module 6 should
handle; the full plan comes later. Nothing here has been changed in the pipeline yet.

## 1. IPL: "Who won IPL 2016?" needs the final, which has no `stage` column

**Data check (2026-11-23).** `ipl_matches` has no `stage` / playoff column. Its columns are
`match_id, team1, team2, date, season, event, match_number, venue, city, toss_winner,
toss_decision, player_of_match, winner, winner_runs, winner_wickets, outcome, eliminator,
method, team1_franchise, team2_franchise, toss_winner_franchise, winner_franchise`; `event` is
always "Indian Premier League".

What does mark the final:
- **Playoff matches have no `match_number`** (NULL): 4 per season from 2010 (qualifiers,
  eliminator, final), 3 in 2008 and 2009. League matches are numbered.
- **The final is the season's last match by date**, and it is the only match on that date.
  Verified for all 19 seasons 2008-2026; the result matches the known champions (2016 Sunrisers
  Hyderabad, 2008 Rajasthan Royals, 2025 Royal Challengers Bengaluru, ...).
- No final in the data was tied; for a tie, the super-over winner would be in `eliminator`
  (Cricsheet), so `COALESCE(winner, eliminator)` covers that case.

**Correct SQL** (DuckDB, returns "Sunrisers Hyderabad"):

```sql
SELECT COALESCE(winner, eliminator) AS champion
FROM ipl_matches
WHERE season = 2016 AND match_number IS NULL
ORDER BY date DESC
LIMIT 1;
```

For all seasons:

```sql
SELECT season, COALESCE(winner, eliminator) AS champion
FROM ipl_matches m
WHERE match_number IS NULL
  AND date = (SELECT MAX(date) FROM ipl_matches x WHERE x.season = m.season)
ORDER BY season;
```

`winner` is the team name of that season (e.g. "Royal Challengers Bangalore" 2008-2023,
"Royal Challengers Bengaluru" from 2024); `winner_franchise` groups renamed teams if a question asks per franchise.

**Why the LLM gets it wrong:** nothing in the schema says "final", so a model tends to count
wins per team in the season or pick the last row by `match_id`. Module 6 options (to decide
there, then evaluate on new train-split questions, never test or held-out):
1. a short per-dataset domain note in the prompt, the way BIRD passes `evidence`
   (`build_prompt(..., evidence=...)` already supports it), e.g. "Playoff matches have no
   match_number; a season's final is its last match by date";
2. or add a derived `stage` column (league / playoff / final) in `scripts/prepare_data.py`.
   This changes the processed data, so `build_questions.py --check` must still pass.

## 2. Player names: "Kohli" vs "V Kohli"

Seen in Module 5 testing: "kohli ne kitne runs banaye?" produced valid SQL with
`striker = 'Virat Kohli'`, which matches nothing (the data has "V Kohli"), so the answer was
empty. Same root as the classifier's vocabulary gap (`docs/results/README.md`): names in the
question do not match how the data spells them. Option: look up candidate values (e.g. a
case-insensitive `LIKE '%kohli%'` over player columns) and give the matches to the LLM.

## 3. Classifier vocabulary gap

Recorded 2026-11-18 (`docs/notes/handoff.md`, `docs/results/README.md`): computed metrics
("sixes", "economy", "podiums") and lowercase names give `schema_overlap` 0, so short questions
about them are refused. Fix with a per-dataset metric vocabulary or case-insensitive value
matching, validated on new questions, not the held-out set.
