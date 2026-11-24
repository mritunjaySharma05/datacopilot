# DataCopilot user guide

DataCopilot answers questions about four public datasets in **English or Hinglish**. An AI model
writes a SQL query, the query is checked, and the database computes the answer; you see the
result table and the exact SQL that produced it.

## 1. Sign in

1. Open the address you were given (for the project demo: `https://datacopilot.duckdns.org`).
   The padlock in the address bar shows the connection is encrypted.
2. Your browser asks for a **username and password**. Use the ones the project owner gave you.
   If you type them wrong, the browser asks again.
3. The DataCopilot page opens. The sidebar shows which AI model the server uses.

The demo server is sometimes stopped to save cost. If the page does not load at all, ask the
project owner to start it (it takes 2-3 minutes).

## 2. Pick a dataset

Choose one in the **Dataset** list. Each question is answered from the chosen dataset only.

| Dataset | What it contains | Example question |
|---|---|---|
| `crop` | Indian crop production by state, district, season and crop, 1997-2015 (33 states and union territories) | Which state had the highest total rice production? |
| `ipl` | Indian Premier League matches and ball-by-ball deliveries, seasons 2008-2026 | Which batter scored the most runs in IPL history? |
| `f1` | Formula 1 races, results, laps, pit stops and standings, seasons 2014-2026 | How many races were held in 2021? |
| `olist` | Brazilian e-commerce (Olist): about 99,000 orders 2016-2018, customers, products, payments, reviews | How many orders were delivered? |

## 3. Ask a question

Type the question in the **Question** box (up to 500 characters) and click **Ask**. The
grey example in the empty box shows the kind of question that works.

- **English:** "Which venue has hosted the most matches?"
- **Hinglish:** "Kis state mein sabse zyada customers hain?", "Har season mein kitne matches khele gaye?"

Tips:
- Ask about what is in the data: counts, totals, averages, top-N, per year / season / state.
- Use names as they appear in the data when you can (for example IPL players are stored like
  "V Kohli"); a different spelling can give an empty answer.
- One question at a time works best; follow-up questions do not remember the previous one.

## 4. Read the answer

After a few seconds you see:

1. **A green line** with the number of rows, then the **result table**. At most 200 rows are
   shown (the query itself stops at 1,000 rows).
2. **The SQL** that produced the table. Every number comes from running this query on the data;
   the AI never calculates numbers itself. You can check what was counted, filtered or grouped.
3. **A grey line** at the bottom: the gate score `P(answerable)`, how long the AI and the whole
   request took, and a request id (useful when reporting a problem).

Read the SQL when an answer looks surprising: the question may have been understood differently
(for example "delivered orders" counted by delivery date instead of order status).

If you see **"No answer: ..."** (yellow), the AI's SQL was rejected by the safety check or failed
twice (one automatic retry); the SQL and the reason are shown. Try rephrasing the question.

## 5. Why a question can be refused

Before any SQL is written, a small classifier (the "answerability gate") checks whether the
question can be answered from the chosen dataset. If not, you see a blue message:

> This question does not look answerable from the crop data (P(answerable) = 0.05), so no SQL was run.

Typical reasons:

| Kind of question | Example | Why |
|---|---|---|
| Information the data does not have | "What was the rainfall in Kerala?" (crop) | No such column |
| Outside the data's years | "Rice production in 2022?" (crop ends in 2015) | Out of range |
| Outside knowledge | "Who is the agriculture minister of India?" | Not in the dataset |
| Opinion | "Which is the best IPL team?" | Subjective |
| Changing data | "Delete all rows for Kerala" | Only reading is allowed; data is never modified |

The gate is right about 87% of the time, so a sensible question is occasionally refused (short
questions about computed things like "sixes" or "economy" are the most common case). Rephrase
it with the words and names used in the data, for example "How many deliveries by V Kohli had
runs_off_bat = 6?" instead of "kohli ke sixes".

## 6. "Rate limit" message

The demo uses a free AI service with a per-minute limit. If several questions are asked quickly
you may see a red message like:

> HTTP 429: Groq rate limit reached (HTTP 429). Try again in 7 s.

Nothing is wrong: wait the given number of seconds and click **Ask** again.

Other red messages:
- **"Cannot reach the API ..."**: the server is starting or restarting; wait a minute and reload.
- **"HTTP 503 ..."**: the AI service is not available; try again later or tell the project owner
  (they get an alert automatically).

## 7. Privacy

Your question and the generated SQL are logged on the server for 14 days to monitor the service.
To write the SQL, the question, the dataset's table and column names and a few sample values are
sent to the AI provider (Groq). Do not type personal or confidential information into questions.
