# Data sources (Module 3)

Raw files are downloaded by `python scripts/download_datasets.py` into
`data/raw/<name>/`, which is gitignored. They are never committed.
The BIRD benchmark is documented separately in [datasets.md](datasets.md).

Checked on 2026-11-11. Sizes are what the script downloaded on that date.

| Dataset | Folder | Licence | Download size | On disk |
|---|---|---|---|---|
| Olist Brazilian E-commerce | `data/raw/olist/` | CC BY-NC-SA 4.0 | 44.7 MB zip | 126 MB CSV |
| IPL (Cricsheet) | `data/raw/ipl/` | ODC-By 1.0 | 6.9 MB zip | 95 MB CSV |
| Crop production (Kaggle mirror of data.gov.in) | `data/raw/crop_production/` | GODL-India (underlying data); mirror states none | 15.3 MB CSV (manual) | 15.3 MB CSV |
| F1 World Championship (Jolpica-F1) | `data/raw/f1/` | CC BY-NC-SA 4.0 | 14.7 MB zip | 45 MB CSV |

## Licence notes for this project

- **Non-commercial (Olist, F1):** both are CC BY-NC-SA 4.0. Fine for this academic
  capstone. Give attribution, don't use them commercially, and keep the raw files out
  of the public repo and the deployed app's downloadable content.
- **Attribution (all four):** the final report and the app's About page should credit
  Olist, Cricsheet, data.gov.in (Ministry of Agriculture and Farmers Welfare) and
  Jolpica-F1, with links. For crop data, also mention the Kaggle mirror we used.

---

## 1. Olist Brazilian E-commerce

- **Official source:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
  (published by Olist on Kaggle).
- **Licence:** CC BY-NC-SA 4.0 (from the Kaggle dataset metadata).
- **How the script gets it:** Kaggle's public download endpoint
  `https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce`.
  On 2026-11-11 it worked without a login. If Kaggle starts requiring one, the
  script prints these manual steps:
  1. Sign in at https://www.kaggle.com (free account).
  2. Open the dataset page and click **Download**.
  3. Unzip `archive.zip` into `data/raw/olist/` (the 9 CSVs directly inside).
  4. Or use the Kaggle CLI with an API token in `~/.kaggle/kaggle.json`:
     `kaggle datasets download olistbr/brazilian-ecommerce -p data/raw/olist --unzip`
- **Coverage:** about 99k orders, 2016-09 to 2018-10.
- **Files (9 CSVs):**

| File | Size | Rows |
|---|---|---|
| olist_customers_dataset.csv | 9.0 MB | 99,441 |
| olist_geolocation_dataset.csv | 61.3 MB | 1,000,163 |
| olist_order_items_dataset.csv | 15.4 MB | 112,650 |
| olist_order_payments_dataset.csv | 5.8 MB | 103,886 |
| olist_order_reviews_dataset.csv | 14.5 MB | 104,719 |
| olist_orders_dataset.csv | 17.7 MB | 99,441 |
| olist_products_dataset.csv | 2.4 MB | 32,951 |
| olist_sellers_dataset.csv | 0.2 MB | 3,095 |
| product_category_name_translation.csv | <0.1 MB | 71 |

## 2. IPL data (Cricsheet)

- **Official source:** https://cricsheet.org/downloads/ (Indian Premier League,
  CSV "new" format): `https://cricsheet.org/downloads/ipl_csv2.zip`.
- **Licence:** Open Data Commons Attribution License (ODC-By 1.0), per
  https://cricsheet.org/licensing/. `LICENSE.txt` is included in the zip.
  (Cricsheet's XML data is CC BY 4.0. We don't use it.)
- **No login needed.**
- **Coverage:** 1,243 matches, seasons 2007/08 to 2026 (19 seasons), men's IPL.
- **Files:**
  - `<match_id>.csv` (1,243): ball-by-ball, one row per delivery. Columns:
    match_id, season, start_date, venue, innings, ball, batting_team, bowling_team,
    striker, non_striker, bowler, runs_off_bat, extras, wides, noballs, byes, legbyes,
    penalty, wicket_type, player_dismissed, fielders, and others.
  - `<match_id>_info.csv` (1,243): match metadata (teams, toss, result, venue,
    officials, players).
  - `all_matches.csv`: all deliveries in one file.
  - `README.txt`, `LICENSE.txt`.

## 3. Crop production (Kaggle mirror of data.gov.in)

### Copy in use: Kaggle mirror (downloaded manually, 2026-11-11)

- **Source used:** https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
  ("Crop Production in India" by Abhinand, last updated 2019-11-12). Downloaded by
  hand from Kaggle and saved as `data/raw/crop_production/crop_production.csv`.
  `scripts/download_datasets.py crop` skips it because the file already exists.
- **Licence on the mirror:** Kaggle lists "Other (specified in description)", but
  the description does not name any licence.
- **Licence of the underlying data:** the rows are the data.gov.in "District-wise,
  season-wise crop production statistics" (see below), published under the
  **Government Open Data License - India (GODL-India)**. GODL allows use and
  adaptation with attribution, so we credit data.gov.in / Ministry of Agriculture and
  Farmers Welfare as the source and mention the Kaggle mirror.
- **Provenance caveat:** the mirror says its content was "taken from data.world"
  (`data.world/thatzprem/agriculture-india`), not directly from data.gov.in. That
  data.world page has since been retired, so this middle step can't be verified.
  Treat the mirror as an unofficial copy of the official dataset.
- **File:** `crop_production.csv`, 15.3 MB (15,316,741 bytes, same size as listed
  on Kaggle), 246,091 rows.
- **Columns:** `State_Name`, `District_Name`, `Crop_Year`, `Season`, `Crop`,
  `Area`, `Production`. The file has no units; the official series reports area in
  hectares and production in tonnes. Check units per crop during EDA before
  comparing crops with each other.
- **Coverage:** crop years 1997 to 2015; 33 states/UTs, 646 districts, 124 crops,
  6 seasons.
- **Known gaps:** `Production` is missing in 3,730 rows (1.5%). Coverage stops at
  2015, while the official data.gov.in series has later years.
- **Granularity:** district-level, with the state on every row, so state-wise
  production is a `GROUP BY` over the same file.

### Official source (for a later refresh)

- **Official dataset:** "District-wise, season-wise crop production statistics from
  1997", Ministry of Agriculture and Farmers Welfare, on the Open Government Data
  Platform India:
  https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997
  (catalog: https://www.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0).
- **Licence:** Government Open Data License - India (GODL-India), as stated on the
  resource page.
- **API resource ID:** `35be999b-0208-4354-b557-f6ca9a5355de`
  (`https://api.data.gov.in/resource/<id>`).
- **Needs a free API key.** The API rejects requests without a key, and the public
  demo key in data.gov.in's docs is shared and rate-limited. To replace the mirror
  with the official data later, delete the mirror file (or pass `--force`), then:
  1. Register or sign in at https://data.gov.in (top right: Login, then Sign up).
  2. Open **My Account** in the profile menu and generate an API key.
  3. Add it to `.env`: `DATA_GOV_IN_API_KEY=<your key>` (placeholder is in `.env.example`).
  4. Run `python scripts/download_datasets.py crop`. The script pages through the API
     and writes `data/raw/crop_production/crop_production.csv`.

  **Alternative without the API:** on the resource page click **Download**, choose
  CSV, fill in the short usage form, and save the file as
  `data/raw/crop_production/crop_production.csv`.
  Check that the column names match the mirror's before switching, since the
  official export may name them differently.

## 4. F1 World Championship (Jolpica-F1)

- **Official source:** Jolpica-F1 (https://github.com/jolpica/jolpica-f1), the
  community successor to the Ergast API, which shut down at the start of 2025.
  Database dumps: `https://api.jolpi.ca/data/dumps/download/` (index), and the free
  CSV dump at `https://api.jolpi.ca/data/dumps/download/delayed/?dump_type=csv`.
- **Licence:** data is CC BY-NC-SA 4.0 for non-commercial use (Jolpica Terms of Use,
  `TERMS.md`). Commercial use needs a supporter key. The code repo itself is Apache 2.0.
- **No login needed** for the free tier. The free dump lags up to 14 days; the
  2026-11-11 download was the dump uploaded on 2026-10-27.
- **Coverage:** every season from 1950 to 2026. "Recent seasons" (for example the
  hybrid era from 2014) will be selected during data preparation (`data/processed/`),
  not at download time.
- **Schema:** Jolpica's own model, not the old Ergast or Kaggle F1 CSV layout. Schema
  docs: https://dbdocs.io/jolpica/jolpica-f1. Some status columns are integer enums
  defined in the Jolpica source code.
- **Files (18 CSVs):**

| File | Rows |
|---|---|
| formula_one_lap.csv | 740,109 (37 MB) |
| formula_one_sessionentry.csv | 51,983 |
| formula_one_driverchampionship.csv | 36,869 |
| formula_one_roundentry.csv | 27,708 |
| formula_one_teamchampionship.csv | 14,579 |
| formula_one_pitstop.csv | 13,087 |
| formula_one_session.csv | 5,130 |
| formula_one_teamdriver.csv | 3,630 |
| formula_one_round.csv | 1,173 |
| formula_one_driver.csv | 881 |
| formula_one_team.csv | 214 |
| formula_one_circuit.csv | 78 |
| formula_one_season.csv | 77 |
| formula_one_pointsystem.csv | 24 |
| formula_one_championshipsystem.csv | 11 |
| formula_one_championshipadjustment.csv | 3 |
| formula_one_baseteam.csv | 0 |
| formula_one_penalty.csv | 0 |
