"""Build a tiny fake data/raw/ tree with the same layout and known quality issues."""
from pathlib import Path

import pandas as pd


def _csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_olist(raw: Path) -> None:
    o = raw / "olist"
    _csv(o / "olist_customers_dataset.csv", pd.DataFrame({
        "customer_id": ["c1", "c2"], "customer_unique_id": ["u1", "u2"],
        "customer_zip_code_prefix": [1000, 2000], "customer_city": ["São Paulo", "rio de janeiro"],
        "customer_state": ["sp", "RJ"]}))
    _csv(o / "olist_geolocation_dataset.csv", pd.DataFrame({
        "geolocation_zip_code_prefix": [1000, 1000, 1000, 2000, 3000],
        "geolocation_lat": [-23.5, -23.5, -23.5, -22.9, 40.0],      # last one: not in Brazil
        "geolocation_lng": [-46.6, -46.6, -46.6, -43.2, -3.0],
        "geolocation_city": ["sao paulo", "sao paulo", "sa£o paulo", "rio de janeiro", "madrid"],
        "geolocation_state": ["SP", "SP", "SP", "RJ", "XX"]}))
    _csv(o / "olist_orders_dataset.csv", pd.DataFrame({
        "order_id": ["o1", "o2", "o3"], "customer_id": ["c1", "c2", "c1"],
        "order_status": ["delivered", "delivered", "canceled"],
        "order_purchase_timestamp": ["2017-01-01 10:00:00", "2017-01-02 10:00:00", "2017-01-03 10:00:00"],
        "order_approved_at": ["2017-01-01 11:00:00", "2017-01-02 11:00:00", None],
        "order_delivered_carrier_date": ["2017-01-01 10:30:00", "2017-01-03 10:00:00", None],
        "order_delivered_customer_date": ["2017-01-05 10:00:00", None, None],
        "order_estimated_delivery_date": ["2017-01-10 00:00:00"] * 3}))
    _csv(o / "olist_order_items_dataset.csv", pd.DataFrame({
        "order_id": ["o1", "o2"], "order_item_id": [1, 1], "product_id": ["p1", "p2"],
        "seller_id": ["s1", "s1"], "shipping_limit_date": ["2017-01-02 00:00:00"] * 2,
        "price": [10.0, 0.0], "freight_value": [1.0, 2.0]}))
    _csv(o / "olist_order_payments_dataset.csv", pd.DataFrame({
        "order_id": ["o1", "o2"], "payment_sequential": [1, 1], "payment_type": ["credit_card", "not_defined"],
        "payment_installments": [1, 0], "payment_value": [11.0, 0.0]}))
    _csv(o / "olist_order_reviews_dataset.csv", pd.DataFrame({
        "review_id": ["r1", "r1"], "order_id": ["o1", "o2"], "review_score": [5, 4],
        "review_comment_title": [None, None], "review_comment_message": ["great\nproduct", None],
        "review_creation_date": ["2017-01-06 00:00:00"] * 2, "review_answer_timestamp": ["2017-01-07 00:00:00"] * 2}))
    _csv(o / "olist_products_dataset.csv", pd.DataFrame({
        "product_id": ["p1", "p2"], "product_category_name": ["beleza_saude", "pc_gamer"],
        "product_name_lenght": [10, 20], "product_description_lenght": [100, 200], "product_photos_qty": [1, 2],
        "product_weight_g": [100, 0], "product_length_cm": [10, 10], "product_height_cm": [5, 5],
        "product_width_cm": [5, 5]}))
    _csv(o / "olist_sellers_dataset.csv", pd.DataFrame({
        "seller_id": ["s1"], "seller_zip_code_prefix": [1000], "seller_city": ["sao paulo"], "seller_state": ["SP"]}))
    _csv(o / "product_category_name_translation.csv", pd.DataFrame({
        "product_category_name": ["beleza_saude"], "product_category_name_english": ["health_beauty"]}))


IPL_INFO = """version,2.4.0
info,team,{t1}
info,team,{t2}
info,season,{season}
info,date,{date}
info,venue,{venue}
info,city,X
info,toss_winner,{t1}
info,toss_decision,field
info,winner,{t2}
info,winner_runs,10
"""


def build_ipl(raw: Path) -> None:
    folder = raw / "ipl"
    rows = []
    matches = [
        (1, "2007/08", "2008-04-18", "M Chinnaswamy Stadium, Bangalore", "Royal Challengers Bangalore", "Kings XI Punjab"),
        (2, "2016", "2016-04-10", "M.Chinnaswamy Stadium", "Rising Pune Supergiants", "Delhi Daredevils"),
        (3, "2017", "2017-04-10", "M Chinnaswamy Stadium", "Rising Pune Supergiant", "Mumbai Indians"),
    ]
    for mid, season, date, venue, t1, t2 in matches:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{mid}_info.csv").write_text(IPL_INFO.format(
            t1=t1, t2=t2, season=season, date=date.replace("-", "/"), venue=f'"{venue}"'))
        for ball in ("0.1", "0.2", "2.1", "2.10"):
            rows.append({"match_id": mid, "season": season, "start_date": date, "venue": venue, "innings": 1,
                         "ball": ball, "batting_team": t1, "bowling_team": t2, "striker": "A ", "runs_off_bat": 1,
                         "extras": 0, "wides": None, "noballs": None, "byes": None, "legbyes": None, "penalty": None})
    _csv(folder / "all_matches.csv", pd.DataFrame(rows))


def build_f1(raw: Path) -> None:
    f = raw / "f1"
    _csv(f / "formula_one_season.csv", pd.DataFrame({"id": [1, 2], "year": [2013, 2014]}))
    _csv(f / "formula_one_circuit.csv", pd.DataFrame({
        "id": [1], "name": ["Albert Park"], "locality": ["Melbourne"], "country": ["Australia"]}))
    _csv(f / "formula_one_round.csv", pd.DataFrame({
        "id": [10, 20], "circuit_id": [1, 1], "date": ["2013-03-17", "2014-03-16"], "is_cancelled": ["f", "f"],
        "name": ["Australian GP", "Australian GP"], "number": [1, 1], "season_id": [1, 2]}))
    _csv(f / "formula_one_driver.csv", pd.DataFrame({
        "id": [1], "forename": ["Nico"], "surname": ["Rosberg"], "abbreviation": ["ROS"], "nationality": ["German"]}))
    _csv(f / "formula_one_team.csv", pd.DataFrame({"id": [1], "name": ["Mercedes"]}))
    _csv(f / "formula_one_teamdriver.csv", pd.DataFrame({"id": [1], "driver_id": [1], "team_id": [1], "season_id": [2]}))
    _csv(f / "formula_one_roundentry.csv", pd.DataFrame({
        "id": [100, 200], "car_number": [9, 6], "round_id": [10, 20], "team_driver_id": [1, 1]}))
    _csv(f / "formula_one_session.csv", pd.DataFrame({
        "id": [1000, 2000], "round_id": [10, 20], "type": ["R", "R"], "number": [3, 3], "is_cancelled": ["f", "f"]}))
    _csv(f / "formula_one_sessionentry.csv", pd.DataFrame({
        "id": [1, 2], "session_id": [1000, 2000], "round_entry_id": [100, 200], "grid": [3, 3], "position": [2, 1],
        "points": [18, 25], "laps_completed": [58, 57], "is_classified": ["t", "t"], "status": [0, 0],
        "detail": ["Finished", "Finished"], "time": ["01:30:00.5", "01:32:58.71"], "fastest_lap_rank": [1, 1]}))
    _csv(f / "formula_one_lap.csv", pd.DataFrame({
        "id": [1, 2, 3], "session_entry_id": [1, 2, 2], "number": [1, 1, 2], "position": [2, 1, 1],
        "time": ["00:01:35.1", "00:01:34.2", "00:01:33.9"], "average_speed": [None] * 3,
        "is_deleted": ["f", "f", "f"], "is_entry_fastest_lap": ["f", "f", "t"]}))
    _csv(f / "formula_one_pitstop.csv", pd.DataFrame({
        "id": [1], "session_entry_id": [2], "lap_id": [3], "number": [1], "duration": ["00:00:22.5"],
        "local_timestamp": ["17:00:00"]}))
    standing = {"adjustment_type": [0, 0], "highest_finish": [1, 1], "is_eligible": ["t", "t"],
                "points": [0, 25], "position": [1, 1], "round_number": [0, 1], "session_number": [3, 3],
                "year": [2014, 2014]}
    _csv(f / "formula_one_driverchampionship.csv", pd.DataFrame({**standing, "driver_id": [1, 1], "win_count": [0, 1]}))
    _csv(f / "formula_one_teamchampionship.csv", pd.DataFrame({**standing, "team_id": [1, 1], "win_count": [0, 1]}))
    _csv(f / "formula_one_penalty.csv", pd.DataFrame({"id": pd.Series(dtype=int)}))


def build_crop(raw: Path) -> None:
    _csv(raw / "crop_production" / "crop_production.csv", pd.DataFrame({
        "State_Name": ["Telangana ", "Telangana", "Kerala", "Kerala"],
        "District_Name": ["ADILABAD", "ADILABAD", "KOLLAM", "KOLLAM"],
        "Crop_Year": [2014, 2014, 2014, 2015],
        "Season": ["Kharif     ", "Kharif", "Whole Year ", "Whole Year "],
        "Crop": ["Rice", "Rice", "Coconut ", "Coconut "],
        "Area": [100.0, 100.0, 50.0, 60.0],
        "Production": [200.0, 200.0, None, 0.0]}))


def build_all(raw: Path) -> Path:
    build_olist(raw)
    build_ipl(raw)
    build_f1(raw)
    build_crop(raw)
    return raw
