from pathlib import Path

import pandas as pd

from datacopilot.raw_data import load_crop, load_f1, load_ipl, load_olist, olist_table_name, parse_ipl_info

INFO = """version,2.4.0
info,team,Royal Challengers Bangalore
info,team,Kolkata Knight Riders
info,season,2007/08
info,date,2008/04/18
info,date,2008/04/19
info,match_number,1
info,venue,"M Chinnaswamy Stadium, Bangalore"
info,city,Bangalore
info,toss_winner,Royal Challengers Bangalore
info,toss_decision,field
info,player_of_match,BB McCullum
info,winner,Kolkata Knight Riders
info,winner_runs,140
info,player,Kolkata Knight Riders,SC Ganguly
info,registry,people,SC Ganguly,abc123
"""

DELIVERIES = "match_id,season,start_date,venue,innings,ball,batting_team,bowling_team\n" \
             "335982,2007/08,2008-04-18,\"M Chinnaswamy Stadium, Bangalore\",1,2.10,A,B\n"


def write_ipl(raw: Path) -> None:
    (raw / "ipl").mkdir(parents=True)
    (raw / "ipl" / "335982_info.csv").write_text(INFO)
    (raw / "ipl" / "all_matches.csv").write_text(DELIVERIES)


def test_olist_table_name():
    assert olist_table_name(Path("olist_order_items_dataset.csv")) == "order_items"
    assert olist_table_name(Path("product_category_name_translation.csv")) == "product_category_name_translation"


def test_parse_ipl_info(tmp_path):
    path = tmp_path / "335982_info.csv"
    path.write_text(INFO)
    rec = parse_ipl_info(path)
    assert rec["match_id"] == 335982
    assert (rec["team1"], rec["team2"]) == ("Royal Challengers Bangalore", "Kolkata Knight Riders")
    assert rec["date"] == "2008/04/18"  # first day kept
    assert rec["venue"] == "M Chinnaswamy Stadium, Bangalore"
    assert rec["winner_runs"] == "140"
    assert "player" not in rec and "registry" not in rec


def test_load_ipl_keeps_ball_as_text(tmp_path):
    write_ipl(tmp_path)
    tables = load_ipl(tmp_path)
    assert tables["deliveries"].ball.tolist() == ["2.10"]
    assert tables["matches"].match_id.tolist() == [335982]


def test_load_olist_f1_crop(tmp_path):
    (tmp_path / "olist").mkdir()
    pd.DataFrame({"order_id": [1]}).to_csv(tmp_path / "olist" / "olist_orders_dataset.csv", index=False)
    (tmp_path / "f1").mkdir()
    pd.DataFrame({"id": [1], "year": [2014]}).to_csv(tmp_path / "f1" / "formula_one_season.csv", index=False)
    (tmp_path / "crop_production").mkdir()
    pd.DataFrame({"Crop": ["Rice"]}).to_csv(tmp_path / "crop_production" / "crop_production.csv", index=False)
    assert list(load_olist(tmp_path)) == ["orders"]
    assert list(load_f1(tmp_path)) == ["season"]
    assert list(load_crop(tmp_path)) == ["crop_production"]
