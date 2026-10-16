import fitz

from src import parse


def test_page_numbers_and_metadata(tmp_path, monkeypatch):
    raw = tmp_path / "raw" / "TCS"
    raw.mkdir(parents=True)
    doc = fitz.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Revenue grew {i} percent in the year under review " * 2)
    doc.new_page()  # blank page, should be skipped
    doc.save(raw / "FY2026.pdf")

    out = tmp_path / "pages.jsonl"
    monkeypatch.setattr(parse, "RAW", tmp_path / "raw")
    monkeypatch.setattr(parse, "PAGES", out)
    parse.main()

    import json
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["page"] for r in recs] == [1, 2, 3]
    assert recs[0]["company"] == "TCS" and recs[0]["fy"] == "FY2026"
    assert "Revenue grew 2 percent" in recs[2]["text"]
