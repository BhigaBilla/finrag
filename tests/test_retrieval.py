from src.chunk import split_words
from src.retrieve import rrf, tokenize


def test_split_covers_all_words_with_overlap():
    words = [str(i) for i in range(600)]
    chunks = split_words(words, size=250, overlap=50)
    assert chunks[0][0] == "0" and chunks[-1][-1] == "599"
    assert chunks[1][0] == "200"  # step = size - overlap
    assert all(len(c) <= 250 for c in chunks)


def test_short_page_is_one_chunk():
    assert split_words(["a", "b"], size=250, overlap=50) == [["a", "b"]]


def test_rrf_rewards_agreement():
    # 'b' is 2nd in both lists, 'a' is 1st in one and absent from the other.
    assert rrf([["a", "b", "c"], ["d", "b", "e"]])[0] == "b"


def test_tokenize_keeps_numbers_and_decimals():
    assert tokenize("Revenue of ₹2,40,893 crore grew 6.0% in FY2025") == \
        ["revenue", "2", "40", "893", "crore", "grew", "6.0", "fy2025"]


def test_ranked_pages_dedupes_and_scores():
    from src.evaluate import ranked_pages, score
    chunks = ["TCS|FY2026|5|0", "TCS|FY2026|5|1", "TCS|FY2026|9|0", "TCS|FY2026|56|2"]
    pages = ranked_pages(chunks)
    assert pages == [("TCS", "FY2026", 5), ("TCS", "FY2026", 9), ("TCS", "FY2026", 56)]
    hits, rr = score(pages, {("TCS", "FY2026", 56)})
    assert rr == 1 / 3 and hits[1] == 0 and hits[3] == 1
    hits, rr = score(pages, {("TCS", "FY2026", 99)})
    assert rr == 0 and hits[10] == 0
