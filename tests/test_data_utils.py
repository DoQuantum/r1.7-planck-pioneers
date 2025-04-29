from src.data.utils import load_wikitext

def test_load():
    ds = load_wikitext(split="validation")
    assert len(ds) > 0