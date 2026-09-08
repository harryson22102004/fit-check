from indic_st.languages import KOKBOROK, describe
from indic_st.metrics import bleu1, wer
from indic_st.registry import DATASETS


def test_kokborok_is_tonal_bodo_garo():
    assert KOKBOROK.tonal is True
    assert "Bodo" in KOKBOROK.nearest[0]
    assert describe("trp").iso639_3 == "trp"


def test_registry_has_the_three_tracks():
    assert "kokborok_asr" in DATASETS
    assert DATASETS["kokborok_asr"].hub_id.endswith("ne-asr-dataset-trp")
    assert DATASETS["bengali_indicvoices_st"].config == "indic2en"
    assert DATASETS["marathi_indicvoices_st"].language == "mr"


def test_metrics():
    assert wer("a b c", "a b c") == 0
    assert bleu1("the cat", "the cat") == 1
