import pytest

from graphai.core.retrieval import anonymization


class FakeNerModelConfiguration:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTransformersNlpEngine:
    def __init__(self, models, ner_model_configuration):
        self.models = models
        self.ner_model_configuration = ner_model_configuration


class FakeAnonymizerEngine:
    pass


class FakeRegistry:
    def __init__(self):
        self.recognizers = []

    def add_recognizer(self, recognizer):
        self.recognizers.append(recognizer)


class FakeGLiNERRecognizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test__retrieval_anonymization__load_models__uses_explicit_model_mapping(monkeypatch):
    captured = {}

    class FakeAnalyzerEngine:
        def __init__(self, nlp_engine, supported_languages):
            captured["nlp_engine"] = nlp_engine
            captured["supported_languages"] = supported_languages
            self.registry = FakeRegistry()

    monkeypatch.setattr(anonymization, "NerModelConfiguration", FakeNerModelConfiguration)
    monkeypatch.setattr(anonymization, "TransformersNlpEngine", FakeTransformersNlpEngine)
    monkeypatch.setattr(anonymization, "AnalyzerEngine", FakeAnalyzerEngine)
    monkeypatch.setattr(anonymization, "AnonymizerEngine", FakeAnonymizerEngine)
    monkeypatch.setattr(anonymization, "_is_gliner_available", lambda: False)

    model = anonymization.AnonymizerModels()
    model.load_models()

    ner_configuration = captured["nlp_engine"].ner_model_configuration
    mapping = ner_configuration.kwargs["model_to_presidio_entity_mapping"]

    assert mapping["B-PER"] == "PERSON"
    assert mapping["I-PER"] == "PERSON"
    assert mapping["B-DATE"] == "DATE_TIME"
    assert mapping["I-DATE"] == "DATE_TIME"
    assert captured["supported_languages"] == ["en", "fr"]
    assert set(model.models) == {"analyzer", "anonymizer"}


def test__retrieval_anonymization__load_models__failed_load_can_retry(monkeypatch):
    state = {"fail": True}

    class FakeAnalyzerEngine:
        def __init__(self, nlp_engine, supported_languages):
            if state["fail"]:
                raise ValueError("model_to_presidio_entity_mapping is missing from model configuration")
            self.registry = FakeRegistry()

    monkeypatch.setattr(anonymization, "NerModelConfiguration", FakeNerModelConfiguration)
    monkeypatch.setattr(anonymization, "TransformersNlpEngine", FakeTransformersNlpEngine)
    monkeypatch.setattr(anonymization, "AnalyzerEngine", FakeAnalyzerEngine)
    monkeypatch.setattr(anonymization, "AnonymizerEngine", FakeAnonymizerEngine)
    monkeypatch.setattr(anonymization, "_is_gliner_available", lambda: False)

    model = anonymization.AnonymizerModels()

    with pytest.raises(ValueError):
        model.load_models()

    assert model.models is None

    state["fail"] = False
    model.load_models()

    assert set(model.models) == {"analyzer", "anonymizer"}


def test__retrieval_anonymization__build_gliner_recognizers__uses_general_pii_mapping(monkeypatch):
    monkeypatch.setitem(
        anonymization.config,
        "anonymization",
        {
            "gliner_enabled": "true",
            "gliner_model_name": "test/gliner-pii",
            "gliner_threshold": "0.42",
            "gliner_flat_ner": "false",
            "gliner_multi_label": "true",
        },
    )
    monkeypatch.setattr(anonymization, "_is_gliner_available", lambda: True)
    monkeypatch.setattr(anonymization, "GLiNERRecognizer", FakeGLiNERRecognizer)

    recognizers = anonymization.build_gliner_recognizers(device="cpu", cache_dir="/tmp/huggingface")

    assert len(recognizers) == 2
    recognizer_kwargs = recognizers[0].kwargs
    assert recognizer_kwargs["model_name"] == "test/gliner-pii"
    assert recognizer_kwargs["supported_language"] == "en"
    assert recognizer_kwargs["threshold"] == 0.42
    assert recognizer_kwargs["flat_ner"] is False
    assert recognizer_kwargs["multi_label"] is True
    assert recognizer_kwargs["map_location"] == "cpu"
    assert recognizer_kwargs["cache_dir"] == "/tmp/huggingface"
    assert recognizer_kwargs["entity_mapping"]["address"] == "STREET_ADDRESS"
    assert recognizer_kwargs["entity_mapping"]["username"] == "USERNAME"
    assert recognizer_kwargs["entity_mapping"]["bank account number"] == "BANK_ACCOUNT_NUMBER"
    assert recognizer_kwargs["entity_mapping"]["reservation number"] == "RESERVATION_NUMBER"
    assert recognizer_kwargs["entity_mapping"]["cvv"] == "CVV"


def test__retrieval_anonymization__load_models__registers_gliner_when_available(monkeypatch):
    class FakeAnalyzerEngine:
        def __init__(self, nlp_engine, supported_languages):
            self.registry = FakeRegistry()

    monkeypatch.setitem(anonymization.config, "anonymization", {"gliner_enabled": "true"})
    monkeypatch.setattr(anonymization, "NerModelConfiguration", FakeNerModelConfiguration)
    monkeypatch.setattr(anonymization, "TransformersNlpEngine", FakeTransformersNlpEngine)
    monkeypatch.setattr(anonymization, "AnalyzerEngine", FakeAnalyzerEngine)
    monkeypatch.setattr(anonymization, "AnonymizerEngine", FakeAnonymizerEngine)
    monkeypatch.setattr(anonymization, "_is_gliner_available", lambda: True)
    monkeypatch.setattr(anonymization, "GLiNERRecognizer", FakeGLiNERRecognizer)

    model = anonymization.AnonymizerModels()
    model.load_models()

    recognizers = model.models["analyzer"].registry.recognizers
    assert len(recognizers) == 2
    assert {recognizer.kwargs["supported_language"] for recognizer in recognizers} == {"en", "fr"}
