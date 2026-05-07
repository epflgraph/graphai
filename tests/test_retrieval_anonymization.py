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


def test__retrieval_anonymization__load_models__uses_explicit_model_mapping(monkeypatch):
    captured = {}

    class FakeAnalyzerEngine:
        def __init__(self, nlp_engine, supported_languages):
            captured["nlp_engine"] = nlp_engine
            captured["supported_languages"] = supported_languages

    monkeypatch.setattr(anonymization, "NerModelConfiguration", FakeNerModelConfiguration)
    monkeypatch.setattr(anonymization, "TransformersNlpEngine", FakeTransformersNlpEngine)
    monkeypatch.setattr(anonymization, "AnalyzerEngine", FakeAnalyzerEngine)
    monkeypatch.setattr(anonymization, "AnonymizerEngine", FakeAnonymizerEngine)

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

    monkeypatch.setattr(anonymization, "NerModelConfiguration", FakeNerModelConfiguration)
    monkeypatch.setattr(anonymization, "TransformersNlpEngine", FakeTransformersNlpEngine)
    monkeypatch.setattr(anonymization, "AnalyzerEngine", FakeAnalyzerEngine)
    monkeypatch.setattr(anonymization, "AnonymizerEngine", FakeAnonymizerEngine)

    model = anonymization.AnonymizerModels()

    with pytest.raises(ValueError):
        model.load_models()

    assert model.models is None

    state["fail"] = False
    model.load_models()

    assert set(model.models) == {"analyzer", "anonymizer"}
