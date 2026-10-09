"""A: correct then NER; B: NER, protect, correct, repeat NER."""


def process(text, corrector, recognizer, order="A", protect_entities=True):
    if order not in {"A", "B"}:
        raise ValueError("Порядок должен быть A или B")
    initial = recognizer.recognize(text) if order == "B" else []
    spans = [(e["start"], e["end"]) for e in initial] if order == "B" and protect_entities else []
    correction = corrector.correct(text, protected_spans=spans)
    final = recognizer.recognize(correction["proposed_text"])
    correction["config"].update(order=order, protect_entities=order == "B" and protect_entities,
                                 recognizer=recognizer.metadata())
    correction.update(initial_entities=initial, initial_entity_positions="original_text",
                      entities=final, entity_positions="proposed_text")
    return correction
