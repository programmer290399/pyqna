from pyqna.models.reading_comprehension.transformer_models import TransformerQnAModel


def test_model_inference():
    """
    Test the TransformerModel class inference.
    """
    model = TransformerQnAModel(
        {"model_name": "distilbert-base-uncased-distilled-squad", "pre_trained": True}
    )
    answers = model.get_answer(
        "New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean.\
                It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of\
                4.9 million. New Zealand's capital city is Wellington, and its most populous city is Auckland.",
        ["How many people live in New Zealand?", "What's the largest city?"],
    )
    assert answers[0]["answer"] == "4. 9 million"
    assert answers[1]["answer"] == "auckland"
