![img](docs/images/logo.png)
<hr>

![TESTS](https://img.shields.io/github/workflow/status/programmer290399/pyqna/Python%20application?style=for-the-badge) 
![pypi](https://img.shields.io/pypi/v/pyqna?style=for-the-badge) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pyqna?style=for-the-badge)](https://pepy.tech/project/pyqna)
![last-commit](https://img.shields.io/github/last-commit/programmer290399/pyqna?style=for-the-badge)
![GitHub issues](https://img.shields.io/github/issues-raw/programmer290399/pyqna?style=for-the-badge)
![license](https://img.shields.io/pypi/l/pyqna?style=for-the-badge)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyqna?style=for-the-badge)
<hr>

**PyQnA** is a simple package that aims to provide a consistent and unified API for all Question Answering related tasks in Python. 


## Installation 
* Currently we provide a PyPi package:
    ```bash
    $ pip install pyqna[all]
    ```
* In future we'd also have a Conda package.

## Example:

```python
# Import a specific model
from pyqna.models.reading_comprehension.transformer_models import TransformerQnAModel

# Instantiate the model
model = TransformerQnAModel(
    {"model_name": "distilbert-base-uncased-distilled-squad", "pre_trained": True}
)

# Take a context 
context = """ 
New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean.
It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of
4.9 million. New Zealand's capital city is Wellington, and its most populous city is Auckland.
"""

# Make a list of your queries
questions = ["How many people live in New Zealand?", "What's the largest city?"]

# Run inference using the instantiated models
answers = model.get_answer(context, questions)

# Print the output
print(answers)
```
**Output:**
```bash
[{'score': 0.9889324307441711, 'start': 55, 'end': 58, 'answer': '4. 9 million'}, {'score': 0.31774288415908813, 'start': 74, 'end': 74, 'answer': 'auckland'}]
```

## License

PyQnA is distributed under the [BSD 3-Clause License](https://github.com/programmer290399/pyqna/blob/main/LICENSE).

