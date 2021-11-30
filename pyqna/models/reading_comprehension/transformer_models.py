import re
import torch
from torch.nn.functional import softmax
from typing import Dict, List, Union
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig

from ..base import ReadingComprehensionModel
from ...utils.logging import create_logger, raise_if_not

logger = create_logger(__name__)


class TransformerQnAModel(ReadingComprehensionModel):
    """
    A unified API for all QuestionAnswering Models form HuggingFace Transformers.
    You can use any pre-trained model from HuggingFace Transformers.
    https://huggingface.co/models?pipeline_tag=question-answering
    """

    def __init__(self, config: Dict) -> None:
        """
        Initialize the model.

        Parameters
        ----------
        config : Dict
            Configuration of the model. See the config of model you're using for more details.
            Pretrained config is used by default and can be overridden by the config passed here.
        """
        super().__init__(config)

        # Initialize tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.config = AutoConfig.from_pretrained(config["model_name"])

        # Override config with external config
        self._override_config(config)

        # Initialize model
        self.device =  "cuda:0" if torch.cuda.is_available() else "cpu"

        if config.get("pre_trained", False):
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                config["model_name"]
            )
            self.is_trained = True
        else:
            self.model = AutoModelForQuestionAnswering.from_config(self.config)
        self.model.to(self.device)
    def _override_config(self, external_config: Dict) -> None:
        """
        Override the config of the model with the given config.
        """
        for key, value in external_config.items():
            if key in self.config.__dict__:
                self.config.__dict__[key] = value

    def _infer_from_model(self, context: str, question: str) -> Dict:
        """
        Infer the answer from the model. One question at a time.
        """

        ret_val = dict()
        inputs = self.tokenizer.encode_plus(
            question, context, add_special_tokens=True, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        non_answer_tokens = [x if x in [0, 1] else 0 for x in inputs.sequence_ids()]
        non_answer_tokens = torch.tensor(non_answer_tokens, dtype=torch.bool, device=self.device)

        potential_start = torch.where(
            non_answer_tokens,
            outputs.start_logits,
            torch.tensor(float("-inf"), dtype=torch.float,device=self.device),
        )
        potential_end = torch.where(
            non_answer_tokens,
            outputs.end_logits,
            torch.tensor(float("-inf"), dtype=torch.float,device=self.device),
        )

        potential_start = softmax(potential_start, dim=1)
        potential_end = softmax(potential_end, dim=1)
        answer_start = torch.argmax(potential_start)
        answer_end = torch.argmax(potential_end)
        answer = self.tokenizer.decode(
            inputs.input_ids.squeeze()[answer_start : answer_end + 1]
        )
        ret_val["score"] = (
            potential_start.squeeze()[answer_start]
            * potential_end.squeeze()[answer_end]
        ).item()
        ret_val["start"], ret_val["end"] = answer_start.item(), answer_end.item()
        ret_val["answer"] = answer

        return ret_val

    def train(self, dataset) -> None:
        self.model.train()
        self.is_trained = True
        raise NotImplementedError()

    def test(self, dataset) -> None:
        raise_if_not(self.is_trained, "Model is not trained yet.")
        self.model.eval()
        raise NotImplementedError()

    def get_answer(
        self, context: str, question: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        """
        Get the answer from the model.

        Parameters
        ----------
        context : str
            The context of the question.
        question : Union[str, List[str]]
            The question(s) to answer.

        Returns
        -------
        Union[str, List[str]]
            The answer(s) to the question(s).
        """
        raise_if_not(self.is_trained, "Cannot get answers from an untrained model.")

        self.model.eval()

        if isinstance(question, str):
            return self._infer_from_model(context, question)
        elif isinstance(question, list):
            return [self._infer_from_model(context, q) for q in question]
