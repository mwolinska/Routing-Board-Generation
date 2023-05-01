import collections
import logging
from typing import Optional, Dict, Any

from jumanji.training.loggers import Logger


class PickleLogger(Logger):
    """Logs information in format accepted by BenchmarkOnRandomAgent."""

    def __init__(
        self, name: Optional[str] = None, save_checkpoint: bool = False
    ) -> None:
        super().__init__(save_checkpoint=save_checkpoint)
        if name:
            logging.info(f"Experiment: {name}.")
        self.file_name = "martas_test_file.json"
        self.training_log = {}
        self.eval_log_greedy = {}
        self.eval_log_stochastic = {}
        self.episode_counter = 0

    def _format_values(self, data: Dict[str, Any]) -> Dict[float, float]:
        return {key: float(value) for key,value in sorted(data.items())}

    def return_eval_dict(self):
        return {key: (value / self.episode_counter) for key, value in self.eval_log_greedy.items()}

    def write(
        self,
        data: Dict[str, Any],
        label: Optional[str] = None,
        env_steps: Optional[int] = None,
    ) -> None:

        self.episode_counter += 1

        data = self._format_values(data)
        if label == "train":
            self.training_log = self._update_dictionary(self.training_log, data)
        elif label == "eval_stochastic":
            self.eval_log_stochastic = self._update_dictionary(self.eval_log_stochastic, data)
        elif label == "eval_greedy":
            self.eval_log_greedy = self._update_dictionary(self.eval_log_greedy, data)

    def _update_dictionary(self, dictionary_to_update, new_dictionary):
        temp_dict = collections.defaultdict(list)
        if not dictionary_to_update:
            # temp_dict = new_dictionary
            for key, value in new_dictionary.items():
                temp_dict[key] = [value]
        else:
            temp_dict = collections.defaultdict(list)
            for key, value in dictionary_to_update.items():
                temp_dict[key] = value + [new_dictionary[key]]

        return temp_dict
