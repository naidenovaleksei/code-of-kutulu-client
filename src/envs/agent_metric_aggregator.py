import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

METRICS_SMOOTH_COEF = 0.05


class AgentMetricsAggregator:
    def __init__(self):
        self.reset_metrics()
        self.last_metrics = dict()

    def reset_metrics(self):
        self.metrics = defaultdict(list)

    def add_metrics(self, batch_metrics):
        for k, v in batch_metrics.items():
            self.metrics[k].append(v)

    def _save_smooth_mean_metric(self, metric_name):
        metric = np.mean(self.metrics[metric_name])
        last_metric = self.last_metrics.get(metric_name, metric)
        self.last_metrics[metric_name] = METRICS_SMOOTH_COEF * metric + (1 - METRICS_SMOOTH_COEF) * last_metric

    def get_metrics(self):
        return self.last_metrics
    
    def save_metrics(self, episode_idx=None):
        for metric_name in self.metrics.keys():
            self._save_smooth_mean_metric(metric_name)
        if logger.isEnabledFor(logging.INFO):
            lines = []
            if episode_idx is not None:
                lines.append(f"Episode {episode_idx}")
            for k, v in self.last_metrics.items():
                 lines.append(f"{k}: {v}")
            logger.info(", ".join(lines))
