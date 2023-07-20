import csv
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from yarr.agents.agent import ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary, AttentionSummary
from torch.utils.tensorboard import SummaryWriter
import wandb


class LogWriter(object):

    def __init__(self,
                 logdir: str,
                 tensorboard_logging: bool,
                 csv_logging: bool,
                 wandb_logging: bool = False,
                 configs: dict = {},
                 project_name: str = 'dev_project'
        ):
        self._tensorboard_logging = tensorboard_logging
        self._wandb_logging = wandb_logging
        self._csv_logging = csv_logging
        self._configs = configs
        os.makedirs(logdir, exist_ok=True)
        if wandb_logging:
            self._writer = wandb.init(
                project=project_name,
                config=configs,
            )
        elif tensorboard_logging:
            self._writer = SummaryWriter(logdir)

        if csv_logging:
            self._prev_row_data = self._row_data = OrderedDict()
            self._csv_file = os.path.join(logdir, 'data.csv')
            self._field_names = None

    def add_scalar(self, i, name, value):
        if self._wandb_logging:
            self._writer.log({name: value}, i)
        elif self._tensorboard_logging:
            self._writer.add_scalar(name, value, i)

        if self._csv_logging:
            if len(self._row_data) == 0:
                self._row_data['step'] = i
            self._row_data[name] = value.item() if isinstance(
                value, torch.Tensor) else value

    def _add_summaries_tb(self, i, summaries):
        for summary in summaries:
            try:
                if isinstance(summary, ScalarSummary):
                    self.add_scalar(i, summary.name, summary.value)
                elif self._tensorboard_logging:
                    if isinstance(summary, HistogramSummary):
                        self._writer.add_histogram(
                            summary.name, summary.value, i)
                    elif isinstance(summary, ImageSummary):
                        # Only grab first item in batch
                        v = (summary.value if summary.value.ndim == 3 else
                             summary.value[0])
                        self._writer.add_image(summary.name, v, i)
                    elif isinstance(summary, VideoSummary):
                        # Only grab first item in batch
                        v = (summary.value if summary.value.ndim == 5 else
                             np.array([summary.value]))
                        self._writer.add_video(
                            summary.name, v, i, fps=summary.fps)
            except Exception as e:
                logging.error('Error on summary: %s' % summary.name)
                raise e

    def _add_summaries_wb(self, i, summaries):
        data = {}
        for summary in summaries:
            try:
                if isinstance(summary, ScalarSummary):
                    data[summary.name] = summary.value
                elif isinstance(summary, HistogramSummary):
                    continue
                elif isinstance(summary, ImageSummary):
                    # Only grab first item in batch
                    v = (summary.value if summary.value.ndim == 3 else
                         summary.value[0])
                    data[summary.name] = wandb.Image(v)
                elif isinstance(summary, VideoSummary):
                    # Only grab first item in batch
                    v = (summary.value if summary.value.ndim == 5 else
                         np.array([summary.value]))
                    data[summary.name] = wandb.Video(v, fps=summary.fps)
                elif isinstance(summary, AttentionSummary):
                    assert summary.value.ndim == 3
                    dx, dy, dz = summary.value.shape
                    v = summary.value
                    v = (v - v.min()) / (v.max() - v.min()) * 13
                    v = np.floor(v).astype(np.float32) + 1

                    log_arr = np.array([
                        [x, y, z, v[x, y, z]]
                        for x in range(dx)
                        for y in range(dy)
                        for z in range(dz)
                    ])
                    data[summary.name] = wandb.Object3D(log_arr)

            except Exception as e:
                logging.error('Error on summary: %s' % summary.name)
                raise e
        self._writer.log(data, i)
    
    def add_summaries(self, i, summaries):
        if self._wandb_logging:
            self._add_summaries_wb(i, summaries)
        else:
            self._add_summaries_tb(i, summaries)
    
    def end_iteration(self):
        if self._csv_logging and len(self._row_data) > 0:
            with open(self._csv_file, mode='a+') as csv_f:
                names = self._field_names or self._row_data.keys()
                writer = csv.DictWriter(csv_f, fieldnames=names)
                if self._field_names is None:
                    writer.writeheader()
                else:
                    if not np.array_equal(self._field_names, self._row_data.keys()):
                        # Special case when we are logging faster than new
                        # summaries are coming in.
                        missing_keys = list(set(self._field_names) - set(
                            self._row_data.keys()))
                        for mk in missing_keys:
                            self._row_data[mk] = self._prev_row_data[mk]
                self._field_names = names
                writer.writerow(self._row_data)
            self._prev_row_data = self._row_data
            self._row_data = OrderedDict()

    def close(self):
        if self._wandb_logging:
            self._writer.finish()
        elif self._tensorboard_logging:
            self._writer.close()
