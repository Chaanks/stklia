import os

from tqdm import tqdm
from loguru import logger
from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import dataset
from models import resnet_bottleneck, resnet_basic, NeuralNetAMSM
from test_resnet import score_utt_utt

class NNET(pl.LightningModule):
    def __init__(self, cfg):
        super(NNET, self).__init__()
        self.cfg = cfg
        self.generator = resnet_basic(cfg)
        self.classifier = NeuralNetAMSM(cfg.emb_size, cfg.num_classes)

    def forward(self, x):
        x = self.generator(x)
        x = self.classifier(x)
        return x

    def train_dataloader(self):
        assert self.cfg.train_data_path, "No training dataset given in train mode"
        self.ds_train = dataset.make_kaldi_ds(
            self.cfg.train_data_path, 
            seq_len=self.cfg.max_seq_len, 
            evaluation=False, 
            trials=None)
        logger.info(len(self.ds_train))
        return DataLoader(self.ds_train, batch_size=self.cfg.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        if self.cfg.eval_data_path and self.cfg.eval_trials_path:
            self.ds_val = dataset.make_kaldi_ds(self.cfg.eval_data_path, 
                seq_len=None, 
                evaluation=True, 
                trials=self.cfg.eval_trials_path)

            logger.info(len(self.ds_val))    
            return DataLoader(self.ds_val, batch_size=1, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD([{'params': self.generator.parameters(), 'lr': self.cfg.generator_lr},
                                     {'params': self.classifier.parameters(), 'lr': self.cfg.classifier_lr}],
                                    momentum=self.cfg.momentum)
        return [optimizer] #, [scheduler]

    def training_step(self, batch, _batch_idx):
        feats, spks, _ = batch
        feats = feats.unsqueeze(1)
        logits = self(feats)
        # logger.debug("node: {}, device {}".format(os.environ["SLURMD_NODENAME"], logits.device))
        loss = F.cross_entropy(logits, spks)
        return loss

    def training_step_end(self, losses):
        # batch_parts_outputs has outputs of each part of the batch
        loss = losses.mean()
        self.log('train_loss', loss.detach().cpu())
        return loss

    def validation_step(self, batch, _batch_idx):
        feats, spks, utts = batch
        # feats = feats.unsqueeze(0)
        embeds = self.generator(feats)
        logits = self.classifier(embeds)
        loss = F.cross_entropy(logits, spks)
        embeds = embeds

        print(loss.device)
        print(embeds.device)
        print(utts.device)

        return {'val_loss': loss, 'val_embeds': embeds, 'val_utts': utts}

    def validation_epoch_end(self, outputs):
        print(outputs[0]['val_loss'].shape)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(avg_loss.shape)
        embeds = [x['val_embeds'] for x in outputs]
        utts = [x['val_utts'] for x in outputs]

        #eer = scoring(embeds, utts, self.ds_val)
        #logs = {'val_avg_loss': avg_loss, 'eer': eer}
        #return {'val_loss': avg_loss, 'log': logs}