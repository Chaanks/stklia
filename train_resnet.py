#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_resnet.py: This file contains function to train
the Resnet model. 
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import time
import numpy as np

from tqdm import tqdm
from loguru import logger
from pathlib import Path

import fairscale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataset
from seq_res_101 import resnet101
from models import resnet34, NeuralNetAMSM
from test_resnet import score_utt_utt

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

@logger.catch
def train(args, dataloader_train, device, dataset_validation=None):
    # Tensorflow logger
    writer = SummaryWriter(comment='_{}'.format(args.model_dir.name))
    num_classes = dataloader_train.dataset.num_classes

    # loguru
    logger.info("num_classes: " + str(num_classes))

    # Generator and classifier definition
    
    # generator = resnet34(args)
    generator = resnet101()
    generator = fairscale.nn.Pipe(generator, balance=[76, 76, 76, 75], chunks=8)

    classifier = NeuralNetAMSM(args.emb_size, num_classes)

    generator.train()
    classifier.train()

    #generator = generator.to(device)
    classifier = classifier.to(device)

    # Load the trained model if we continue from a checkpoint
    start_iteration = 0
    if args.checkpoint > 0:
        start_iteration = args.checkpoint
        for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
            model.load_state_dict(torch.load(args.checkpoints_dir / f'{modelstr}_{args.checkpoint}.pt'))
    
    elif args.checkpoint == -1:
        start_iteration = max([int(filename.stem[2:]) for filename in args.checkpoints_dir().iterdir()])
        for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
            model.load_state_dict(torch.load(args.checkpoints_dir / f'{modelstr}_{start_iteration}.pt'))

    # Optimizer definition
    optimizer = torch.optim.SGD([{'params': generator.parameters(), 'lr': args.generator_lr},
                                 {'params': classifier.parameters(), 'lr': args.classifier_lr}],
                                momentum=args.momentum)

    criterion = nn.CrossEntropyLoss()

    # multi GPU support :
    if args.multi_gpu:
        dpp_generator = nn.DataParallel(generator).to(device)
    
    if dataset_validation is not None:
        best_eer = {v.name:{'eer':1, 'ite':-1} for v in dataset_validation.trials} # best eer of all iterations

    start = time.process_time()
    for iterations in range(start_iteration, args.num_iterations + 1):
        # The current iteration is specified in the scheduler
        # Reduce the learning rate by the given factor (args.scheduler_lambda)
        if iterations in args.scheduler_steps:
            for params in optimizer.param_groups:
                params['lr'] *= args.scheduler_lambda
            print(optimizer)

        avg_loss = 0
        for feats, spk, utt in dataloader_train:
            feats = feats.unsqueeze(1).to(device)
            spk = torch.LongTensor(spk).to(device)

            # Creating embeddings
            if args.multi_gpu:
                embeds = dpp_generator(feats)
            else:
                embeds = generator(feats)

            # Classify embeddings
            preds = classifier(embeds, spk)

            # Calc the loss
            loss = criterion(preds, spk)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        
        avg_loss /= len(dataloader_train)
        # Write the loss in tensorflow
        writer.add_scalar('Loss', avg_loss, iterations)

        # loguru logging :
        if iterations % args.log_interval == 0:
            msg = "{}: {}: [{}/{}] \t C-Loss:{:.4f}, lr: {}, bs: {}".format(args.model_dir,
                                                                            time.ctime(),
                                                                            iterations,
                                                                            args.num_iterations,
                                                                            avg_loss,
                                                                            get_lr(optimizer),
                                                                            args.batch_size
                                                                            )
            logger.info(msg) 

         # Saving checkpoint
        if iterations % args.checkpoint_interval == 0:
            
            for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
                model.eval().cpu()
                cp_model_path = args.checkpoints_dir / f"{modelstr}_{iterations}.pt"
                torch.save(model.state_dict(), cp_model_path)
                model.to(device).train()

            # Testing the saved model
            if dataset_validation is not None:
                logger.info('Model Evaluation')
                test_res = score_utt_utt(generator, dataset_validation, device)
                for veri_pair, res in test_res.items():
                    eer = res['eer']
                    logger.info(f'EER on {veri_pair}: {eer}')
                    writer.add_scalar(f'{veri_pair}_EER', eer, iterations)
                    if eer < best_eer[veri_pair]["eer"]:
                        best_eer[veri_pair]["eer"] = eer
                        best_eer[veri_pair]["ite"] = iterations
                msg = ""
                for veri, vals in best_eer.items():
                    msg += f"\nBest score for {veri} is at iteration {vals['ite']} : {vals['eer']} eer"
                logger.success(msg)
            logger.info(f"Saved checkpoint at iteration {iterations}")

    # Final model saving
    for model, modelstr in [(generator, 'g'), (classifier, 'c')]:
        model.eval().cpu()
        cp_filename = "final_{}_{}.pt".format(modelstr, iterations)
        cp_model_path = args.model_dir / cp_filename
        torch.save(model.state_dict(), cp_model_path)
    logger.success(f'Training complete in {time.process_time()-start} seconds')
