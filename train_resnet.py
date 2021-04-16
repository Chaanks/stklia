#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_resnet.py: This file contains function to train
the Resnet model. 
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import time
import copy
import numpy as np

from tqdm import tqdm
from loguru import logger
from pathlib import Path

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataset
from models import resnet, NeuralNetAMSM, XTDNN, LightCNN
from test_resnet import score_utt_utt
from lottery_ticket import weight_init, make_mask, prune_by_percentile, original_initialization, print_nonzeros

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
    if args.model == 'RESNET':
        generator = resnet(args)
    elif args.model == 'XTDNN':
        generator = XTDNN()
    elif args.model == 'CNN':
        generator = LightCNN()

    # Lottery ticket

    # Weight Initialization
    generator.apply(weight_init)

    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(generator.state_dict())

    save_dir = args.model_dir / 'saves'
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator, save_dir / f"initial_state_dict_{args.prune_type}.pth.tar")
    
    # Making Initial Mask
    mask, step = make_mask(generator)

    # Load the trained model if we continue from a checkpoint
    start_iteration = 0
    if args.checkpoint > 0:
        start_iteration = args.checkpoint
        for model, modelstr in [(generator, 'g')]:
            model.load_state_dict(torch.load(args.checkpoints_dir / f'{modelstr}_{args.checkpoint}.pt'))
    
    elif args.checkpoint == -1:
        start_iteration = max([int(filename.stem[2:]) for filename in args.checkpoints_dir().iterdir()])
        for model, modelstr in [(generator, 'g')]:
            model.load_state_dict(torch.load(args.checkpoints_dir / f'{modelstr}_{start_iteration}.pt'))

    # Optimizer definition
    optimizer = torch.optim.SGD([{'params': generator.parameters(), 'lr': args.generator_lr}])

    criterion = nn.MSELoss() # nn.CosineEmbeddingLoss() #nn.CrossEntropyLoss()
    
    if dataset_validation is not None:
        best_eer = {v.name:{'eer':100, 'ite':-1} for v in dataset_validation.trials} # best eer of all iterations

    start = time.process_time()
    pruned = False
    comp1 = print_nonzeros(generator)

    for iterations in range(start_iteration, args.num_iterations + 1):

        # Prune
        if iterations == 0:
            pruned = True
            original_initialization(generator, mask, initial_state_dict)
        elif iterations % args.prune_iterations == 0:
            pruned = True
            prune_by_percentile(generator, step, mask, args.prune_percent, resample=False, reinit=False)
            for name, param in generator.named_parameters():
                if 'weight' in name:
                    weight_dev = param.device
                    param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_dev)
                    step = step + 1
            step = 0
        
        if pruned:
            pruned = False
            print(f"\n--- Pruning Level [{iterations}/{args.prune_iterations}]: ---")
            # Print the table of Nonzeros in each layer
            comp1 = print_nonzeros(generator)
            generator = generator.to(device)

            # Optimizer definition
            optimizer = torch.optim.SGD([{'params': generator.parameters(), 'lr': args.generator_lr}])

            generator.train()

            # multi GPU support :
            if args.multi_gpu:
                dpp_generator = nn.DataParallel(generator).to(device)   
        
        # The current iteration is specified in the scheduler
        # Reduce the learning rate by the given factor (args.scheduler_lambda)
        if iterations in args.scheduler_steps:
            for params in optimizer.param_groups:
                params['lr'] *= args.scheduler_lambda
            print(optimizer)

        avg_loss = 0
        for feats, targets, _ in dataloader_train:
            feats = feats.unsqueeze(1).to(device)
            targets = targets.to(device)

            # Creating embeddings
            if args.multi_gpu:
                embeds = dpp_generator(feats)
            else:
                embeds = generator(feats)

            # Calc the loss
            y = torch.Tensor([1.0]).to(device)
            loss = criterion(embeds, targets) # y

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Freezing Pruned weights by making their gradients Zero
            for name, p in generator.named_parameters():
                if 'weight' in name:
                    tensor = p.data.cpu().numpy()
                    grad_tensor = p.grad.data.cpu().numpy()
                    grad_tensor = np.where(tensor < 1e-6, 0, grad_tensor)
                    p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            avg_loss += loss.item()
        
        avg_loss /= len(dataloader_train)
        # Write the loss in tensorflow
        writer.add_scalar('Loss', avg_loss, iterations)
        writer.add_scalar('lr', get_lr(optimizer), iterations)


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
            
            for model, modelstr in [(generator, 'g')]:
                model.eval().cpu()
                cp_model_path = args.checkpoints_dir / f"{modelstr}_{iterations}.pt"
                torch.save(model.state_dict(), cp_model_path)
                model.to(device).train()

            # Dumping mask
            with open(save_dir / f"{iterations}_{args.prune_type}_mask_{comp1}.pkl", 'wb') as fp:
                pickle.dump(mask, fp)

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


    # # Final model saving
    # for model, modelstr in [(generator, 'g')]:
    #     model.eval().cpu()
    #     cp_filename = "final_{}_{}.pt".format(modelstr, iterations)
    #     cp_model_path = args.model_dir / cp_filename
    #     torch.save(model.state_dict(), cp_model_path)
    # logger.success(f'Training complete in {time.process_time()-start} seconds')
