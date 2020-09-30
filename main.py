""" Main Python file to start training """

import os
from argparse import ArgumentParser
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger
from model import LightningModel
from data import JFRDataModule
from data.preprocess import Preprocess
from predict import PredictorCalci
import config as cfg


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                          INIT                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

def make_config():
    network   = cfg.Network()
    optimizer = cfg.Optimizer()
    scheduler = cfg.Scheduler()
    criterion = cfg.Criterion()
    return cfg.Model(network, optimizer, scheduler, criterion)

def init_data():
    return JFRDataModule(cfg.Dataloader())

def init_model(config):
    return  LightningModel(config)

def init_trainer():
    """ Init a Lightning Trainer using from_argparse_args
    Thus every CLI command (--gpus, distributed_backend, ...) become available.
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args   = parser.parse_args()
    lr_logger = LearningRateLogger()
    return Trainer.from_argparse_args(args, callbacks = [lr_logger])




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                          RUN                                        | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #


def run_preprocessing(fast=True):
    config = cfg.Preprocess()
    preprocessor = Preprocess(config.input_dir, config.output_dir, config.max_depth)
    if fast:
        preprocessor.fast_all_steps(config.cube_side, config.factor, config.margin, config.target_depth,
                                    config.augment_factor, config.augment_proba)
    else:
        preprocessor.preprocess_dataset(config.steps, config.cube_side, config.factor, config.margin, 
                                        config.target_depth,config.augment_factor, config.augment_proba)

def test_preprocessing():
    config = cfg.Preprocess()
    preprocessor = Preprocess(config.input_dir, config.output_dir, config.max_depth)
    preprocessor.test()


def augment():
    config = cfg.Preprocess()
    input_dir, output_dir = config.output_dir, os.path.join(config.output_dir, "augmented/")
    preprocessor = Preprocess(config.input_dir, config.output_dir, config.max_depth)
    preprocessor.augment(input_dir, output_dir, config.augment_factor, config.augment_proba)


def run_training():
    """ Instanciate a datamodule, a model and a trainer and run trainer.fit(model, data) """
    data   = init_data()
    config = make_config()
    model, trainer = init_model(config), init_trainer()
    trainer.fit(model, data)


def test(path):
    data    = init_data()
    model   = LightningModel.load_from_checkpoint(path)
    trainer = init_trainer()
    trainer.test()


def preprocess_eval():
    config = cfg.Preprocess()
    preprocessor = Preprocess(config.input_dir, config.output_dir, config.max_depth)
    preprocessor.preprocess_eval(config.cube_side, config.factor, config.margin, config.target_depth)


def predict(model_path, input_path, output_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    predictor = PredictorCalci(model_path, 0.5, device)
    result_df = predictor.predict_score(input_path)
    result_df.to_csv(output_path, index=False, sep=';')

if __name__ == '__main__':
    # run_preprocessing()
    # augment()
    # test_preprocessing()
    # run_training()
    # test('./lightning_logs/version_') 
    #preprocess_eval()
    predict("./lightning_logs/version_5/checkpoints/epoch=45.ckpt", "../eval/scans/", "../predictions.csv")

