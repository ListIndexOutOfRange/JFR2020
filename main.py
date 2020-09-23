""" Main Python file to start training """

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger
from model import LightningModel
from datamodule import JFRDataModule
from preprocess import Preprocess
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
        preprocessor.fast_all_steps(config.cube_side, config.factor, config.margin, config.target_depth)
    else:
        preprocessor.preprocess_dataset(config.steps, config.cube_side, config.factor, config.margin)

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
    trainer.test(model, data)


if __name__ == '__main__':
    run_preprocessing()
    # run_training()
    # test('./lightning_logs/version_') 

