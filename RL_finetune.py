import torch
import pytorch_lightning as pl
from core.models.RL_finetune_loop import RLFinetuneLoop
import argparse
from core.config.config import Config
import os

from train_bfn import get_logger


def get_args():
    parser = argparse.ArgumentParser()
    # pretraining args
    parser.add_argument(
        "--pretrain_folder",
        type=str,
        default="checkpoints",
        help="path to pretrain model folder",
    )
    parser.add_argument(
        "--pre_config_file",
        type=str,
        default="checkpoints/config.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--pretrain_ckpt_path",
        type=str,
        default="checkpoints\last.ckpt",
        help="path to the checkpoint",
    )
    # parser.add_argument("--config_file", type=str, default="configs/default.yaml", help="path to config file")
    # -------------------------------------------------

    # finetuning args
    # meta
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/finetune_default.yaml",
    )
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--revision", type=str, default="default")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb_resume_id", type=str, default=None)
    parser.add_argument("--empty_folder", action="store_true")
    parser.add_argument("--test_only", action="store_true")

    # global config
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--logging_level", type=str, default="warning")

    # train data params
    parser.add_argument("--random_rot", action="store_true")
    parser.add_argument("--pos_noise_std", type=float, default=0)
    parser.add_argument("--pos_normalizer", type=float, default=2.0)

    # RL params
    parser.add_argument(
        "--reward",
        type=[],
        default=["vina_score"],
        choices=["vina_score", "SA", "QED"],
    )

    # train params
    # parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--epochs", type=int, default=15)
    # parser.add_argument("--v_loss_weight", type=float, default=1)
    # parser.add_argument("--lr", type=float, default=5e-4)
    # parser.add_argument(
    #     "--scheduler", type=str, default="plateau", choices=["cosine", "plateau"]
    # )
    # parser.add_argument("--weight_decay", type=float, default=0)
    # parser.add_argument("--max_grad_norm", type=str, default="Q")  # '8.0' for

    # bfn param
    # parser.add_argument("--sigma1_coord", type=float, default=0.03)
    # parser.add_argument("--beta1", type=float, default=1.5)
    # parser.add_argument("--t_min", type=float, default=0.0001)
    # parser.add_argument('--use_discrete_t', type=eval, default=True)
    # parser.add_argument('--discrete_steps', type=int, default=1000)
    # parser.add_argument('--destination_prediction', type=eval, default=True)
    # parser.add_argument('--sampling_strategy', type=str, default='end_back_pmf', choices=['vanilla', 'end_back_pmf']) #vanilla or end_back

    # eval param
    parser.add_argument(
        "--ckpt_path", type=str, default="best", help="path to the checkpoint"
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument(
        "--sample_num_atoms", type=str, default="ref", choices=["prior", "ref"]
    )
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument(
        "--docking_mode",
        type=str,
        default="vina_score",
        choices=["vina_score", "vina_dock"],
    )
    # -------------------------------------------------

    return parser.parse_args()


def get_pre_config(args):
    """
    Input: finetuning args
    Search config by --pre_config_file(yamlfile)
    Output: pretrain config
    """
    pre_config = Config(args.pre_config_file)
    return pre_config


def main():
    args = get_args()
    pre_config = get_pre_config(args)
    print("pretrain_config: ", pre_config)

    config = Config(**args.__dict__)
    print(config)

    return

    wandb_logger = get_logger(config)

    model = RLFinetuneLoop()
    trainer = pl.Trainer(
        default_root_dir=os.path.join(config.exp_name, config.revision),
        max_epochs=config.epochs,
        devices=1,
        logger=wandb_logger,
    )
    trainer.fit(
        model=model,
        train_dataloaders=None,
        val_dataloaders=None,
        ckpt_path=None,
    )
    pass


if __name__ == "__main__":
    main()
