import torch
import pytorch_lightning as pl
from core.models.RL_finetune_loop import RLFinetuneLoop
import argparse
from core.config.config import Config
import os

from torch_geometric.transforms import Compose
from pytorch_lightning import seed_everything
from torch_geometric.loader import DataLoader

from train_bfn import get_logger
import core.utils.transforms as trans
from core.datasets import get_dataset
from core.datasets.pl_data import FOLLOW_BATCH
from core.callbacks.basic import (
    RecoverCallback,
    GradientClip,
    NormalizerCallback,
    EMACallback,
)
from absl import logging


def get_dataloader(cfg):
    if cfg.data.name == "pl_tr":
        dataset, subsets = get_dataset(config=cfg.data)
        train_set, test_set = subsets["train"], subsets["test"]
        cfg.dynamics.protein_atom_feature_dim = dataset.protein_atom_feature_dim
        cfg.dynamics.ligand_atom_feature_dim = dataset.ligand_atom_feature_dim
    else:
        protein_featurizer = trans.FeaturizeProteinAtom()
        ligand_featurizer = trans.FeaturizeLigandAtom(
            cfg.data.transform.ligand_atom_mode
        )
        transform_list = [
            protein_featurizer,
            ligand_featurizer,
            # trans.FeaturizeLigandBond(),
        ]

        transform = Compose(transform_list)
        cfg.dynamics.protein_atom_feature_dim = protein_featurizer.feature_dim
        cfg.dynamics.ligand_atom_feature_dim = ligand_featurizer.feature_dim
        dataset, subsets = get_dataset(config=cfg.data, transform=transform)
        train_set, test_set = subsets["train"], subsets["test"]
    if "val" in subsets and len(subsets["val"]) > 0:
        val_set = subsets["val"]
    else:
        val_set = test_set

    print(
        f"protein feature dim: {cfg.dynamics.protein_atom_feature_dim}, "
        + f"ligand feature dim: {cfg.dynamics.ligand_atom_feature_dim}"
    )

    collate_exclude_keys = ["ligand_nbh_list"]
    # size-1 debug set
    if cfg.debug:
        debug_set = torch.utils.data.Subset(val_set, [0] * 800)
        debug_set_val = torch.utils.data.Subset(val_set, [0] * 10)
        cfg.train.val_freq = 100
        # get debug set val data batch
        debug_batch_val = next(
            iter(
                DataLoader(
                    debug_set_val, batch_size=cfg.train.batch_size, shuffle=False
                )
            )
        )
        print(f"debug batch val: {debug_batch_val.ligand_filename}")
        train_loader = DataLoader(
            debug_set,
            batch_size=1,
            shuffle=False,  # set shuffle to False
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
        )
        val_loader = DataLoader(
            debug_set_val,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
        )
        test_loader = DataLoader(
            debug_set_val,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
        )
    else:
        logging.info(f"Training: {len(train_set)} Validation: {len(val_set)}")
        train_loader = DataLoader(
            train_set,
            batch_size=1,
            shuffle=True,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.evaluation.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
        )
    cfg.train.scheduler.max_iters = cfg.train.epochs * len(train_loader)

    return train_loader, val_loader, test_loader


def get_args():
    parser = argparse.ArgumentParser()
    # pretraining args
    parser.add_argument(
        "--pretrain_folder",
        type=str,
        default=".//checkpoints",
        help="path to pretrain model folder",
    )
    parser.add_argument(
        "--pre_config_file",
        type=str,
        default=".//checkpoints/config.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--pretrain_ckpt_path",
        type=str,
        default=".//checkpoints/last.ckpt",
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
        type=str,
        default="[vina_score, sa, qed]",
        choices=["vina_score", "sa", "qed"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--mol_per_protein",
        type=int,
        default=1,
        help="number of molecules per protein during RL finetuning",
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
        "--sample_num_atoms", type=str, default="prior", choices=["prior", "ref"]
    )
    parser.add_argument("--visual_chain", action="store_true")
    parser.add_argument(
        "--docking_mode",
        type=str,
        default="vina_score",
        choices=["vina_score", "vina_dock"],
    )
    # -------------------------------------------------
    # post process args
    args = parser.parse_args()
    args.reward = (
        args.reward.replace("[", "").replace("]", "").replace(" ", "").split(",")
    )
    return args


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

    seed_everything(config.seed, workers=True)

    if not args.no_wandb:
        wandb_logger = get_logger(config)
    else:
        wandb_logger = None

    model = RLFinetuneLoop(config=config, pre_config=pre_config)

    # get dataloader
    train_loader, val_loader, test_loader = get_dataloader(pre_config)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(config.exp_name, config.revision),
        max_epochs=config.RL.epochs,
        devices=1,
        logger=wandb_logger,
        callbacks=[NormalizerCallback(normalizer_dict=pre_config.data.normalizer_dict)],
    )
    trainer.fit(
        model=model,
        train_dataloaders=test_loader,
        ckpt_path=None,
    )
    pass


if __name__ == "__main__":
    main()
