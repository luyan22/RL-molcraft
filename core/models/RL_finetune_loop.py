import pytorch_lightning as pl
from core.config.config import Config
from core.models.bfn4sbdd import BFN4SBDDScoreModel
import torch
import numpy as np
from core.models.sbdd_train_loop import center_pos
import core.evaluation.utils.atom_num as atom_num
import copy, os, json, time
from torch_scatter import scatter_mean, scatter_sum
from core.callbacks.validation_callback import reconstruct_mol_and_filter_invalid

import core.utils.transforms as trans
import core.evaluation.utils.atom_num as atom_num
from core.evaluation.metrics import CondMolGenMetric


class RLFinetuneLoop(pl.LightningModule):
    def __init__(self, config: Config, pre_config: Config):
        super().__init__()
        self.cfg = config
        self.pre_config = pre_config
        self.ref_model = BFN4SBDDScoreModel(**pre_config.dynamics.todict())
        ckpt_path = config.pretrain.pretrain_ckpt_path
        ckpt = torch.load(ckpt_path)
        # check the state_dict keys
        ref_model_keys = set(self.ref_model.state_dict().keys())
        for k in ckpt["state_dict"].keys():
            if k in ref_model_keys:
                ref_model_keys.remove(k)
            else:
                print(f"key {k} not in ref_model.state_dict().keys()")
        print(f"ref_model_keys not in ckpt: {ref_model_keys}")

        self.ref_model.load_state_dict(ckpt["state_dict"], strict=False)

        # deep copy ref model
        self.model = copy.deepcopy(self.ref_model)
        self.model.train()
        self.ref_model.eval()

        # setup the metric
        self.metric = CondMolGenMetric(config.evaluation.metric_name, use_RL=True)
        pass

    def assert_parameter_freeze_status(self):
        # model is unfreeze, ref_model is freeze
        assert (
            not self.ref_model.training,
            "ref_model should be in eval mode",
        ), "ref_model should be in eval mode"
        assert (
            self.model.training,
            "model should be in train mode",
        ), "model should be in train mode"
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        # 1. sample chains by ref_model
        m_hat_traj, out_data_list = self.sample_chain(
            batch, desc="f'RlSampling-{_}/{n_samples}'"
        )
        # 2. calculate reward
        # DockingTestCallback.on_test_epoch_end
        self.evaluate_molecule_and_get_reward(out_data_list)

        # 3. train
        self.assert_parameter_freeze_status()
        pass

    def evaluate_molecule_and_get_reward(self, out_data_list):
        """
        Evaluate the molecule and get the reward.
        """
        results, recon_dict = reconstruct_mol_and_filter_invalid(self.outputs)

        if len(results) == 0:
            print("skip validation, no mols are valid & complete")
            return

        out_metrics = self.metric.evaluate(results)

        print(f"out_metrics: {out_metrics}")
        pass

    def reward(self, metrics):
        """
        Calculate the reward for the sampled chains.
        Use reward function given by molFormer.
        R_vina = 1 / (1 + 10 ^ (0.625 * (vina_score + 10)))
        R_qed = I[QED > 0.25]
        R_sa = I[SA > 0.59]

        Returns:
            reward: 1/n * sum(R_), n is the number of reward functions used.
        """
        reward_list = self.cfg.RL.reward
        if "vina_score" in reward_list:
            vina_score_list = metrics["vina_score_list"]
            R_vina = 1 / (1 + 10 ** (0.625 * (metrics["vina_score"] + 10)))

        pass

    def sample_chain(self, batch, desc=""):
        """
        Sample chains from the reference model.
        Calculate the reward and return the sampled chains.
        Args:
            batch: a data object
        Returns:
            m_hat_traj: a list of m_hat_t | [{"input": (t, gamma_coord), "output": (coord_pred, p0_h_pred, _)}]

            out_data_list: a list of pyg data objects, each corresponds to a sampled chain.
        """
        protein_pos, protein_v, batch_protein, ligand_pos, ligand_v, batch_ligand = (
            batch.protein_pos,
            batch.protein_atom_feature.float(),
            batch.protein_element_batch,
            batch.ligand_pos,
            batch.ligand_atom_feature_full,
            batch.ligand_element_batch,
        )  # get the data from the batch
        # batch is a data object
        # protein_pos: [N_pro,3]
        # protein_v: [N_pro,27]
        # batch_protein: [N_pro]
        # ligand_pos: [N_lig,3]
        # ligand_v: [N_lig,13]
        # protein_element_batch: [N_protein]

        num_protein = batch_protein.max().item() + 1
        assert num_protein == len(
            batch
        ), f"num_protein: {num_protein} != len(batch): {len(batch)}"
        assert self.cfg.train.pos_noise_std == 0, "pos_noise_std should be 0"
        assert (
            self.cfg.dynamics.center_pos_mode == "protein"
        ), "center_pos_mode should be protein"
        protein_pos, ligand_pos, offset = center_pos(
            protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.cfg.dynamics.center_pos_mode,  # protein
        )

        sample_num_atoms = self.cfg.evaluation.sample_num_atoms

        # determine the number of atoms in the ligand
        if sample_num_atoms == "prior":  # default
            ligand_num_atoms = []
            for data_id in range(len(batch)):
                data = batch[data_id]
                pocket_size = atom_num.get_space_size(
                    data.protein_pos.detach().cpu().numpy()
                    * self.cfg.data.normalizer_dict.pos
                )
                ligand_num_atoms.append(
                    atom_num.sample_atom_num(pocket_size).astype(int)
                )
            batch_ligand = torch.repeat_interleave(
                torch.arange(len(batch)), torch.tensor(ligand_num_atoms)
            ).to(ligand_pos.device)
            ligand_num_atoms = torch.tensor(
                ligand_num_atoms, dtype=torch.long, device=ligand_pos.device
            )
        # elif sample_num_atoms == "ref":
        #     batch_ligand = batch.ligand_element_batch
        #     ligand_num_atoms = scatter_sum(
        #         torch.ones_like(batch_ligand), batch_ligand, dim=0
        #     ).to(ligand_pos.device)
        else:
            raise ValueError(f"sample_num_atoms mode: {sample_num_atoms} not supported")

        ligand_cum_atoms = torch.cat(
            [
                torch.tensor([0], dtype=torch.long, device=ligand_pos.device),
                ligand_num_atoms.cumsum(dim=0),
            ]
        )

        theta_chain, sample_chain, y_chain, m_hat_traj = self.ref_model.sample_chain(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            # n_nodes=n_nodes,
            sample_steps=self.cfg.evaluation.sample_steps,
            n_nodes=num_protein,
            # ligand_pos=ligand_pos,  # for debug only
            desc=desc,
        )

        """
            m_hat_traj = [{
                "input": (
                    t,
                    gamma_coord,
                ),
                "output": (coord_pred, p0_h_pred, _),
            }]
        """
        # restore ligand to original position
        final = sample_chain[-1]  # mu_pos_final, k_final, k_hat_final
        pred_pos, one_hot = final[0] + offset[batch_ligand], final[1]

        # along with normalizer
        pred_pos = pred_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos,
            dtype=torch.float32,
            device=ligand_pos.device,
        )
        out_batch = copy.deepcopy(batch)
        out_batch.protein_pos = out_batch.protein_pos * torch.tensor(
            self.cfg.data.normalizer_dict.pos,
            dtype=torch.float32,
            device=ligand_pos.device,
        )

        pred_v = one_hot.argmax(dim=-1)
        # TODO: ugly, should be done in metrics.py (but needs a way to make it compatible with pyg batch)
        pred_atom_type = trans.get_atomic_number_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        )  # List[int]

        # for visualization
        atom_type = [
            trans.MAP_ATOM_TYPE_ONLY_TO_INDEX[i] for i in pred_atom_type
        ]  # List[int]
        atom_type = torch.tensor(
            atom_type, dtype=torch.long, device=ligand_pos.device
        )  # [N_lig]

        # for reconstruction
        pred_aromatic = trans.is_aromatic_from_index(
            pred_v, mode=self.cfg.data.transform.ligand_atom_mode
        )  # List[bool]

        # print('[DEBUG]', num_graphs, len(ligand_cum_atoms))

        # add necessary dict to new pyg batch
        out_batch.x, out_batch.pos = atom_type, pred_pos
        out_batch.atom_type = torch.tensor(
            pred_atom_type, dtype=torch.long, device=ligand_pos.device
        )
        out_batch.is_aromatic = torch.tensor(
            pred_aromatic, dtype=torch.long, device=ligand_pos.device
        )
        # out_batch.mol = results

        _slice_dict = {
            "x": ligand_cum_atoms,
            "pos": ligand_cum_atoms,
            "atom_type": ligand_cum_atoms,
            "is_aromatic": ligand_cum_atoms,
            # "mol": out_batch._slice_dict["ligand_filename"],
        }
        _inc_dict = {
            "x": out_batch._inc_dict[
                "ligand_element"
            ],  # [0] * B, TODO: figure out what this is
            "pos": out_batch._inc_dict["ligand_pos"],
            "atom_type": out_batch._inc_dict["ligand_element"],
            "is_aromatic": out_batch._inc_dict["ligand_element"],
            # "mol": out_batch._inc_dict["ligand_filename"],
        }
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        out_data_list = out_batch.to_data_list()

        return m_hat_traj, out_data_list

    def configure_optimizers(self):
        pass
