from .base import PLModelWrapper

import os
import shutil
import torch
import numpy as np

#from schnetpack.representation import SchNet
#import schnetpack.properties as structure


class SchNetModelWrapper(PLModelWrapper):
    """
    A wrapper for a SchNet model. Assumes that `model_path` contains a model
    checkpoint file with the name 'best_model'
    """
    def __init__(self, model_dir, **kwargs):
        if 'representation_type' in kwargs:
            self.representation_type = kwargs['representation_type']
        else:
            self.representation_type = 'node'


        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        self.model = torch.load(
            os.path.join(model_path, 'best_model'),
            map_location=torch.device('cpu')
        )


    def compute_loss(self, batch):
        true_DS = batch['DS']

        results = self.model.forward(batch)

        pred_DS = results['DS']

        DSdiff = (pred_DS - true_DS).detach().cpu().numpy()

        return {
            'DS_mse': np.mean(DSdiff**2),
            'batch_size': batch['DS'].shape[0],
            'natoms': int(sum(batch['_n_atoms']).detach().cpu().numpy()),
        }

    def compute_DS(self, batch):
        true_DS = batch['DS']
        results = self.model.forward(batch)
        pred_DS = results['DS']
        
        #print(true_DS)
        #print(type(true_DS))

        return {
            'true_DS': torch.Tensor(true_DS),
            'pred_DS': torch.Tensor(pred_DS),
        }


    # def compute_atom_representations(self, batch):

    #     # remember: .forward() overwrites the ['energy'] key
    #     true_eng = (batch['energy']/batch['_n_atoms']).clone()

    #     out = self.model.forward(batch)

    #     with torch.no_grad():
    #         representations = []

    #         if self.representation_type in ['node', 'both']:
    #             representations.append(batch['scalar_representation'])

    #         if self.representation_type in ['edge', 'both']:

    #             if isinstance(self.model.representation, SchNet):
    #                 x = batch['scalar_representation']
    #                 z = x.new_zeros((batch['scalar_representation'].shape[0], self.model.radial_basis.n_rbf))

    #                 idx_i = batch[structure.idx_i]
    #                 idx_j = batch[structure.idx_j]

    #                 z.index_add_(0, idx_i, batch['distance_representation'])
    #                 z.index_add_(0, idx_j, batch['distance_representation'])
    #             else:  # PaiNN
    #                 z = batch['vector_representation']
    #                 z = torch.mean(z, dim=1)  # average over cartesian dimension

    #             representations.append(z)

    #     representations = torch.cat(representations, dim=1)

    #     true_eng = (batch['energy']/batch['_n_atoms']).clone()
    #     per_atom_energies = torch.cat([
    #         true_eng.new_ones(n)*e for n,e in zip(batch['_n_atoms'], true_eng)
    #     ])

    #     return {
    #         'representations': representations,
    #         'representations_splits': batch['_n_atoms'].detach().cpu().numpy().tolist(),
    #         'representations_energy': true_eng,
    #     }



    def copy(self, model_path):
        shutil.copyfile(
            os.path.join(model_path, 'best_model'),
            os.path.join(os.getcwd(), 'best_model'),
        )


