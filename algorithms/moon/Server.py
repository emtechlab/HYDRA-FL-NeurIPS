import time
import copy
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.moon.ClientTrainer import ClientTrainer
from algorithms.moon.criterion import ModelContrastiveLoss
from algorithms.BaseServer import BaseServer
from algorithms.measures import *

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, **kwargs
        )
        """
        Server class controls the overall experiment.
        """
        moon_criterion = ModelContrastiveLoss(algo_params.mu, algo_params.tau)

        self.client = ClientTrainer(
            moon_criterion,
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
        )

        self.prev_locals = []
        self._init_prev_locals()

        print("\n>>> MOON Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()
        max_acc = 0
        for round_idx in range(self.n_rounds):

            # Initial Model Statistics
            if round_idx == 0:
                test_acc = evaluate_model(
                    self.model, self.testloader, device=self.device
                )
                self.server_results["test_accuracy"].append(test_acc)

            start_time = time.time()

            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            self.server_results["client_history"].append(sampled_clients)

            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results = self._clients_training(
                sampled_clients
            )
            
            user_updates = []
            for net_id, net in enumerate(updated_local_weights):
                params = []
                for _, (name, param) in enumerate(net.items()):
                    params = param.view(-1).data if len(params) == 0 else torch.cat((params, param.view(-1).data))
                user_updates = params[None, :] if len(user_updates) == 0 else torch.cat((user_updates, params[None, :]), 0)

            print("aggregating from moon server file: ", user_updates.shape)

            #new
            # agg_updates = torch.mean(user_updates, dim = 0)

            # user_updates = self.full_trim(user_updates, 2)

            # mal_update = self.ndss_trim(user_updates, 2, 'std')
            # user_updates[0] = mal_update
            # user_updates[1] = mal_update

            # agg_updates = torch.mean(user_updates, dim = 0)

            agg_updates = self.tr_mean(user_updates, 2)

            del user_updates

            start_idx = 0
            state_dict = {}
            previous_name = 'none'
            for i, (name, param) in enumerate(self.model.state_dict().items()):
                start_idx = 0 if i == 0 else start_idx + len(
                    self.model.state_dict()[previous_name].data.view(-1))
                start_end = start_idx + len(self.model.state_dict()[name].data.view(-1))
                #                     print(np.shape(user_updates[k][start_idx:start_end]))
                params = agg_updates[start_idx:start_end].reshape(
                    self.model.state_dict()[name].data.shape)
                #                     params = local_models[0][0].state_dict()[name]
                state_dict[name] = params
                previous_name = name

            # Get aggregated weights & update global
            # ag_weights = self._aggregation(updated_local_weights, client_sizes)

            # Update global weights and evaluate statistics
            # self._update_and_evaluate(ag_weights, round_results, round_idx, start_time)
            test_acc = self._update_and_evaluate(state_dict, round_results, round_idx, start_time)
            if (test_acc>max_acc):
                max_acc = test_acc
            print("max acc: ", max_acc)

    def full_trim(self, v, f):
        '''
        Full-knowledge Trim attack. w.l.o.g., we assume the first f worker devices are compromised.
        v: the list of squeezed gradients
        f: the number of compromised worker devices
        '''
        vi_shape = v[0].unsqueeze(0).T.shape
        v_tran = v.T

        maximum_dim = torch.max(v_tran, dim=1)
        maximum_dim = maximum_dim[0].reshape(vi_shape)
        minimum_dim = torch.min(v_tran, dim=1)
        minimum_dim = minimum_dim[0].reshape(vi_shape)
        direction = torch.sign(torch.sum(v_tran, dim=-1, keepdims=True))
        directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

        for i in range(f):
            random_12 = 2
            #         random_12 = random.randint(0,9)
            tmp = directed_dim * (
                        (direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
            tmp = tmp.squeeze()
            v[i] = tmp
        return v
    
    def ndss_trim(self, all_updates, n_attackers, dev_type='sign', threshold=5.0, threshold_diff=1e-5):
    
        model_re = torch.mean(all_updates, 0)
        
        if dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([threshold]).cuda()  # compute_lambda_our(all_updates, model_re, n_attackers)

        threshold_diff = threshold_diff
        prev_loss = -1
        lamda_fail = lamda
        lamda_succ = 0

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            mal_updates = torch.stack([mal_update] * n_attackers)
            mal_updates = torch.cat((mal_updates, all_updates), 0)

            agg_grads = self.tr_mean(mal_updates, n_attackers)

            loss = torch.norm(agg_grads - model_re)

            if prev_loss < loss:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2
            prev_loss = loss

        mal_update = (model_re - lamda_succ * deviation)
        
        return mal_update


    def tr_mean(self, all_updates, n_attackers):
        sorted_updates = torch.sort(all_updates, 0)[0]
        out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates, 0)
        return out

    def _clients_training(self, sampled_clients):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes = [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:

            # Fetch client datasets
            self._set_client_data(client_idx)

            # Download global
            self.client.download_global(
                server_weights, server_optimizer, self.prev_locals[client_idx]
            )

            # Local training
            local_results, local_size = self.client.train()

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            for local_weights, client in zip(updated_local_weights, sampled_clients):
                self.prev_locals[client] = local_weights

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results

    def _init_prev_locals(self):
        weights = self.model.state_dict()
        for _ in range(self.n_clients):
            self.prev_locals.append(copy.deepcopy(weights))
