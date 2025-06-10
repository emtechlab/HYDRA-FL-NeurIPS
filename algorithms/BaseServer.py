import torch
import torch.nn as nn
import numpy as np
import copy
import time
import wandb
import sys

from .measures import *

__all__ = ["BaseServer"]


class BaseServer:
    def __init__(
        self,
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        n_rounds=200,
        sample_ratio=0.1,
        local_epochs=5,
        device="cuda:0",
    ):
        """
        Server class controls the overall experiment.
        """
        self.algo_params = algo_params
        self.num_classes = data_distributed["num_classes"]
        self.model = model
        self.testloader = data_distributed["global"]["test"]
        self.criterion = nn.CrossEntropyLoss()
        self.data_distributed = data_distributed
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_ratio = sample_ratio
        self.n_rounds = n_rounds
        self.device = device
        self.n_clients = len(data_distributed["local"].keys())
        self.local_epochs = local_epochs
        self.server_results = {
            "client_history": [],
            "test_accuracy": [],
        }

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

            # print(type(updated_local_weights))
            # print(len(updated_local_weights))
            # print(sum(p.numel() for p in updated_local_weights[0].values()))

            user_updates = []
            for net_id, net in enumerate(updated_local_weights):
                params = []
                for _, (name, param) in enumerate(net.items()):
                    params = param.view(-1).data if len(params) == 0 else torch.cat((params, param.view(-1).data))
                user_updates = params[None, :] if len(user_updates) == 0 else torch.cat((user_updates, params[None, :]), 0)

            print(user_updates.shape)

            #new
            # agg_updates = torch.mean(user_updates, dim = 0)

            # user_updates = self.full_trim(user_updates, 2)

            # mal_update = self.ndss_trim(user_updates, 2, 'std')
            # user_updates[0] = mal_update
            # user_updates[1] = mal_update

            # agg_updates = torch.mean(user_updates, dim = 0)

            # print("running trmean without ndss attack")
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
            self.client.download_global(server_weights, server_optimizer)

            # Local training
            local_results, local_size = self.client.train()

            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results

    def _client_sampling(self, round_idx):
        """Sample clients by given sampling ratio"""

        # make sure for same client sampling for fair comparison
        np.random.seed(round_idx)
        clients_per_round = max(int(self.n_clients * self.sample_ratio), 1)
        sampled_clients = np.random.choice(
            self.n_clients, clients_per_round, replace=False
        )

        return sampled_clients

    # def _personalized_evaluation(self):
    #    """Personalized FL performance evaluation for all clients."""

    #     finetune_results = {}

    #     server_weights = self.model.state_dict()
    #     server_optimizer = self.optimizer.state_dict()

    #     # Client finetuning stage
    #     for client_idx in [client_idx for client_idx in self.n_clients]:
    #         self._set_client_data(client_idx)

    #         # Local finetuning
    #         local_results = self.client.finetune(server_weights, server_optimizer)
    #         finetune_results = self._results_updater(finetune_results, local_results)

    #         # Reset local model
    #         self.client.reset()

    #     # Get overall statistics
    #     local_results = {
    #         "local_train_acc": np.mean(round_results["train_acc"]),
    #         "local_test_acc": np.mean(round_results["test_acc"]),
    #     }
    #     wandb.log(local_results, step=round_idx)

    #     return finetune_results

    def _set_client_data(self, client_idx):
        """Assign local client datasets."""
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.testloader = self.data_distributed["global"]["test"]

    def _aggregation(self, w, ns):
        """Average locally trained model parameters"""
        prop = torch.tensor(ns, dtype=torch.float)
        prop /= torch.sum(prop)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * prop[0]

        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * prop[i]

        return copy.deepcopy(w_avg)

    def _results_updater(self, round_results, local_results):
        """Combine local results as clean format"""

        for key, item in local_results.items():
            if key not in round_results.keys():
                round_results[key] = [item]
            else:
                round_results[key].append(item)

        return round_results

    def _print_start(self):
        """Print initial log for experiment"""

        if self.device == "cpu":
            return "cpu"

        if isinstance(self.device, str):
            device_idx = int(self.device[-1])
        elif isinstance(self.device, torch._device):
            device_idx = self.device.index

        device_name = torch.cuda.get_device_name(device_idx)
        print("")
        print("=" * 50)
        print("Train start on device: {}".format(device_name))
        print("=" * 50)

    def _print_stats(self, round_results, test_accs, round_idx, round_elapse):
        print(
            "[Round {}/{}] Elapsed {}s (Current Time: {})".format(
                round_idx + 1,
                self.n_rounds,
                round(round_elapse, 1),
                time.strftime("%H:%M:%S"),
            )
        )
        print(
            "[Local Stat (Train Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
                round_results["train_acc"],
                np.mean(round_results["train_acc"]),
                np.std(round_results["train_acc"]),
            )
        )

        print(
            "[Local Stat (Test Acc)]: {}, Avg - {:2.2f} (std {:2.2f})".format(
                round_results["test_acc"],
                np.mean(round_results["test_acc"]),
                np.std(round_results["test_acc"]),
            )
        )

        print("[Server Stat] Acc - {:2.2f}".format(test_accs))

    def _wandb_logging(self, round_results, round_idx):
        """Log on the W&B server"""

        # Local round results
        local_results = {
            "local_train_acc": np.mean(round_results["train_acc"]),
            "local_test_acc": np.mean(round_results["test_acc"]),
        }
        wandb.log(local_results, step=round_idx)

        # Server round results
        server_results = {"server_test_acc": self.server_results["test_accuracy"][-1]}
        wandb.log(server_results, step=round_idx)

    def _update_and_evaluate(self, ag_weights, round_results, round_idx, start_time):
        """Evaluate experiment statistics."""

        # Update Global Server Model
        self.model.load_state_dict(ag_weights)

        # Measure Accuracy Statistics
        test_acc = evaluate_model(self.model, self.testloader, device=self.device,)
        self.server_results["test_accuracy"].append(test_acc)

        # Evaluate Personalized FL performance
        eval_results = get_round_personalized_acc(
            round_results, self.server_results, self.data_distributed
        )
        wandb.log(eval_results, step=round_idx)

        # Change learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        round_elapse = time.time() - start_time

        # Log and Print
        self._wandb_logging(round_results, round_idx)
        self._print_stats(round_results, test_acc, round_idx, round_elapse)
        print("-" * 50)
        return test_acc
