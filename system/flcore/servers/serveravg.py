import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy
import copy
# import yaml
import statistics


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)
        self.args = args

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.device = args.device


    def train(self):
        if self.eval_gap <= 0:
            raise ValueError(f"eval_gap must be positive, got {self.eval_gap}")

        # Evaluate the initial global model before any local training.
        for client in self.clients:
            client.set_parameters(self.global_model)

        print("\n-------------Round number: 0-------------")
        print("\nEvaluate initial global model")
        self.evaluate()

        # Run exactly global_rounds local-training and aggregation rounds.
        for round_idx in range(1, self.global_rounds + 1):
            start_time = time.time()

            self.selected_clients = self.select_clients()
            self.send_models()

            for client in self.selected_clients:
                client.train()

            self.receive_models()

            if not self.uploaded_models:
                raise RuntimeError(
                    f"No client model was uploaded at round {round_idx}. "
                    "Check client_drop_rate, time_threthold, and client selection."
                )

            # Standard sample-count-weighted FedAvg aggregation.
            self.aggregate_parameters()

            # Server.evaluate() evaluates client-held models. Synchronize all
            # clients first so the metric is measured on the aggregated global
            # model, not on their stale local models.
            for client in self.clients:
                client.set_parameters(self.global_model)

            round_time = time.time() - start_time
            self.Budget.append(round_time)

            if round_idx % self.eval_gap == 0 or round_idx == self.global_rounds:
                print(f"\n-------------Round number: {round_idx}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            print("-" * 25, "time cost", "-" * 25, round_time)

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc],
                top_cnt=self.top_cnt,
            ):
                print(f"Early stopping triggered at round {round_idx}.")
                break

        if self.rs_test_acc:
            print("\nBest accuracy.")
            print(max(self.rs_test_acc))

        if self.Budget:
            print("\nAverage time cost per round.")
            print(sum(self.Budget) / len(self.Budget))

        self.save_results()
        self.save_global_model()
