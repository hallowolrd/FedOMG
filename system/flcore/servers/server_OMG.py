import copy
import time

import torch
from torch.nn.utils import vector_to_parameters
from torch.optim.lr_scheduler import StepLR

from flcore.clients.client_OMG import client_OMG
from flcore.servers.serverbase import Server
from utils.model_utils import ParamDict


def tensor_diagnostics(name, tensor):
    """Print compact numerical diagnostics for a tensor."""
    values = tensor.detach().float()
    finite_mask = torch.isfinite(values)
    finite_count = int(finite_mask.sum().item())

    if finite_count == 0:
        print(
            f"[DIAG] {name}: shape={tuple(values.shape)}, "
            "NO FINITE VALUES",
            flush=True,
        )
        return

    finite_values = values[finite_mask]
    print(
        f"[DIAG] {name}: "
        f"shape={tuple(values.shape)}, "
        f"norm={finite_values.norm().item():.6e}, "
        f"mean={finite_values.mean().item():.6e}, "
        f"max_abs={finite_values.abs().max().item():.6e}, "
        f"finite={finite_count}/{values.numel()}",
        flush=True,
    )


def model_diagnostics(name, model):
    parameters = [p.detach().reshape(-1) for p in model.parameters()]
    if not parameters:
        print(f"[DIAG] {name}: model has no parameters", flush=True)
        return
    tensor_diagnostics(name, torch.cat(parameters))


class FedOMG(Server):
    """Federated Learning via On-server Matching Gradient."""

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(client_OMG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        self.omg_learning_rate = args.omg_learning_rate
        self.omg_step_size = args.omg_step_size
        self.omg_c = args.omg_c
        self.omg_rounds = args.omg_rounds
        self.omg_momentum = args.omg_momentum
        self.omg_gamma = args.omg_gamma
        self.omg_meta_lr = args.omg_meta_lr

        self.debug = getattr(args, "debug", False)

        if self.debug:
            print(
                "[DIAG] FedOMG runtime config: "
                f"kappa={self.omg_c:.8g}, "
                f"meta_lr={self.omg_meta_lr:.8g}, "
                "optimizer=PyTorch SGD, "
                f"optimizer_steps={self.omg_rounds}, "
                f"optimizer_lr={self.omg_learning_rate:.8g}, "
                f"momentum={self.omg_momentum:.8g}, "
                f"step_size={self.omg_step_size}, "
                f"scheduler_gamma={self.omg_gamma:.8g}",
                flush=True,
            )

        if self.omg_c < 0:
            raise ValueError(f"omg_c (kappa) must be non-negative, got {self.omg_c}")
        if self.omg_rounds < 0:
            raise ValueError(
                f"omg_rounds must be non-negative, got {self.omg_rounds}"
            )
        if self.omg_rounds > 0 and self.omg_learning_rate <= 0:
            raise ValueError(
                "omg_learning_rate must be positive when omg_rounds > 0"
            )
        if self.omg_rounds > 0 and self.omg_step_size <= 0:
            raise ValueError(
                "omg_step_size must be positive when omg_rounds > 0"
            )
        if self.eval_gap <= 0:
            raise ValueError(f"eval_gap must be positive, got {self.eval_gap}")

    def train(self):
        # Evaluate the initial global model at round 0. The loop below then
        # performs exactly global_rounds local-training/server-update rounds.
        print("\n-------------Round number: 0-------------")
        print("\nEvaluate initial global model")
        self.evaluate()

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

            if self.debug:
                print("\n========== DIAG: CLIENT MODELS ==========", flush=True)
                print("Uploaded client IDs:", self.uploaded_ids, flush=True)
                print("Uploaded clients:", len(self.uploaded_models), flush=True)
                model_diagnostics("Global model before OMG", self.global_model)
                for client_id, model in zip(self.uploaded_ids, self.uploaded_models):
                    model_diagnostics(f"Client {client_id} model", model)
                print("========== END CLIENT MODELS ==========", flush=True)

            # FedOMG constructs the client model deltas directly. Calling
            # receive_grads() here is unnecessary: that method pre-multiplies
            # each individual delta by its FedAvg weight, whereas Eq. (15)
            # needs the unweighted client deltas plus a separate reference
            # FedAvg weight vector.
            updated_state = self.omg_high(
                meta_weights=self.global_model,
                inner_weights=self.uploaded_models,
                lr_meta=self.omg_meta_lr,
            )
            self.global_model.load_state_dict(copy.deepcopy(updated_state))

            round_time = time.time() - start_time
            self.Budget.append(round_time)

            if round_idx % self.eval_gap == 0 or round_idx == self.global_rounds:
                # Evaluate the aggregated global model, not each client's
                # preceding local model.
                for client in self.clients:
                    client.set_parameters(self.global_model)

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

    def omg_high(self, meta_weights, inner_weights, lr_meta):
        """
        Build client parameter deltas, solve Eq. (15), and apply Eq. (16).

        Each column of gradient_matrix is
            g_u = theta_local_u - theta_global.

        uploaded_weights contains the normalized sample-count weights
        generated by Server.receive_models().
        """
        if len(inner_weights) == 0:
            raise ValueError("FedOMG received no client models.")

        if len(self.uploaded_weights) != len(inner_weights):
            raise ValueError(
                "Uploaded model/weight count mismatch: "
                f"{len(inner_weights)} models vs "
                f"{len(self.uploaded_weights)} weights."
            )

        global_named_parameters = list(meta_weights.named_parameters())
        if not global_named_parameters:
            raise ValueError("The global model has no trainable parameters.")

        global_names = [name for name, _ in global_named_parameters]
        global_parameters = dict(global_named_parameters)

        client_deltas = []
        for client_index, client_model in enumerate(inner_weights):
            client_named_parameters = list(client_model.named_parameters())
            client_names = [name for name, _ in client_named_parameters]

            if client_names != global_names:
                raise ValueError(
                    f"Parameter-name mismatch for uploaded client index "
                    f"{client_index}."
                )

            client_parameters = dict(client_named_parameters)
            delta_parts = []

            for name in global_names:
                global_parameter = global_parameters[name]
                client_parameter = client_parameters[name]

                if global_parameter.shape != client_parameter.shape:
                    raise ValueError(
                        f"Parameter-shape mismatch for {name}: "
                        f"global={tuple(global_parameter.shape)}, "
                        f"client={tuple(client_parameter.shape)}"
                    )

                delta_parts.append(
                    (
                        client_parameter.detach()
                        - global_parameter.detach()
                    ).reshape(-1)
                )

            client_deltas.append(torch.cat(delta_parts))

        # Shape: [number_of_model_parameters, number_of_uploaded_clients].
        gradient_matrix = torch.stack(client_deltas, dim=1)

        sample_weights = torch.as_tensor(
            self.uploaded_weights,
            dtype=gradient_matrix.dtype,
            device=gradient_matrix.device,
        )

        if not torch.isfinite(sample_weights).all():
            raise FloatingPointError("Uploaded client weights contain NaN/Inf.")
        if (sample_weights < 0).any() or sample_weights.sum().item() <= 0:
            raise ValueError(
                "Uploaded client weights must be non-negative "
                "and have a positive sum."
            )

        # Normalize again to protect against small numerical drift.
        sample_weights = sample_weights / sample_weights.sum()

        if self.debug:
            print(
                "[DIAG] uploaded sample weights: "
                f"sum={sample_weights.sum().item():.8f}, "
                f"min={sample_weights.min().item():.8e}, "
                f"max={sample_weights.max().item():.8e}, "
                f"std={sample_weights.std(unbiased=False).item():.8e}",
                flush=True,
            )
            for client_id, client_delta in zip(self.uploaded_ids, client_deltas):
                tensor_diagnostics(f"Client {client_id} parameter delta", client_delta)

        g_fl = gradient_matrix @ sample_weights

        g_igd, gamma, objective = self.omg_low(
            gradient_matrix=gradient_matrix,
            reference_weights=sample_weights,
        )

        if not torch.isfinite(g_igd).all():
            raise FloatingPointError("FedOMG produced NaN/Inf in g_IGD.")

        if self.debug:
            tensor_diagnostics("FedAvg reference update g_FL", g_fl)
            tensor_diagnostics("FedOMG invariant update g_IGD", g_igd)

            update_difference = g_igd - g_fl
            relative_l2_difference = update_difference.norm() / (
                g_fl.norm() + 1e-12
            )
            cosine_similarity = torch.dot(g_igd, g_fl) / (
                g_igd.norm() * g_fl.norm() + 1e-12
            )
            print(
                "[DIAG] FedOMG vs FedAvg update: "
                f"kappa={self.omg_c:.8g}, "
                f"max_abs_diff={update_difference.abs().max().item():.10e}, "
                f"relative_l2_diff={relative_l2_difference.item():.10e}, "
                f"cosine_similarity={cosine_similarity.item():.10e}, "
                f"norm_ratio={g_igd.norm().item() / (g_fl.norm().item() + 1e-12):.10e}",
                flush=True,
            )

            gamma_sum = gamma.sum().item()
            gamma_min = gamma.min().item()
            gamma_max = gamma.max().item()
            gamma_simplex_ok = (
                gamma_min >= -1e-8 and abs(gamma_sum - 1.0) <= 1e-5
            )
            gamma_weight_difference = gamma - sample_weights
            print(
                "[DIAG] gamma constraints/reference: "
                f"sum={gamma_sum:.10e}, min={gamma_min:.10e}, "
                f"max={gamma_max:.10e}, simplex_ok={gamma_simplex_ok}, "
                f"||gamma-sample_weights||_2={gamma_weight_difference.norm().item():.10e}, "
                f"max_abs_gamma_weight_diff={gamma_weight_difference.abs().max().item():.10e}",
                flush=True,
            )
            print(
                "[DIAG] gamma values: "
                f"{gamma.detach().cpu().tolist()}",
                flush=True,
            )
            print(f"[DIAG] server meta learning rate: {lr_meta}", flush=True)

        flat_global_parameters = torch.cat(
            [parameter.detach().reshape(-1) for _, parameter in global_named_parameters]
        )

        # g_u is a parameter delta (local - global), i.e. the direction
        # that FedAvg adds to the global parameters. Therefore we ADD the
        # FedOMG delta here, matching the official implementation's update
        # convention and standard model-delta aggregation.
        updated_flat_parameters = (
            flat_global_parameters + float(lr_meta) * g_igd
        )

        if not torch.isfinite(updated_flat_parameters).all():
            raise FloatingPointError(
                "FedOMG produced NaN/Inf in the updated global parameters."
            )

        with torch.no_grad():
            vector_to_parameters(
                updated_flat_parameters,
                meta_weights.parameters(),
            )

        if self.debug:
            applied_flat_parameters = torch.cat(
                [
                    parameter.detach().reshape(-1)
                    for parameter in meta_weights.parameters()
                ]
            )
            intended_update = float(lr_meta) * g_igd
            actual_update = applied_flat_parameters - flat_global_parameters
            application_error = actual_update - intended_update
            print(
                "[DIAG] applied global update: "
                f"actual_norm={actual_update.norm().item():.10e}, "
                f"intended_norm={intended_update.norm().item():.10e}, "
                f"max_abs_application_error={application_error.abs().max().item():.10e}, "
                f"allclose={torch.allclose(actual_update, intended_update, rtol=1e-5, atol=1e-6)}",
                flush=True,
            )

            # Match Server.aggregate_parameters(): start from the first
            # uploaded model, zero its parameters, and add weighted models
            # in upload order. This is a parameter-only comparison.
            fedavg_reference_model = copy.deepcopy(inner_weights[0])
            for parameter in fedavg_reference_model.parameters():
                parameter.data.zero_()

            for weight, client_model in zip(self.uploaded_weights, inner_weights):
                for server_parameter, client_parameter in zip(
                    fedavg_reference_model.parameters(),
                    client_model.parameters(),
                ):
                    server_parameter.data += client_parameter.data.clone() * weight

            fedavg_flat_parameters = torch.cat(
                [
                    parameter.detach().reshape(-1)
                    for parameter in fedavg_reference_model.parameters()
                ]
            )
            aggregation_diff = applied_flat_parameters - fedavg_flat_parameters
            relative_l2_diff = aggregation_diff.norm() / (
                fedavg_flat_parameters.norm() + 1e-12
            )
            is_close = torch.allclose(
                applied_flat_parameters,
                fedavg_flat_parameters,
                rtol=1e-5,
                atol=1e-6,
            )
            print(
                "[DIAG] Same-round FedAvg vs FedOMG aggregation: "
                f"kappa={self.omg_c:.8g}, "
                f"max_abs_diff={aggregation_diff.abs().max().item():.10e}, "
                f"relative_l2_diff={relative_l2_diff.item():.10e}, "
                f"allclose={is_close}, "
                f"meta_lr={float(lr_meta):.8g}",
                flush=True,
            )

        return ParamDict(meta_weights.state_dict())

    def omg_low(self, gradient_matrix, reference_weights):
        """
        Optimize Gamma* for Eq. (15), then calculate g_IGD using Eq. (16).

        Args:
            gradient_matrix: Tensor [P, K], each column is one uploaded
                client's unweighted parameter delta.
            reference_weights: Tensor [K], normalized sample-count weights.

        Returns:
            g_igd: FedOMG invariant update, Tensor [P].
            best_gamma: non-negative coefficients summing to one, Tensor [K].
            best_objective: optimized Eq. (15) objective (up to a positive
                common normalization factor).
        """
        if gradient_matrix.ndim != 2:
            raise ValueError(
                "gradient_matrix must be a 2-D tensor [num_parameters, num_clients]."
            )

        num_parameters, num_clients = gradient_matrix.shape
        if num_parameters == 0 or num_clients == 0:
            raise ValueError("gradient_matrix must not be empty.")
        if reference_weights.numel() != num_clients:
            raise ValueError(
                "reference_weights length must equal the number of client updates."
            )

        kappa = float(self.omg_c)
        eps = 1e-12

        # The reference update is the weighted FedAvg model delta.
        g_fl = gradient_matrix @ reference_weights
        g_fl_norm = torch.linalg.vector_norm(g_fl)

        uniform_gamma = torch.full(
            (num_clients,),
            1.0 / num_clients,
            dtype=gradient_matrix.dtype,
            device=gradient_matrix.device,
        )

        # Corollary 1: when kappa -> 0, FedOMG reduces to the reference FL
        # update. This also avoids an unnecessary optimization pass.
        if kappa == 0.0:
            if self.debug:
                print(
                    "[DIAG] OMG optimizer skipped: kappa=0; "
                    "gamma is a uniform placeholder and objective was not evaluated.",
                    flush=True,
                )
            return g_fl, uniform_gamma, 0.0

        if g_fl_norm.item() <= eps:
            if self.debug:
                print(
                    "[DIAG] OMG optimizer skipped: reference update norm "
                    f"({g_fl_norm.item():.10e}) <= eps ({eps:.1e}).",
                    flush=True,
                )
            return g_fl, uniform_gamma, 0.0

        # Form the K x K Gram matrix. This avoids repeatedly operating on
        # the full model-parameter dimension during Gamma optimization.
        gram = gradient_matrix.transpose(0, 1) @ gradient_matrix
        gram = 0.5 * (gram + gram.transpose(0, 1))

        # Divide the Gram matrix by a positive scalar for numerical
        # conditioning. This does not change the minimizer of Eq. (15).
        scale_squared = torch.diagonal(gram).mean().clamp_min(eps)
        normalized_gram = gram / scale_squared

        cross_term = normalized_gram @ reference_weights
        reference_norm_scaled = torch.sqrt(
            torch.clamp(
                reference_weights @ normalized_gram @ reference_weights,
                min=0.0,
            )
        )

        logits = torch.zeros(
            num_clients,
            dtype=gradient_matrix.dtype,
            device=gradient_matrix.device,
            requires_grad=True,
        )

        optimizer = torch.optim.SGD(
            [logits],
            lr=self.omg_learning_rate,
            momentum=self.omg_momentum,
        )
        scheduler = StepLR(
            optimizer,
            step_size=self.omg_step_size,
            gamma=self.omg_gamma,
        )

        best_gamma = uniform_gamma.detach().clone()
        best_objective = float("inf")
        initial_objective = None
        last_objective = None
        optimizer_steps = 0

        for step in range(self.omg_rounds + 1):
            optimizer.zero_grad()

            gamma = torch.softmax(logits, dim=0)
            mixed_norm_scaled = torch.sqrt(
                torch.clamp(
                    gamma @ normalized_gram @ gamma,
                    min=eps,
                )
            )

            # Eq. (15):
            # min_Gamma <Gamma G, g_FL> + kappa * ||g_FL|| * ||Gamma G||
            # The normalized Gram matrix scales the complete objective by
            # one positive constant, leaving Gamma* unchanged.
            objective = (
                torch.dot(gamma, cross_term)
                + kappa * reference_norm_scaled * mixed_norm_scaled
            )

            if not torch.isfinite(objective):
                raise FloatingPointError(
                    f"Non-finite FedOMG objective at optimizer step {step}."
                )

            objective_value = float(objective.detach().item())
            if initial_objective is None:
                initial_objective = objective_value
            last_objective = objective_value

            if objective_value < best_objective:
                best_objective = objective_value
                best_gamma = gamma.detach().clone()

            if step < self.omg_rounds:
                objective.backward()
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1

        if self.debug:
            gamma_sum = best_gamma.sum().item()
            gamma_min = best_gamma.min().item()
            gamma_max = best_gamma.max().item()
            gamma_simplex_ok = (
                gamma_min >= -1e-8 and abs(gamma_sum - 1.0) <= 1e-5
            )
            print(
                "[DIAG] OMG optimizer: "
                "name=PyTorch SGD, status=completed, "
                f"steps={optimizer_steps}/{self.omg_rounds}, "
                f"initial_objective={initial_objective:.10e}, "
                f"last_objective={last_objective:.10e}, "
                f"best_objective={best_objective:.10e}, "
                f"best_gamma_sum={gamma_sum:.10e}, "
                f"best_gamma_min={gamma_min:.10e}, "
                f"best_gamma_max={gamma_max:.10e}, "
                f"simplex_ok={gamma_simplex_ok}",
                flush=True,
            )
            print(
                "[DIAG] optimized gamma values: "
                f"{best_gamma.detach().cpu().tolist()}",
                flush=True,
            )

        mixed_update = gradient_matrix @ best_gamma
        mixed_update_norm = torch.linalg.vector_norm(mixed_update)

        if mixed_update_norm.item() <= eps:
            # Eq. (16) is undefined if ||Gamma* G|| is zero. In this
            # degenerate case, fall back to the well-defined FedAvg update.
            if self.debug:
                print(
                    "[DIAG] Eq. (16) fallback: mixed update norm "
                    f"({mixed_update_norm.item():.10e}) <= eps ({eps:.1e}); "
                    "using g_FL.",
                    flush=True,
                )
            g_igd = g_fl
        else:
            # Eq. (16). Do NOT divide by (1 + kappa**2): that factor is not
            # present in the FedOMG paper's Theorem 1 / Algorithm 1.
            g_igd = (
                g_fl
                + kappa
                * g_fl_norm
                / mixed_update_norm
                * mixed_update
            )

        return g_igd, best_gamma, best_objective
