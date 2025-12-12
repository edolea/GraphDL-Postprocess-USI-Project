from spatiotemporal_postprocessing.losses import get_loss
from spatiotemporal_postprocessing.nn import get_model
from spatiotemporal_postprocessing.datasets import get_datamodule
import xarray as xr
from tsl.ops.connectivity import adj_to_edge_index
import torch
from torch.utils.data import DataLoader
import mlflow
from omegaconf import DictConfig, OmegaConf
import hydra
from spatiotemporal_postprocessing.utils import log_prediction_plots
from tqdm import tqdm
import os
import random
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import shutil

CUDA_MEM = False

@torch.no_grad()
def collect_model_outputs(model, 
                          data_loader, 
                          device, 
                          edge_index=None):
    """
    Collect mu, sigma, and targets over an entire dataloader.

    Returns:
    mu_all, sigma_all, targets_all : tensors of shape [N, L, S]
        N = total number of forecast reference times over all batches.
    """
    model.eval()
    all_mu = []
    all_sigma = []
    all_targets = []

    for x_batch, y_batch in tqdm(data_loader, desc="Collecting model mu and sigma"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if edge_index is not None:
            pred_dist = model(x_batch, edge_index=edge_index)
        else:
            pred_dist = model(x_batch)

        mu = pred_dist.loc      # [B, L, S]
        sigma = pred_dist.scale # [B, L, S]

        all_mu.append(mu.cpu())
        all_sigma.append(sigma.cpu())
        all_targets.append(y_batch.cpu())

    mu_all = torch.cat(all_mu, dim=0)          # [N, L, S]
    sigma_all = torch.cat(all_sigma, dim=0)    # [N, L, S]
    targets_all = torch.cat(all_targets, dim=0)# [N, L, S]

    return mu_all, sigma_all, targets_all


@torch.no_grad()
def rank_histogram_for_lead(mu_all,
                            sigma_all,
                            targets_all,
                            device,
                            lead_idx=(1, 24, 48, 96),
                            n_samples=20,
):
    """
    Compute Talagrand / rank histogram for a single lead time.

    Parameters:
    mu_all, sigma_all, targets_all : torch.Tensor
        Shape [N, L, S].
    lead_idx : int
        0-based index of the lead time (e.g. lead_idx = 0, 23, 47, 95).
    n_samples : int
        Number of samples drawn from the predictive distribution.
        Histogram will have n_samples + 1 bins.
    device : str or torch.device
        Device on which to perform sampling.

    Returns:
    hist : torch.Tensor
        Shape [n_samples + 1], counts of ranks 0..n_samples.
    """

    # Slice lead: [N, S]
    mu_t = mu_all[:, lead_idx].to(device)        # [N, S]
    sigma_t = sigma_all[:, lead_idx].to(device)  # [N, S]
    y_t = targets_all[:, lead_idx].to(device)    # [N, S]

    # Build LogNormal distribution at this lead
    dist = torch.distributions.LogNormal(mu_t, sigma_t)

    # Sample shape: [n_samples, N, S]
    samples = dist.rsample((n_samples,))


    y_flat = y_t.reshape(-1)                    # [M]
    samples_flat = samples.reshape(n_samples, -1)  # [n_samples, M]

    # Keep only finite targets
    valid_flat = torch.isfinite(y_flat)
    y_flat = y_flat[valid_flat]                  # [M_valid]
    samples_flat = samples_flat[:, valid_flat]   # [n_samples, M_valid]

    # Compute ranks:
    # ranks in {0, 1, ..., n_samples}
    ranks = (samples_flat < y_flat.unsqueeze(0)).sum(dim=0)  # [M_valid]

    # Histogram of ranks
    hist = torch.bincount(ranks, minlength=n_samples + 1)

    return hist.cpu()


@torch.no_grad()
def evaluate_crps_weighted_over_batches(model,
                dataloader,
                criterion,              # MaskedCRPSLogNormal()
                tgt_denormalizer,       # dm.test_dataset.target_denormalizer
                device,
                edge_index=None,
                important_leads=(1, 24, 48, 96),
):
    """
    Evaluates CRPS (log-normal) on a dataset.

    Returns:
    crps_mean : float
        Mean CRPS over all batches, all leads, all stations (original range).
    crps_at_leads : dict
        Dictionary {lead_time: crps_value} for the requested lead times.
    """
    model.eval()

    total_crps_sum = 0.0
    total_valid_count = 0

    # per-lead CRPS
    lead_times = list(important_leads)
    lead_crps_sum = {t: 0.0 for t in lead_times}
    lead_crps_count = {t: 0 for t in lead_times}

    for x_batch, y_batch in tqdm(dataloader, desc="Evaluating CRPS"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)          # [B, L, S]

        if edge_index is not None:
            pred_dist = model(x_batch, edge_index=edge_index)
        else:
            pred_dist = model(x_batch)

        # ---- Overall CRPS (original range) ----
        pred_den = tgt_denormalizer(pred_dist)
        y_den = tgt_denormalizer(y_batch)

        mask_all = ~torch.isnan(y_den)
        valid_count = mask_all.sum().item()
        if valid_count == 0:
            continue

        batch_crps = criterion(pred_den, y_den)   # MaskedCRPSLogNormal
        total_crps_sum += batch_crps.item() * valid_count
        total_valid_count += valid_count

        # ---- CRPS per selected lead time ----
        # pred_den.loc and .scale are [B, L, S]
        for t in lead_times:
            lead_idx = t - 1          # lead time 1 -> index 0, etc.
            if lead_idx >= y_den.size(1):
                continue  # safety check, in case L < max requested lead

            y_t = y_den[:, lead_idx, :]
            mu_t = pred_den.loc[:, lead_idx, :]
            sigma_t = pred_den.scale[:, lead_idx, :]

            mask = ~torch.isnan(y_t)
            lead_count = mask.sum().item()
            if lead_count == 0:
                continue

            y_mask = y_t[mask]
            mu_mask = mu_t[mask]
            sigma_mask = sigma_t[mask]

            eps = 1e-5
            y_mask = y_mask + eps

            normal = torch.distributions.Normal(
                torch.zeros_like(mu_mask),
                torch.ones_like(sigma_mask)
            )

            omega = (torch.log(y_mask) - mu_mask) / sigma_mask

            ex_input = mu_mask + (sigma_mask ** 2) / 2
            ex_input = torch.clamp(ex_input, max=15)  # same as in original loss.py
            ex = 2 * torch.exp(ex_input)

            crps_vals = (
                y_mask * (2 * normal.cdf(omega) - 1.0)
                - ex * (normal.cdf(omega - sigma_mask)
                        + normal.cdf(sigma_mask / (2 ** 0.5))
                        - 1.0)
            )

            crps_t = crps_vals.mean().item()

            lead_crps_sum[t] += crps_t * lead_count
            lead_crps_count[t] += lead_count

    crps_mean = total_crps_sum / max(total_valid_count, 1)

    crps_at_leads = {}
    for t in lead_times:
        if lead_crps_count[t] > 0:
            crps_at_leads[t] = lead_crps_sum[t] / lead_crps_count[t]
        else:
            crps_at_leads[t] = float("nan")

    return crps_mean, crps_at_leads


@torch.no_grad()
def evaluate_crps(
    model,
    dataloader,
    tgt_denormalizer,
    device,
    edge_index=None,
    important_leads=(1, 24, 48, 96)
):
    model.eval()

    # all samples for full unbiased averaging
    all_y = []
    all_mu = []
    all_sigma = []

    # per-lead reporting
    lead_stats = {t: {"y": [], "mu": [], "sigma": []} for t in important_leads}

    for x_batch, y_batch in tqdm(dataloader, desc="Evaluating CRPS"):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)   # [B, L, S]

        # prediction
        pred = model(x_batch, edge_index=edge_index)
        pred = tgt_denormalizer(pred)
        y_batch = tgt_denormalizer(y_batch) #NOTE: no-op

        # extract μ, σ
        mu = pred.loc
        sigma = pred.scale

        # global mask
        mask = ~torch.isnan(y_batch)

        # flatten masked valid values
        all_y.append(y_batch[mask])
        all_mu.append(mu[mask])
        all_sigma.append(sigma[mask])

        # per-lead metrics
        for t in important_leads:
            idx = t - 1
            if idx >= y_batch.size(1):
                continue

            y_t = y_batch[:, idx, :]
            mu_t = mu[:, idx, :]
            sigma_t = sigma[:, idx, :]

            mask_t = ~torch.isnan(y_t)

            lead_stats[t]["y"].append(y_t[mask_t])
            lead_stats[t]["mu"].append(mu_t[mask_t])
            lead_stats[t]["sigma"].append(sigma_t[mask_t])

    # concatenate everything
    all_y = torch.cat(all_y)
    all_mu = torch.cat(all_mu)
    all_sigma = torch.cat(all_sigma)

    # compute CRPS vector for all samples at once
    eps = 1e-5
    all_y = all_y + eps

    normal = torch.distributions.Normal(
        torch.zeros_like(all_mu), torch.ones_like(all_sigma)
    )

    omega = (torch.log(all_y) - all_mu) / all_sigma
    ex_input = torch.clamp(all_mu + (all_sigma ** 2) / 2, max=15)
    ex = 2 * torch.exp(ex_input)

    crps_all = (
        all_y * (2 * normal.cdf(omega) - 1.0)
        - ex * (normal.cdf(omega - all_sigma)
                + normal.cdf(all_sigma / (2 ** 0.5))
                - 1.0)
    )

    crps_mean = crps_all.mean().item()

    # per-lead
    crps_at_leads = {}
    for t, d in lead_stats.items():
        if len(d["y"]) == 0:
            crps_at_leads[t] = float("nan")
            continue

        y_t = torch.cat(d["y"]) + eps
        mu_t = torch.cat(d["mu"])
        sigma_t = torch.cat(d["sigma"])

        normal_t = torch.distributions.Normal(
            torch.zeros_like(mu_t), torch.ones_like(sigma_t)
        )

        omega_t = (torch.log(y_t) - mu_t) / sigma_t
        ex_input_t = torch.clamp(mu_t + (sigma_t ** 2) / 2, max=15)
        ex_t = 2 * torch.exp(ex_input_t)

        crps_vec = (
            y_t * (2 * normal_t.cdf(omega_t) - 1.0)
            - ex_t * (
                normal_t.cdf(omega_t - sigma_t)
                + normal_t.cdf(sigma_t / (2 ** 0.5))
                - 1.0
            )
        )

        crps_at_leads[t] = crps_vec.mean().item()

    return crps_mean, crps_at_leads


def _init_metric_store():
    return {"crps_sum": 0.0, "mae_sum": 0.0, "count": 0}


def _update_metric_store(store, batch_crps, batch_mae, valid_count):
    store["crps_sum"] += batch_crps * valid_count
    store["mae_sum"]  += batch_mae  * valid_count
    store["count"]    += valid_count


def _finalize_metric_store(store):
    if store["count"] == 0:
        return None, None
    crps_mean = store["crps_sum"] / store["count"]
    mae_mean  = store["mae_sum"]  / store["count"]
    return crps_mean, mae_mean


def lognormal_params_from_mean_std(mean, std, eps=1e-6):
    """Return (mu, sigma) of a LogNormal given mean/std on the original scale."""
    mean = torch.clamp(mean, min=eps)
    std = torch.clamp(std, min=eps)
    variance = std ** 2
    log_term = torch.log1p(variance / (mean ** 2))
    mu = torch.log(mean) - 0.5 * log_term
    sigma = torch.sqrt(torch.clamp(log_term, min=eps))
    return mu, sigma


@torch.no_grad()
def evaluaate_crps_with_nwp_baseline(model,
                                    dataloader,
                                    device,
                                    test_crps_metric,        # instance of MaskedCRPSLogNormal()
                                    test_mae_metric,         # MaskedL1Loss() or similar masked MAE
                                    input_denormalizer,      # dm.test_dataset.input_denormalizer
                                    target_denormalizer,     # dm.test_dataset.target_denormalizer
                                    nwp_mean_idx,            # index in last dim of x_denorm for ensemble mean
                                    nwp_std_idx,             # index in last dim of x_denorm for ensemble std
                                    available_leads=(1, 24, 48, 96),
                                    edge_index=None,
):
    """
    Evaluate CRPS and MAE for:
      - the model (LogNormal output)
      - the NWP baseline (LogNormal fitted from ensemble mean/std)

    Returns:
    results : dict
        {
          "model": {
              "overall_crps": float,
              "overall_mae": float,
              "crps_per_lead": {lead_hour: float or None},
              "mae_per_lead":  {lead_hour: float or None},
          },
          "nwp": {
              "overall_crps": float,
              "overall_mae": float,
              "crps_per_lead": {lead_hour: float or None},
              "mae_per_lead":  {lead_hour: float or None},
          }
        }
    """
    
    model.eval()


    # Overall metrics
    model_metrics = _init_metric_store()
    nwp_metrics   = _init_metric_store()

    # Per-lead metrics
    model_lead_metrics = {lt: _init_metric_store() for lt in available_leads}
    nwp_lead_metrics   = {lt: _init_metric_store() for lt in available_leads}


    for x_batch, y_batch in tqdm(dataloader, desc="Testing"):
        x_batch = x_batch.to(device)          # [B, L, S, P]
        y_batch = y_batch.to(device)          # [B, L, S]

        # Count valid targets in this batch
        mask = ~torch.isnan(y_batch)
        valid_count = mask.sum().item()
        if valid_count == 0:
            continue


        # ---- MODEL ----
        if edge_index is not None:
            pred_dist_norm = model(x_batch, edge_index=edge_index)
        else:
            pred_dist_norm = model(x_batch)

        pred_dist = target_denormalizer(pred_dist_norm) #no-op
        y_den = target_denormalizer(y_batch) #no-op

        batch_crps_model = test_crps_metric(pred_dist, y_den).item()
        point_predictions = pred_dist.mean
        batch_mae_model = test_mae_metric(point_predictions, y_den).item()
        
        _update_metric_store(model_metrics, batch_crps_model, batch_mae_model, valid_count)


        # ---- NWP ----
        x_denorm = input_denormalizer(x_batch)  # [B, L, S, P]

        nwp_mean = x_denorm[..., nwp_mean_idx].unsqueeze(-1)  # [B, L, S, 1]
        nwp_std  = x_denorm[..., nwp_std_idx].unsqueeze(-1)   # [B, L, S, 1]

        nwp_mu, nwp_sigma = lognormal_params_from_mean_std(nwp_mean, nwp_std)
        nwp_dist = torch.distributions.LogNormal(nwp_mu, nwp_sigma)

        batch_crps_nwp = test_crps_metric(nwp_dist, y_den).item()
        batch_mae_nwp  = test_mae_metric(nwp_mean, y_den).item()

        _update_metric_store(nwp_metrics, batch_crps_nwp, batch_mae_nwp, valid_count)

        # ---- PER-LEAD METRICS ----
        for lead_hour in available_leads:
            y_lead = y_den[:, lead_hour:lead_hour+1, ...] # l_h:l_h+1 to keep dim: [B, 1, S]
            #if lead_hour == 96:
             #   print("y_lead: ", y_lead[0,0,:])
            lead_mask = ~torch.isnan(y_lead)
            lead_count = lead_mask.sum().item()
            if lead_count == 0:
                continue

            # Model at this lead
            pred_dist_lead = torch.distributions.LogNormal(
                pred_dist.loc[:, lead_hour:lead_hour+1, ...],
                pred_dist.scale[:, lead_hour:lead_hour+1, ...],
            )
            point_pred_lead = pred_dist_lead.mean

            lead_crps_model = test_crps_metric(pred_dist_lead, y_lead).item()
            lead_mae_model  = test_mae_metric(point_pred_lead, y_lead).item()

            _update_metric_store(
                model_lead_metrics[lead_hour],
                lead_crps_model,
                lead_mae_model,
                lead_count,
            )

            # NWP at this lead
            nwp_mean_lead = nwp_mean[:, lead_hour:lead_hour+1, ...]
            nwp_std_lead  = nwp_std[:, lead_hour:lead_hour+1, ...]
            mu_lead, sigma_lead = lognormal_params_from_mean_std(nwp_mean_lead, nwp_std_lead)
            nwp_dist_lead = torch.distributions.LogNormal(mu_lead, sigma_lead)

            lead_crps_nwp = test_crps_metric(nwp_dist_lead, y_lead).item()
            lead_mae_nwp  = test_mae_metric(nwp_mean_lead, y_lead).item()

            _update_metric_store(
                nwp_lead_metrics[lead_hour],
                lead_crps_nwp,
                lead_mae_nwp,
                lead_count,
            )


    model_crps_overall, model_mae_overall = _finalize_metric_store(model_metrics)
    nwp_crps_overall,   nwp_mae_overall   = _finalize_metric_store(nwp_metrics)

    model_crps_per_lead = {}
    model_mae_per_lead  = {}
    nwp_crps_per_lead   = {}
    nwp_mae_per_lead    = {}

    for lt, store in model_lead_metrics.items():
        m_crps, m_mae = _finalize_metric_store(store)
        model_crps_per_lead[lt] = m_crps
        model_mae_per_lead[lt]  = m_mae

    for lt, store in nwp_lead_metrics.items():
        n_crps, n_mae = _finalize_metric_store(store)
        nwp_crps_per_lead[lt] = n_crps
        nwp_mae_per_lead[lt]  = n_mae

    
    result_str = "Overall | CRPS: {:.4f} (NWP: {:.4f}) | MAE: {:.4f} (NWP: {:.4f})".format(
        model_crps_overall, nwp_crps_overall, model_mae_overall, nwp_mae_overall
    )
    for lt in available_leads:
        model_crps_lt = model_crps_per_lead[lt]
        model_mae_lt = model_mae_per_lead[lt]
        nwp_crps_lt = nwp_crps_per_lead[lt]
        nwp_mae_lt = nwp_mae_per_lead[lt]
        result_str += "\nt= {:>3}h | CRPS: {} (NWP: {}) | MAE: {} (NWP: {})".format(
            lt,
            f"{model_crps_lt:.4f}" if model_crps_lt is not None else "N/A",
            f"{nwp_crps_lt:.4f}" if nwp_crps_lt is not None else "N/A",
            f"{model_mae_lt:.4f}" if model_mae_lt is not None else "N/A",
            f"{nwp_mae_lt:.4f}" if nwp_mae_lt is not None else "N/A",
        )

    return result_str


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# NOTE uncomment to debug issues related to autograd
# torch.autograd.set_detect_anomaly(True)

OmegaConf.register_new_resolver("add_one", lambda x: int(x) + 1)
OmegaConf.register_new_resolver("oc.env_or", lambda var_name, default_value: os.getenv(var_name, default_value))

@hydra.main(version_base="1.1", config_path="./configs", config_name="default_training_conf")
def app(cfg: DictConfig):
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results")

    seed = cfg.get('seed', 42)
    set_seed(seed)
    
    if OmegaConf.select(cfg, "training.optim.kwargs.betas") is not None:
        cfg.training.optim.kwargs.betas = eval(cfg.training.optim.kwargs.betas)
    if 'hidden_sizes' in cfg.model.kwargs:
        cfg.model.kwargs.hidden_sizes = eval(cfg.model.kwargs.hidden_sizes) 
    
    print(cfg)
    predictor_names = list(cfg.dataset.predictors)
    lead_hours_limit = int(cfg.dataset.hours_leadtime)
    nwp_mean_idx = predictor_names.index(f"{cfg.nwp_model}:wind_speed_ensavg")
    nwp_std_idx = predictor_names.index(f"{cfg.nwp_model}:wind_speed_ensstd")

    ds = xr.open_dataset(cfg.dataset.features_pth)
    ds_targets = xr.open_dataset(cfg.dataset.targets_pth)

    dm = get_datamodule(ds=ds, 
                        ds_targets=ds_targets,
                        val_split=cfg.dataset.val_split,
                        test_start_date=cfg.dataset.test_start,
                        train_val_end_date=cfg.dataset.train_val_end,
                        lead_time_hours=cfg.dataset.hours_leadtime,
                        predictors=cfg.dataset.predictors, 
                        target_var=cfg.dataset.target_var,
                        return_graph=True, graph_kwargs=cfg.graph_kwargs)
    print(dm)

    adj_matrix = dm.adj_matrix
    edge_index, edge_weight = adj_to_edge_index(adj=torch.tensor(adj_matrix)) # NOTE not using w_ij for now
    
    
    batch_size = cfg.training.batch_size
    train_dataloader = DataLoader(dm.train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dm.val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dm.test_dataset, batch_size=batch_size, shuffle=False)
    
    assert dm.train_dataset.stations == dm.val_dataset.stations == dm.test_dataset.stations # sanity check 
    
    # MODEL LOAD
    model_kwargs = {'input_size': dm.train_dataset.f, 
                    'n_stations': dm.train_dataset.stations,
                    **cfg.model.kwargs} 
    model = get_model(model_type=cfg.model.type, **model_kwargs)
    
    epochs = cfg.training.epochs
    criterion = get_loss(cfg.training.loss)
    
    # Filter optimizer kwargs to only include valid parameters for the selected optimizer
    import inspect
    optimizer_class = getattr(torch.optim, cfg.training.optim.algo)
    valid_params = inspect.signature(optimizer_class.__init__).parameters.keys()
    filtered_kwargs = {k: v for k, v in cfg.training.optim.kwargs.items() if k in valid_params}
    optimizer = optimizer_class(model.parameters(), **filtered_kwargs)
            
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.training.scheduler.algo)(optimizer, **cfg.training.scheduler.kwargs)
    gradient_clip_value = cfg.training.gradient_clip_value
    gradient_accumulation_steps = cfg.training.get('gradient_accumulation_steps', 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model = model.to(device)
    edge_index = edge_index.to(device)

    # Set DagHub credentials BEFORE setting tracking URI
    if "dagshub.com" in cfg.logging.mlflow_tracking_uri:
        dagshub_token = os.getenv("MLFLOW_TRACKING_TOKEN") or os.getenv("DAGSHUB_TOKEN")
        dagshub_username = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USER_NAME")

    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=cfg.logging.experiment_id)
    
    run_name = OmegaConf.select(cfg, "logging.run_name")
    if run_name is None or run_name == "null":
        run_name = None  # MLflow will auto-generate a name

    if device_type == 'cpu':
        print("######### WARNING: GPU NOT IN USE ##########")
        torch.set_num_threads(16)
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("device_type", device_type)
        mlflow.log_param("optimizer", type(optimizer).__name__) 
        mlflow.log_param("criterion", type(criterion).__name__) 
        mlflow.log_param("num_params", sum(p.numel() for p in model.parameters() if p.requires_grad))
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(cfg_dict, 'training_config.json')
        mlflow.log_params(cfg_dict)
        
        # Early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        patience = 15
        val_loss_history = []    # for moving average
        window_size = 10
        min_delta = 1e-5         # minimum improvement to be considered significant
        best_model_state = None  # store best model weights
        
        total_iter = 0
        loss_str = ""
        epoch_pbar = tqdm(range(1, epochs+1), desc="Epochs")
        for epoch in epoch_pbar:
            model.train()
            total_loss = 0.0

            for batch_idx, (x_batch, y_batch) in tqdm(enumerate(train_dataloader), desc="Training Batches", total=len(train_dataloader), leave=False):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                if CUDA_MEM and batch_idx == 1:
                    torch.cuda.reset_peak_memory_stats()
                
                predictions = model(x_batch, edge_index=edge_index)  

                loss = criterion(predictions, y_batch)
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps
                loss.backward()  
                
                # update weights only at gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                    optimizer.step()
                    optimizer.zero_grad()
                
                if CUDA_MEM and batch_idx == 1:
                    print(f"Epoch {epoch}, Batch {batch_idx}")
                    print("x_batch shape:", x_batch.shape)
                    print("Peak memory allocated (MB):", torch.cuda.max_memory_allocated() / 1e6)
                    # print("Current memory allocated (MB):", torch.cuda.memory_allocated() / 1e6)
                
                total_loss += loss.item() * gradient_accumulation_steps  # Rescale for logging

                if total_iter % 25 == 0:
                    mlflow.log_metric("loss", loss.item() * gradient_accumulation_steps, step=total_iter)
                
                total_iter += 1

            avg_loss = total_loss / len(train_dataloader)
            
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            
            lr_scheduler.step(epoch=epoch-1)
            for group, lr in enumerate(lr_scheduler.get_last_lr()):
                mlflow.log_metric(f'lr_{group}', lr, step=epoch)
            
            # VALIDATION LOOP
            val_loss = 0
            val_loss_original_range = 0
            model.eval()
            with torch.no_grad():
                tgt_denormalizer = dm.val_dataset.target_denormalizer #NOTE: this is no-op --> should double check
                for batch_idx, (x_batch, y_batch) in tqdm(enumerate(val_dataloader), desc="Validation Batches", total=len(val_dataloader), leave=False):
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    predictions = model(x_batch, edge_index=edge_index)  

                    val_loss += criterion(predictions, y_batch).item()
                    val_loss_original_range += criterion(tgt_denormalizer(predictions), tgt_denormalizer(y_batch)).item()
                    # val_loss_original_range += criterion(predictions, y_batch).item()
                    
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_loss_or = val_loss_original_range / len(val_dataloader)

            epoch_pbar.set_postfix({
                'train_loss': f'{avg_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}'
            })
            loss_str += f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {avg_val_loss:.4f}\n"

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_loss_original_range", avg_val_loss_or, step=epoch)
            

            val_loss_history.append(avg_val_loss)            
            # best model
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Check if we should stop based on moving average trend
            if len(val_loss_history) >= window_size + patience:
                recent_avg = np.mean(val_loss_history[-window_size:])
                older_avg = np.mean(val_loss_history[-(window_size + patience):-patience])
                
                if recent_avg >= older_avg - min_delta:
                    print("\n" + "="*50)
                    print(f"Early stopping: no improvement in moving average")
                    print(f"  Recent {window_size} epochs avg: {recent_avg:.4f}")
                    print(f"  Previous {window_size} epochs avg: {older_avg:.4f}")
                    print("="*50 + "\n")
                    loss_str += f"Early stopping at epoch {epoch + 1} (moving avg plateau)\n"
                    break
            
            # fallback: stop if no single-epoch improvement for too long
            if epochs_without_improvement >= patience:
                print("\n" + "="*50)
                print(f"\nEarly stopping: no improvement for {patience} consecutive epochs")
                print("="*50 + "\n")
                loss_str += f"Early stopping at epoch {epoch + 1}\n"
                break
            
            # Optional plotting
            if epoch % 10 == 0:
                with torch.no_grad():
                    x_val_batch, y_val_batch = next(iter(val_dataloader))  
                    x_val_batch = x_val_batch.to(device)
                    y_val_batch = y_val_batch.to(device)
                    val_predictions = model(x_val_batch, edge_index=edge_index)  
                    
                    log_prediction_plots(x=x_val_batch, 
                                        y=y_val_batch, 
                                        pred_dist=val_predictions, 
                                        example_indices=[0,0,0,0], 
                                        stations=[1,2,3,4],
                                        epoch=epoch,
                                        input_denormalizer=dm.val_dataset.input_denormalizer)

        print("\n",   loss_str, "\n")
        
        # restore best model
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            print(f"Restored best model from validation (val_loss={best_val_loss:.4f})")
            loss_str += f"\nRestored best model (val_loss={best_val_loss:.4f})\n"
        
        mlflow.log_text(loss_str, "training_loss_history.txt")
        with open(os.path.join("results", "training_loss_history.txt"), "w") as f:
            f.write(loss_str)

        ## TEST LOOP
        result_str = evaluaate_crps_with_nwp_baseline(
            model=model,
            dataloader=test_dataloader,
            device=device,
            test_crps_metric = get_loss("MaskedCRPSLogNormal"),
            test_mae_metric = get_loss("MaskedL1Loss"),
            input_denormalizer=dm.test_dataset.input_denormalizer,
            target_denormalizer=dm.test_dataset.target_denormalizer, #no-op
            nwp_mean_idx=nwp_mean_idx,
            nwp_std_idx=nwp_std_idx,
            edge_index=edge_index,
        )
        print(result_str)
        mlflow.log_text(result_str, "test_results.txt")
        with open(os.path.join("results", "test_results.txt"), "w") as f:
            f.write(result_str)
        
        ## TALAGRAND DIAGRAM
        # Collect model outputs on TEST set
        mu_all, sigma_all, targets_all = collect_model_outputs(
            model=model,
            data_loader=test_dataloader,
            device=device,
            edge_index=edge_index,
        )

        lead_hours_to_plot = {1, 24, 48, 96}
        histograms = {}
        ml_text_talagrand = ""

        for lead_hour in tqdm(lead_hours_to_plot, desc="Computing Talagrand Histograms"):
            hist = rank_histogram_for_lead(
                mu_all=mu_all,
                sigma_all=sigma_all,
                targets_all=targets_all,
                device=device,
                n_samples=20,
            )

            histograms[lead_hour] = hist

            # print(f"Lead t={lead_hour}: total cases = {hist.sum().item()}, "f"hist = {hist.tolist()}")
            ml_text_talagrand += f"Lead t={lead_hour}:\n\ttotal cases = {hist.sum().item()},\n\thist = {hist.tolist()}\n"

            for lead_hour, hist in histograms.items():
                hist_np = hist.numpy().astype(float)
                hist_np /= hist_np.sum()

                ranks = np.arange(len(hist_np))  # 0..20

                plt.figure()
                plt.bar(ranks, hist_np, width=0.8, edgecolor='black')
                plt.xlabel("Rank")
                plt.ylabel("Relative frequency")
                plt.title(f"Rank histogram - lead t={lead_hour}", fontsize=14)
                plt.xticks(ranks)
                plt.grid(axis='y', alpha=0.3)
                
                # expected uniform line
                expected_freq = 1.0 / len(hist_np)
                plt.axhline(y=expected_freq, color='red', linestyle='--', linewidth=2, label=f'Expected (uniform): {expected_freq:.4f}')
                plt.legend()
                
                os.makedirs("results", exist_ok=True)
                plot_filename = os.path.join("results", f"talagrand_diagram_lead{lead_hour}.png")
                plt.savefig(plot_filename)
                mlflow.log_figure(plt.gcf(), f"talagrand_diagram_lead{lead_hour}.png")
                plt.close()
        
        mlflow.log_text(ml_text_talagrand, "talagrand_histograms.txt")
        # print(ml_text_talagrand)


if __name__ == '__main__':
    app()
    print("\n\nTraining completed.")