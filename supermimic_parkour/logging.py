"""CSV and TensorBoard metrics for the shared model and its selected skills."""

import csv
from pathlib import Path

LOG_FIELDS = (
    "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,"
    "pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,"
    "lr,ep_return,metrics_version,wm_term_bce,num_updates"
).split(",")


def open_csv_log(path, fields, missing="nan"):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open() as stream:
            if next(csv.reader(stream)) != fields:
                raise ValueError(f"CSV columns differ: {path}; use a new log directory")
    stream = path.open("a")
    if stream.tell() == 0:
        csv.writer(stream).writerow(fields)
        stream.flush()
    return stream


def write_scalars(writer, row):
    for key, value in row.items():
        if key not in ("iteration", "metrics_version"):
            writer.add_scalar(key, value, int(row["iteration"]))


def create_writer(log_dir, *, purge_step=None):
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(str(log_dir), purge_step=purge_step, flush_secs=10)
