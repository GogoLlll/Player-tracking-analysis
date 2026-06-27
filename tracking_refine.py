from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

def world_kalman_smooth(
    frames: List[int],
    positions: List[Optional[Tuple[float, float]]],
    fps: float = 30.0,
    q_pos: float = 0.4,
    q_vel: float = 1.5,
    r_meas: float = 1.0,
) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, Tuple[float, float]]]:
    obs = [(int(f), (float(p[0]), float(p[1])))
           for f, p in zip(frames, positions) if p is not None]
    if len(obs) < 2:
        sm = {int(f): (float(p[0]), float(p[1]))
              for f, p in zip(frames, positions) if p is not None}
        return sm, {f: (0.0, 0.0) for f in sm}

    f0, f1 = obs[0][0], obs[-1][0]
    all_frames = list(range(f0, f1 + 1))
    meas = {f: np.asarray(p, dtype=np.float64) for f, p in obs}

    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    R = np.eye(2) * r_meas

    x = np.array([meas[f0][0], meas[f0][1], 0.0, 0.0], dtype=np.float64)
    P = np.eye(4) * 10.0

    xs_pred, Ps_pred, xs_filt, Ps_filt, Fs = [], [], [], [], []
    prev_f = f0
    for fi in all_frames:
        dt = float(fi - prev_f)
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float64)
        Q = np.diag([q_pos * max(dt, 1), q_pos * max(dt, 1),
                     q_vel * max(dt, 1), q_vel * max(dt, 1)])

        x = F @ x
        P = F @ P @ F.T + Q
        xs_pred.append(x.copy()); Ps_pred.append(P.copy()); Fs.append(F.copy())

        if fi in meas:
            z = meas[fi]
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (np.eye(4) - K @ H) @ P
        xs_filt.append(x.copy()); Ps_filt.append(P.copy())
        prev_f = fi

    n = len(all_frames)
    xs_sm = [None] * n
    xs_sm[-1] = xs_filt[-1]
    Ps_sm = [None] * n
    Ps_sm[-1] = Ps_filt[-1]
    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        Pp = Ps_pred[k + 1]
        try:
            G = Ps_filt[k] @ F.T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            G = np.zeros((4, 4))
        xs_sm[k] = xs_filt[k] + G @ (xs_sm[k + 1] - xs_pred[k + 1])
        Ps_sm[k] = Ps_filt[k] + G @ (Ps_sm[k + 1] - Pp) @ G.T

    sm_pos, sm_vel = {}, {}
    for k, fi in enumerate(all_frames):
        s = xs_sm[k]
        sm_pos[fi] = (float(s[0]), float(s[1]))
        sm_vel[fi] = (float(s[2] * fps), float(s[3] * fps))
    return sm_pos, sm_vel

def gsi_smooth(
    frames: List[int],
    values: np.ndarray,
    tau: float = 12.0,
    noise: float = 4.0,
    max_fill: Optional[int] = 25,
) -> Dict[int, np.ndarray]:
    fr = np.asarray(frames, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    n = len(fr)
    if n == 0:
        return {}
    if n == 1:
        return {int(fr[0]): vals[0]}

    q = np.arange(int(fr[0]), int(fr[-1]) + 1).astype(np.float64)

    def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a[:, None] - b[None, :]
        return np.exp(-(d * d) / (2.0 * tau * tau))

    mean_v = vals.mean(axis=0)
    vals_c = vals - mean_v
    K = rbf(fr, fr) + noise * np.eye(n)
    try:
        alpha = np.linalg.solve(K, vals_c)
    except np.linalg.LinAlgError:
        return {int(f): v for f, v in zip(frames, values)}
    Ks = rbf(q, fr)
    mean = Ks @ alpha + mean_v

    out: Dict[int, np.ndarray] = {}
    obs_frames = fr.astype(int)
    for i, fq in enumerate(q.astype(int)):
        if max_fill is not None:
            nearest = int(np.min(np.abs(obs_frames - fq)))
            if nearest > max_fill:
                continue
        out[int(fq)] = mean[i]
    return out
