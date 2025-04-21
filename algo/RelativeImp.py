import numpy as np
import pandas as pd
import scipy.linalg as SLA
import tqdm

import jax
import jax. numpy as jnp 
import polars as pl

SVD = jax.numpy.linalg.svd
INV = jax.numpy.linalg.inv

from typing import List, Union, Tuple, Optional


class RelativeImp:
    """
    Class for computing relative importance of input features on an outcome.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing predictors and outcome.
    outcomeName : str
        Column name of the outcome variable.
    driverNames : list[str]
        Column names of predictor variables.
    SEED : int, optional
        Random seed for reproducibility (default 24).
    """

    def __init__(self, df: pd.DataFrame, outcomeName: str,
                 driverNames: List[str], SEED: int = 24):
        """
        Initialize the RelativeImp instance.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset with predictors and outcome.
        outcomeName : str
            Outcome column name.
        driverNames : list[str]
            Predictor column names.
        SEED : int, default 24
            Seed for NumPy’s PRNG.
        """
        self.df = df
        self.outcomeName = outcomeName
        self.driverNames = driverNames
        self.df_results = None
        self.SEED = SEED

        # Sanity‑check: outcome column must not be in driver list
        assert outcomeName not in driverNames

        self.X = df[driverNames]
        self.Y = df[outcomeName]

    def draw_bs_samples(self, indices: np.ndarray, iter: int, perc: float):
        """
        Draw bootstrap samples (with replacement) of row indices.

        Parameters
        ----------
        indices : np.ndarray
            Array of row indices to sample from.
        iter : int
            Number of bootstrap iterations.
        perc : float
            Fraction of the data to draw each iteration (0–1).

        Returns
        -------
        np.ndarray
            2‑D array of shape (iter, perc·len(indices)) with sampled indices.
        """
        np.random.seed(self.SEED)
        print(f"{int(len(indices)*perc)} samples to be sampled "
              f"with replacement out of n = {len(indices)}")
        idx = np.random.choice(indices,
                               size=(iter, int(len(indices)*perc)))
        # idx = np.array([np.random.choice(indices,
        #          size=int(len(indices)*perc)) for _ in range(iter)])
        return np.array(idx)

    def calculate_svd(self, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Johnson’s relative‑weight calculation via SVD.

        Parameters
        ----------
        x : np.ndarray
            Predictor matrix.
        y : np.ndarray
            Outcome vector.

        Returns
        -------
        pd.DataFrame
            Raw and normalised relative importances plus coefficient signs.
        """
        u, s, vt = SVD(x, full_matrices=False)
        Z = u @ vt
        intermediary = INV(Z.T @ Z) @ Z.T
        beta = intermediary @ y
        lambdaa = intermediary @ x
        eps = (lambdaa**2) @ (beta**2)
        eps = eps.squeeze()

        assert np.allclose((eps * 100 / eps.sum()).sum(), 100.0)
        assert eps.size == len(self.driverNames)

        norm_eps = eps * 1e2 / eps.sum()

        return pd.DataFrame(
            data={
                "driver": self.driverNames,
                "rawRelaImpt": eps,
                "normRelaImpt": norm_eps,
                "sign": np.sign(beta).squeeze(),
            }
        )

    def run_bootstrap_SVD(self, bootstrap: bool = False, n_reps: int = 1000,
                          perc: float = 1.0, alpha: float = 0.05,
                          rscore: bool = False):
        """
        Compute relative weights for the full sample and (optionally)
        bootstrap replicates.

        Parameters
        ----------
        bootstrap : bool, default False
            If True, perform bootstrap resampling.
        n_reps : int, default 1000
            Number of bootstrap replicates.
        perc : float, default 1.0
            Fraction of rows drawn each replicate.
        alpha : float, default 0.05
            Significance level for percentile intervals.
        rscore : bool, default False
            If True, print model R² via self.r2 (if implemented).

        Returns
        -------
        pd.DataFrame
            Relative‑importance summary table.
        tuple (pd.DataFrame, np.ndarray), if `bootstrap=True`
            Summary table and bootstrap array.
        """
        total_rwa = self.calculate_svd(self.X.to_numpy(),
                                       self.Y.to_numpy())
        self.df_results = total_rwa

        if rscore:
            print(f"r2 is {self.r2(df, xcol=self.driverNames,ycol=self.outcomeName)}")

        if bootstrap:
            print("entering bootstrap")
            idx = self.draw_bs_samples(indices=np.arange(len(self.df)),
                                       iter=n_reps, perc=perc)

            rwa_bootstraps = []
            for iteration_number in tqdm.tqdm(range(n_reps)):
                x, y = (self.X[idx[iteration_number]].to_numpy(),
                        self.Y[idx[iteration_number]].to_numpy())
                individual_rwa = self.calculate_svd(x, y)
                eps = individual_rwa["normRelaImpt"].values
                try:
                    assert np.allclose(eps.sum(), 100.0)
                    assert eps.size == len(self.driverNames)
                except AssertionError as exc:
                    print("Bootstrap warning:", exc)
                rwa_bootstraps.append(eps.squeeze())

            rwa_bootstraps = np.array(rwa_bootstraps)
            try:
                assert rwa_bootstraps.shape == (n_reps, len(self.driverNames))
            except AssertionError as exc:
                print("Shape mismatch:", rwa_bootstraps.shape)

            boot_df_results = pd.DataFrame(
                data={
                    "driver": self.driverNames,
                    "normRelaImpt": total_rwa["normRelaImpt"], # you can use this value sans bootstraps.
                    "normRelaImpt_mean": rwa_bootstraps.mean(axis=0),
                    "normRelaImpt_std": rwa_bootstraps.std(axis=0),
                    "normRelaImpt_median": np.median(rwa_bootstraps, axis=0),
                    "normRelaImpt_high": np.percentile(
                        rwa_bootstraps, q=int((1 - alpha/2) * 100), axis=0),
                    "normRelaImpt_low": np.percentile(
                        rwa_bootstraps, q=int(alpha/2 * 100), axis=0),
                    "normRelaImpt_75": np.percentile(rwa_bootstraps, 75, axis=0),
                    "normRelaImpt_25": np.percentile(rwa_bootstraps, 25, axis=0),
                    "normRelaImpt_min": np.nanmin(rwa_bootstraps, axis=0),
                    "normRelaImpt_max": np.nanmax(rwa_bootstraps, axis=0),
                    "sign_total": total_rwa["sign"],
                }
            )

            return boot_df_results, rwa_bootstraps

        return self.df_results

    