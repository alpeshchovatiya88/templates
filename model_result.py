import pandas as pd
import sys
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from IPython.display import display, Markdown

sys.path.append(str(Path.cwd().parent))
warnings.filterwarnings("ignore")

pd.set_option("display.max_colwidth", 20)
pd.set_option("display.max_columns", None)


class ModelResult:
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "target",
        smi2_score_col: str = "SMI2_CTR",
        smi3_funded_score_col: str = "smi3_pred_prob_funded",
        smi3_all_score_col: str = "smi3_pred_prob_all",
        category_col: str = "category",
        features_to_avg=None,
        model=None,
        default_threshold: float = 0.5,
        plot_confusion: bool = True,
        plot_deciles: bool = True,
        plot_category_distributions: bool = True,
        show_score_chart: bool = True,
        show_lift_chart: bool = True,
        show_gain_chart: bool = True,
        show_category_capture_curves: bool = True,
        save_pred_filename: str = "df_scored.csv",
        output_subdir: str = "all",
        show_risk_ranking_df: bool = True,
        smi2_lr_col: str = "SMI2_LR",
        smi3_funded_lr_col: str = "SMI3_LR_funded",
        smi3_all_lr_col: str = "SMI3_LR_all",
        risk_ranking_basis: str = "lr",
        decile_q: int = 10,
        risk_ranking_q: int = 20,
    ):
        self.data = data
        self.target_col = target_col
        self.smi2_score_col = smi2_score_col
        self.smi3_funded_score_col = smi3_funded_score_col
        self.smi3_all_score_col = smi3_all_score_col

        self.smi2_lr_col = smi2_lr_col
        self.smi3_funded_lr_col = smi3_funded_lr_col
        self.smi3_all_lr_col = smi3_all_lr_col

        self.category_col = category_col
        self.features_to_avg = features_to_avg or [
            "CRD_SCORE",
            "ACTLTV",
            "NUM_OF_BOR",
            "TDSR",
            "GDSR",
            "PRK_PROPERTY_AGE",
        ]
        self.model = model
        self.default_threshold = default_threshold

        self.plot_confusion = plot_confusion
        self.plot_deciles = plot_deciles
        self.plot_category_distributions = plot_category_distributions
        self.show_score_chart = show_score_chart
        self.show_lift_chart = show_lift_chart
        self.show_gain_chart = show_gain_chart
        self.show_category_capture_curves = show_category_capture_curves

        self.save_pred_filename = save_pred_filename
        self.output_subdir = output_subdir
        self.show_risk_ranking_df = show_risk_ranking_df

        self.risk_ranking_basis = (risk_ranking_basis or "prob").lower()
        if self.risk_ranking_basis not in {"prob", "ctr", "lr"}:
            raise ValueError("risk_ranking_basis must be one of: 'prob'/'ctr' or 'lr'")

        self.decile_q = decile_q
        self.risk_ranking_q = risk_ranking_q

        self.OUT = Path("artifacts") / self.output_subdir
        self.OUT.mkdir(parents=True, exist_ok=True)

        self.df_scored = None

    # ---------- core ----------
    def _get_X(self) -> pd.DataFrame:
        return self.data.drop(columns=[self.target_col], errors="ignore")

    def _get_probabilities(self) -> np.ndarray:
        if self.model is not None:
            X = self._get_X()
            return self.model.predict_proba(X)[:, 1]
        return self.data[self.smi3_funded_score_col].to_numpy()

    def add_or_update_predictions(self) -> pd.DataFrame:
        df = self.data.copy()
        if self.model is not None:
            df[self.smi3_all_score_col] = self._get_probabilities()
        return df

    def _get_predictions_hard(self) -> np.ndarray:
        return (self._get_probabilities() >= self.default_threshold).astype(int)

    # ---------- helpers ----------
    @staticmethod
    def _rank_qcut(
        s: pd.Series,
        q: int,
        labels: list[str],
        ascending: bool = True,
    ) -> pd.Categorical:
        r = s.rank(method="first", ascending=ascending)
        return pd.qcut(r, q=q, labels=labels, duplicates="drop")

    @staticmethod
    def _sort_decile_labels(
        df: pd.DataFrame,
        decile_col: str = "model_decile",
    ) -> pd.DataFrame:
        return df.sort_values(
            by=decile_col,
            key=lambda s: s.astype(str).str.extract(r"(\d+)").astype(int)[0],
            ascending=True,
        ).reset_index(drop=True)

    def _get_crd_score_col(self, model_name: str) -> str:
        return "CRD_SCORE_SMI2" if model_name == "SMI2" else "CRD_SCORE"

    # ---------- deciles ----------
    def add_deciles(
        self,
        df: pd.DataFrame | None = None,
        q: int | None = None,
        prefix: str = "",
    ) -> pd.DataFrame:
        df = self.add_or_update_predictions() if df is None else df.copy()
        q = q or self.decile_q
        labels = [f"D{i}" for i in range(1, q + 1)]

        # -------------------------
        # SMI3 Funded Prob deciles
        # -------------------------
        required_prob_cols = [self.smi3_funded_score_col, "CRD_SCORE", "ACTLTV"]
        missing_prob_cols = [c for c in required_prob_cols if c not in df.columns]
        if missing_prob_cols:
            raise KeyError(
                f"Missing required SMI3 Funded probability columns: {missing_prob_cols}"
            )

        funded_sorted = df.sort_values(
            by=[self.smi3_funded_score_col, "CRD_SCORE", "ACTLTV"],
            ascending=[False, True, False],
        ).copy()

        funded_sorted[f"{prefix}smi3_funded_decile"] = self._rank_qcut(
            funded_sorted[self.smi3_funded_score_col],
            q=q,
            labels=labels,
            ascending=False,
        )

        df[f"{prefix}smi3_funded_decile"] = funded_sorted[
            f"{prefix}smi3_funded_decile"
        ]

        # -------------------------
        # SMI3 Funded LR deciles
        # -------------------------
        if self.smi3_funded_lr_col in df.columns:
            funded_lr_sorted = df.sort_values(
                by=[self.smi3_funded_lr_col, "CRD_SCORE", "ACTLTV"],
                ascending=[False, True, False],
            ).copy()

            funded_lr_sorted[f"{prefix}smi3_funded_lr_decile"] = self._rank_qcut(
                funded_lr_sorted[self.smi3_funded_lr_col],
                q=q,
                labels=labels,
                ascending=False,
            )

            df[f"{prefix}smi3_funded_lr_decile"] = funded_lr_sorted[
                f"{prefix}smi3_funded_lr_decile"
            ]
        else:
            df[f"{prefix}smi3_funded_lr_decile"] = pd.Series(
                index=df.index, dtype="object"
            )

        return df

    def save_predicted_output(self) -> Path | None:
        df_out = self.add_deciles(q=self.decile_q)
        df_out = self.add_deciles(df=df_out, q=self.risk_ranking_q, prefix="rr_")
        self.df_scored = df_out
        out_file = self.OUT / self.save_pred_filename
        df_out.to_csv(out_file, index=False)
        return out_file

    # ---------- specs ----------
    def _risk_ranking_spec(self, basis: str | None = None) -> dict[str, str]:
        b = (basis or self.risk_ranking_basis).lower()
        if b == "ctr":
            b = "prob"

        if b == "prob":
            return {
                "basis_name": "prob_ctr",
                "smi3_funded_score_col": self.smi3_funded_score_col,
                "smi3_funded_decile_col": "rr_smi3_funded_decile",
            }

        if b == "lr":
            return {
                "basis_name": "lr",
                "smi3_funded_score_col": self.smi3_funded_lr_col,
                "smi3_funded_decile_col": "rr_smi3_funded_lr_decile",
            }

        raise ValueError("basis must be one of: 'prob'/'ctr' or 'lr'")

    def _decile_plot_spec(self, decile_kind: str = "prob") -> list[dict[str, str]]:
        if decile_kind == "prob":
            return [
                {
                    "name": "SMI3_Funded",
                    "decile_col": "smi3_funded_decile",
                    "value_col": self.smi3_funded_score_col,
                }
            ]

        if decile_kind == "lr":
            return [
                {
                    "name": "SMI3_Funded",
                    "decile_col": "smi3_funded_lr_decile",
                    "value_col": self.smi3_funded_lr_col,
                }
            ]

        raise ValueError("decile_kind must be 'prob' or 'lr'")

    # ---------- risk ranking ----------
    def risk_ranking(
        self,
        basis: str | None = None,
    ) -> pd.DataFrame:
        df = self.add_deciles(q=self.decile_q)
        df = self.add_deciles(df=df, q=self.risk_ranking_q, prefix="rr_")
        spec = self._risk_ranking_spec(basis=basis)

        rr_smi3_funded_df = self._risk_ranking_one(
            df=df,
            decile_col=spec["smi3_funded_decile_col"],
            score_col=spec["smi3_funded_score_col"],
            model_name="SMI3_Funded",
            basis=basis,
        )

        return rr_smi3_funded_df

    def _risk_ranking_one(
        self,
        df: pd.DataFrame,
        decile_col: str,
        score_col: str,
        model_name: str,
        basis: str | None = None,
    ) -> pd.DataFrame:
        d = df.copy().rename(columns={decile_col: "model_decile"})
        crd_score_col = self._get_crd_score_col(model_name)

        avg_feature_map = {}
        for c in self.features_to_avg:
            if c == "CRD_SCORE":
                if crd_score_col in d.columns:
                    avg_feature_map[c] = crd_score_col
            elif c in d.columns:
                avg_feature_map[c] = c

        agg_dict = {
            **{src_col: "mean" for src_col in avg_feature_map.values()},
            score_col: ["sum"],
            self.target_col: ["sum"] if self.target_col in d.columns else ["size"],
        }

        summary = d.groupby("model_decile", observed=False).agg(agg_dict)

        basis_name = self._risk_ranking_spec(basis=basis)["basis_name"]

        cols = []
        reverse_avg_map = {v: k for k, v in avg_feature_map.items()}

        for c in summary.columns:
            col_name, _ = c

            if col_name in reverse_avg_map:
                cols.append(reverse_avg_map[col_name])
            elif c == (score_col, "sum"):
                cols.append("sum_prob" if basis_name == "prob_ctr" else "sum_lr")
            else:
                cols.append("act_sum" if self.target_col in d.columns else "n")

        summary.columns = cols
        summary = summary.reset_index()

        cat = self.category_col
        categories = ["Delq_Only", "Delq_Claim", "Declined", "No_Event"]
        bad_categories = ["Delq_Only", "Delq_Claim", "Declined"]

        def _extra(x: pd.DataFrame) -> pd.Series:
            out = {}
            yellow_red = x["POD_DECISION_IND_OVERALL"].astype(str).str.lower().isin(["yellow", "red"])
            declined = x["STATUS"].astype(str).str.lower().eq("declined")
            funded = x["STATUS"].astype(str).str.lower().eq("funded")
            collateral_fail = x["RAS_OVERALL_COLLATERAL"].astype(str).str.upper().eq("F")
            collateral_pass = x["RAS_OVERALL_COLLATERAL"].astype(str).str.upper().eq("P")
            declined_filtered = declined & yellow_red & collateral_pass
            for k in categories:
                if k == "Declined":
                    m = declined_filtered
                    out["n_overall_Declined"] = int(x[cat].eq("Declined").sum())
                else:
                    m = x[cat].eq(k)
                out[f"avg_{k}"] = x.loc[m, score_col].mean()
                out[f"sum_{k}"] = x.loc[m, score_col].sum()
                out[f"n_{k}"] = int(m.sum())
            if self.target_col in x.columns:
                for k in categories:
                    if k == "Declined":
                        m = declined_filtered
                        out[f"act_{k}"] = x.loc[m, self.target_col].sum()
                    else:
                        out[f"act_{k}"] = x.loc[x[cat].eq(k), self.target_col].sum()
            out["n_No_Event"] = x[cat].eq("No_Event").sum()
            out["n_total"] = len(x)
            if crd_score_col in x.columns:
                out["pct_crd_score_ge_700"] = x[crd_score_col].ge(700).mean() * 100
            else:
                out["pct_crd_score_ge_700"] = np.nan
            if "ACTLTV" in x.columns:
                out["pct_actltv_ge_90"] = x["ACTLTV"].ge(90).mean() * 100
            else:
                out["pct_actltv_ge_90"] = np.nan
            if "TDSR" in x.columns:
                out["pct_tdsr_ge_38"] = x["TDSR"].ge(38).mean() * 100
            else:
                out["pct_tdsr_ge_38"] = np.nan
            if "GDSR" in x.columns:
                out["pct_gdsr_ge_33"] = x["GDSR"].ge(33).mean() * 100
            else:
                out["pct_gdsr_ge_33"] = np.nan
            out["n_decline_yellow_red_collateral_fail"] = int((yellow_red & declined & collateral_fail).sum())
            out["n_yellow_red_funded"] = int((yellow_red & funded).sum())
            out["n_yellow_red_not_declined_not_funded"] = int((yellow_red & ~declined & ~funded).sum())
            out["n_overall_bad"] = int(x[cat].isin(bad_categories).sum())
            out["n_Bad"] = int(
                x[cat].isin(["Delq_Only", "Delq_Claim"]).sum()
                + declined_filtered.sum()
            )
            return pd.Series(out)

        extra = d.groupby("model_decile", observed=False).apply(_extra).reset_index()
        out = summary.merge(extra, on="model_decile", how="left")
        out["avg_Bad"] = (
            out.get("avg_Delq_Only", 0).fillna(0)
            + out.get("avg_Delq_Claim", 0).fillna(0)
            + out.get("avg_Declined", 0).fillna(0)
        ) / 3
        out["sum_Bad"] = (
            out.get("sum_Delq_Only", 0).fillna(0)
            + out.get("sum_Delq_Claim", 0).fillna(0)
            + out.get("sum_Declined", 0).fillna(0)
        )
        if self.target_col in d.columns:
            out["act_Bad"] = (
                out.get("act_Delq_Only", 0).fillna(0)
                + out.get("act_Delq_Claim", 0).fillna(0)
                + out.get("act_Declined", 0).fillna(0)
            )
        else:
            out["act_Bad"] = np.nan
        act_cols = [f"act_{k}" for k in categories] + ["act_Bad"]
        for c in act_cols:
            if c in out.columns:
                total = out[c].sum()
                out[f"pct_{c}"] = out[c] / total * 100 if total > 0 else np.nan
        count_cols = [
            "n_Delq_Only",
            "n_Delq_Claim",
            "n_Declined",
            "n_overall_Declined",
            "n_Bad",
            "n_overall_bad",
            "n_No_Event",
            "n_decline_yellow_red_collateral_fail",
            "n_yellow_red_funded",
            "n_yellow_red_not_declined_not_funded",
        ]
        for c in count_cols:
            if c in out.columns:
                total = out[c].sum()
                out[f"pct_{c}"] = out[c] / total * 100 if total > 0 else np.nan
        out = self._sort_decile_labels(out, decile_col="model_decile")
        cum_base_cols = ["Delq_Only", "Delq_Claim", "Declined", "Bad", "No_Event"]

        for k in cum_base_cols:
            n_col = f"n_{k}"
            if n_col in out.columns:
                out[f"cum_{n_col}"] = out[n_col].cumsum()
                total = out[n_col].sum()
                out[f"cum_pct_{k}"] = out[f"cum_{n_col}"] / total * 100 if total > 0 else np.nan
        if "n_overall_bad" in out.columns:
            out["cum_n_overall_bad"] = out["n_overall_bad"].cumsum()
            total = out["n_overall_bad"].sum()
            out["cum_pct_overall_bad"] = out["cum_n_overall_bad"] / total * 100 if total > 0 else np.nan
        if "n_overall_Declined" in out.columns:
            out["cum_n_overall_Declined"] = out["n_overall_Declined"].cumsum()
            total = out["n_overall_Declined"].sum()
            out["cum_pct_overall_Declined"] = out["cum_n_overall_Declined"] / total * 100 if total > 0 else np.nan
        out["cum_population"] = out["n_total"].cumsum()
        total_pop = out["n_total"].sum()
        out["cum_pct_population"] = out["cum_population"] / total_pop * 100 if total_pop > 0 else np.nan
        pop_share = out["cum_pct_population"] / 100
        bad_share = out["cum_pct_Bad"] / 100
        out["lift_Bad"] = np.where(pop_share > 0, bad_share / pop_share, np.nan)
        avg_cols = [f"avg_{k}" for k in categories] + ["avg_Bad"]
        sum_cols = [f"sum_{k}" for k in categories] + ["sum_Bad"] + (
            ["sum_prob"] if "sum_prob" in out.columns else ["sum_lr"]
        )

        base_cols = ["model_decile"] + [c for c in self.features_to_avg if c in out.columns]
        tail_cols = [c for c in ["act_sum", "n"] if c in out.columns]
        pct_count_cols = [f"pct_{c}" for c in count_cols if f"pct_{c}" in out.columns]
        cum_pct_cols = [f"cum_pct_{k}" for k in cum_base_cols if f"cum_pct_{k}" in out.columns] + (
            ["cum_pct_overall_bad"] if "cum_pct_overall_bad" in out.columns else []
        ) + (
            ["cum_pct_overall_Declined"] if "cum_pct_overall_Declined" in out.columns else []
        )
        ordered = (
            base_cols
            + [
                "n_total",
                "n_Delq_Only",
                "n_Delq_Claim",
                "n_Declined",
                "n_overall_Declined",
                "n_Bad",
                "n_overall_bad",
                "n_No_Event",
                "n_decline_yellow_red_collateral_fail",
                "n_yellow_red_funded",
                "n_yellow_red_not_declined_not_funded",
            ]
            + ["pct_crd_score_ge_700", "pct_actltv_ge_90", "pct_tdsr_ge_38", "pct_gdsr_ge_33"]
            + avg_cols
            + sum_cols
            + act_cols
            + tail_cols
            + pct_count_cols
            + cum_pct_cols
            + ["cum_population", "cum_pct_population", "lift_Bad"]
        )
        ordered = [c for c in ordered if c in out.columns]
        return out[ordered]

    def d1_d2_detailed_summary(
        self,
        smi2_decile="rr_smi2_lr_decile",
        smi3_all_decile="rr_smi3_all_lr_decile",
        smi3_funded_decile="rr_smi3_funded_lr_decile",
    ):
        if not hasattr(self, "df_scored"):
            raise ValueError("Run save_predicted_output() first")

        df = self.df_scored
        top = ["D1", "D2"]

        def summarize(d, score_col):
            if len(d) == 0:
                return 0, np.nan, np.nan, np.nan
            return (
                len(d),
                d[score_col].mean() if score_col in d.columns else np.nan,
                d["TDSR"].mean() if "TDSR" in d.columns else np.nan,
                d["GDSR"].mean() if "GDSR" in d.columns else np.nan,
                d["ACTLTV"].mean() if "ACTLTV" in d.columns else np.nan,
            )

        def build_views(model_name, model_decile):
            score_col = "CRD_SCORE_SMI2" if model_name == "SMI2" else "CRD_SCORE"

            smi2_top = df[df[smi2_decile].isin(top)]
            model_top = df[df[model_decile].isin(top)]

            # match / non-match relative to SMI2 top bucket
            match = smi2_top[smi2_top[model_decile].isin(top)]
            non_match = smi2_top[~smi2_top[model_decile].isin(top)]

            pct_match = len(match) / len(smi2_top) * 100 if len(smi2_top) > 0 else np.nan
            match_avg_score = match[score_col].mean() if score_col in match.columns else np.nan
            non_match_avg_score = non_match[score_col].mean() if score_col in non_match.columns else np.nan
            non_match_tdsr = non_match["TDSR"].mean() if "TDSR" in non_match.columns else np.nan
            non_match_gdsr = non_match["GDSR"].mean() if "GDSR" in non_match.columns else np.nan
            non_match_actltv = non_match["ACTLTV"].mean() if "ACTLTV" in non_match.columns else np.nan

            # Left D1/D2: in SMI2 but not in model
            left = non_match

            # new D1/D2: in model but not in SMI2
            new = model_top[~model_top.index.isin(smi2_top.index)]

            l_cnt, l_score, l_tdsr, l_gdsr, l_ltv = summarize(left, score_col)
            n_cnt, n_score, n_tdsr, n_gdsr, n_ltv = summarize(new, score_col)

            return {
                "Model": model_name,
                "% MATCH to SMI2 (D1&D2)": pct_match,
                "MATCH Avg Score": match_avg_score,
                "NON_MATCH Avg Score": non_match_avg_score,
                "NON_MATCH TDSR": non_match_tdsr,
                "NON_MATCH GDSR": non_match_gdsr,
                "NON_MATCH ACTLTV": non_match_actltv,
                "Left_D1D2_Count": l_cnt,
                "Left_D1D2_Avg_Score": l_score,
                "Left_D1D2_TDSR": l_tdsr,
                "Left_D1D2_GDSR": l_gdsr,
                "Left_D1D2_ACTLTV": l_ltv,
                "New_D1D2_Count": n_cnt,
                "New_D1D2_Avg_Score": n_score,
                "New_D1D2_TDSR": n_tdsr,
                "New_D1D2_GDSR": n_gdsr,
                "New_D1D2_ACTLTV": n_ltv,
            }

        smi2_top = df[df[smi2_decile].isin(top)]

        rows = [
            {
                "Model": "SMI2",
                "% MATCH to SMI2 (D1&D2)": 100.0,
                "MATCH Avg Score": smi2_top["CRD_SCORE_SMI2"].mean() if "CRD_SCORE_SMI2" in smi2_top.columns else np.nan,
                "NON_MATCH Avg Score": np.nan,
                "NON_MATCH TDSR": np.nan,
                "NON_MATCH GDSR": np.nan,
                "NON_MATCH ACTLTV": np.nan,
                "Left_D1D2_Count": 0,
                "Left_D1D2_Avg_Score": np.nan,
                "Left_D1D2_TDSR": np.nan,
                "Left_D1D2_GDSR": np.nan,
                "Left_D1D2_ACTLTV": np.nan,
                "New_D1D2_Count": 0,
                "New_D1D2_Avg_Score": np.nan,
                "New_D1D2_TDSR": np.nan,
                "New_D1D2_GDSR": np.nan,
                "New_D1D2_ACTLTV": np.nan,
            },
            build_views("SMI3_All", smi3_all_decile),
            build_views("SMI3_Funded", smi3_funded_decile),
        ]

        return pd.DataFrame(rows)

    # ---------- decile ranges / confusions ----------
    def get_decile_ranges(self, decile_kind: str = "prob") -> pd.DataFrame:
        df = self.add_deciles(q=self.decile_q)
        spec = self._decile_plot_spec(decile_kind=decile_kind)[0]

        if spec["value_col"] not in df.columns:
            return pd.DataFrame(columns=["decile", "SMI3_Funded_min", "SMI3_Funded_max"])

        return (
            df.groupby(spec["decile_col"], observed=False)[spec["value_col"]]
            .agg(["min", "max"])
            .rename(columns={"min": "SMI3_Funded_min", "max": "SMI3_Funded_max"})
            .rename_axis("decile")
            .reset_index()
        )

    def plot_decile_confusions(
        self,
        decile_kind: str = "prob",
        figsize=(18, 5),
        fig_name: str | None = None,
    ):
        display(
            Markdown(
                "Decile confusion comparison skipped: this class is configured "
                "for SMI3_Funded only."
            )
        )
        return None

    # ---------- plots ----------
    def plot_confusion_matrix(self, ax):
        if self.target_col not in self.data.columns:
            ax.set_axis_off()
            ax.set_title("Confusion Matrix (target missing)")
            return

        y_true = self.data[self.target_col].to_numpy()
        y_pred = self._get_predictions_hard()
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title("Confusion Matrix")

    def plot_category_distributions_panel(
        self,
        cats=("Delq_Only", "Delq_Claim", "Declined"),
        figsize=(18, 5),
        fig_name="category_distributions_panel.png",
        binwidth=0.01,
    ):
        df = self.add_or_update_predictions()
        fig, axes = plt.subplots(1, len(cats), figsize=figsize, sharey=False)
        fig.suptitle("SMI3 Funded Probability Distribution by Category", fontsize=12)

        if len(cats) == 1:
            axes = [axes]

        for ax, cat in zip(axes, cats):
            d = df[df[self.category_col] == cat]
            x = d[self.smi3_funded_score_col].dropna()

            if x.empty:
                ax.set_title(f"{cat} (no data)")
                ax.axis("off")
                continue

            lo = x.min()
            hi = x.max()

            if pd.isna(lo) or pd.isna(hi) or lo >= hi:
                ax.set_title(f"{cat} (invalid range)")
                ax.axis("off")
                continue

            bins = np.arange(lo, hi + binwidth, binwidth)
            if len(bins) < 2:
                ax.set_title(f"{cat} (no bins)")
                ax.axis("off")
                continue

            ax.hist(x, bins=bins, alpha=0.5, label="SMI3_Funded")
            ax.set_title(cat)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.legend()

        plt.tight_layout()
        out_path = self.OUT / fig_name
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        return out_path

    def draw_score_chart(
        self,
        rr_smi3_funded_df: pd.DataFrame,
        ax=None,
    ):
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            created_fig = True

        if "pct_crd_score_ge_700" in rr_smi3_funded_df.columns:
            ax.plot(
                rr_smi3_funded_df["model_decile"].astype(str),
                rr_smi3_funded_df["pct_crd_score_ge_700"],
                marker="o",
                linewidth=2,
                label="SMI3_Funded",
            )

        ax.set_title("Credit Score pct > 700 by Risk Decile")
        ax.set_xlabel("Risk Decile")
        ax.set_ylabel("CRD_SCORE % > 700")
        ax.grid(alpha=0.3)
        ax.legend()

        if created_fig:
            plt.tight_layout()
            out_path = self.OUT / "score_chart.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.show()
            return out_path

    def draw_lift_chart(
        self,
        rr_smi3_funded_df: pd.DataFrame,
        basis_name: str,
        ax=None,
    ):
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            created_fig = True

        ax.plot(
            rr_smi3_funded_df["model_decile"].astype(str),
            rr_smi3_funded_df["lift_Bad"],
            marker="o",
            linewidth=2,
            label="SMI3_Funded",
        )

        ax.axhline(1.0, linestyle="--")
        ax.set_title(f"Lift Chart ({basis_name})")
        ax.set_xlabel("Risk Decile")
        ax.set_ylabel("Lift")
        ax.grid(alpha=0.3)
        ax.legend()

        if created_fig:
            plt.tight_layout()
            out_path = self.OUT / f"lift_chart_{basis_name}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.show()
            return out_path

    def draw_gain_chart(
        self,
        rr_smi3_funded_df: pd.DataFrame,
        basis_name: str,
        ax=None,
    ):
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            created_fig = True

        ax.plot(
            rr_smi3_funded_df["cum_pct_population"],
            rr_smi3_funded_df["cum_pct_Bad"],
            marker="o",
            linewidth=2,
            label="SMI3_Funded",
        )

        ax.set_title(f"Gain Chart ({basis_name})")
        ax.set_xlabel("Cumulative % of population")
        ax.set_ylabel("Cumulative % of Bad captured")
        ax.grid(alpha=0.3)
        ax.legend()

        if created_fig:
            plt.tight_layout()
            out_path = self.OUT / f"gain_chart_{basis_name}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.show()
            return out_path

    def plot_category_capture_curves(
        self,
        rr_smi3_funded_df: pd.DataFrame,
        basis_name: str,
        figsize=(20, 5),
        fig_name: str | None = None,
    ):
        fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False)
        fig.suptitle("SMI3 Funded Category Capture Curves", fontsize=20)

        curve_specs = [
            ("cum_pct_Delq_Only", "Delq_Only"),
            ("cum_pct_Delq_Claim", "Delq_Claim"),
            ("cum_pct_Declined", "Declined"),
            ("cum_pct_Bad", "Bad"),
        ]

        for ax, (y_col, title) in zip(axes, curve_specs):
            if y_col in rr_smi3_funded_df.columns:
                ax.plot(
                    rr_smi3_funded_df["cum_pct_population"],
                    rr_smi3_funded_df[y_col],
                    marker="o",
                    linewidth=2,
                    label="SMI3_Funded",
                )
            ax.set_title(title)
            ax.set_xlabel("Cumulative % of population")
            ax.set_ylabel("Cumulative % captured")
            ax.grid(alpha=0.3)
            ax.legend()

        plt.tight_layout()

        if fig_name is None:
            fig_name = f"category_capture_curves_{basis_name}.png"

        out_path = self.OUT / fig_name
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        return out_path

    def plot_score_lift_gain_panel(
        self,
        basis: str | None = None,
        figsize=(18, 5),
        fig_name: str | None = None,
    ):
        rr_smi3_funded_df = self.risk_ranking(basis=basis)
        rr_smi3_funded_df = self._sort_decile_labels(rr_smi3_funded_df)

        basis_name = self._risk_ranking_spec(basis=basis)["basis_name"]

        plots = []
        if self.show_score_chart:
            plots.append("score")
        if self.show_lift_chart:
            plots.append("lift")
        if self.show_gain_chart:
            plots.append("gain")

        if not plots:
            return None

        fig, axes = plt.subplots(1, len(plots), figsize=figsize, squeeze=False)
        axes = axes[0]
        col_idx = 0

        if self.show_score_chart:
            self.draw_score_chart(rr_smi3_funded_df, ax=axes[col_idx])
            col_idx += 1

        if self.show_lift_chart:
            self.draw_lift_chart(
                rr_smi3_funded_df,
                basis_name=basis_name,
                ax=axes[col_idx],
            )
            col_idx += 1

        if self.show_gain_chart:
            self.draw_gain_chart(
                rr_smi3_funded_df,
                basis_name=basis_name,
                ax=axes[col_idx],
            )

        plt.tight_layout()

        if fig_name is None:
            fig_name = f"score_lift_gain_panel_{basis_name}.png"

        out_path = self.OUT / fig_name
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        return out_path

    # ---------- one-call entry ----------
    def result_outputs(self) -> None:
        rr_smi3_funded_df = self.risk_ranking()

        saved = self.save_predicted_output()
        if saved is not None:
            display(Markdown(f"Saved scored df to: `{saved}`"))

        if self.plot_confusion:
            fig, ax = plt.subplots(figsize=(6, 5))
            self.plot_confusion_matrix(ax)
            plt.tight_layout()
            plt.show()

        if self.plot_deciles:
            ranges_prob = self.get_decile_ranges(decile_kind="prob")
            display(Markdown("### Decile Ranges (SMI3_Funded, Prob)"))
            display(ranges_prob)

            if self.smi3_funded_lr_col in self.data.columns:
                ranges_lr = self.get_decile_ranges(decile_kind="lr")
                display(Markdown("### Decile Ranges (SMI3_Funded, LR)"))
                display(ranges_lr)

        if self.plot_category_distributions:
            dist_path = self.plot_category_distributions_panel(figsize=(18, 5))
            display(Markdown(f"Saved category distribution panel to: `{dist_path}`"))

        if self.show_score_chart or self.show_lift_chart or self.show_gain_chart:
            panel_path = self.plot_score_lift_gain_panel(
                basis=self.risk_ranking_basis,
                figsize=(18, 6),
            )
            if panel_path is not None:
                display(Markdown(f"Saved score/lift/gain panel to: `{panel_path}`"))

        if self.show_category_capture_curves:
            capture_path = self.plot_category_capture_curves(
                rr_smi3_funded_df=self._sort_decile_labels(rr_smi3_funded_df),
                basis_name=self._risk_ranking_spec(self.risk_ranking_basis)["basis_name"],
                figsize=(24, 6),
            )
            display(Markdown(f"Saved category capture panel to: `{capture_path}`"))

        if self.show_risk_ranking_df:
            display(
                Markdown(
                    f"### Risk Ranking (basis={self.risk_ranking_basis}) "
                    "for SMI3_Funded"
                )
            )
            display(rr_smi3_funded_df)


cols_mean_base = ["ACTLTV", "NUM_OF_BOR", "TDSR", "GDSR", "PRK_PROPERTY_AGE"]
categories = ["Delq_Only", "Delq_Claim", "Declined", "No_Event"]
bad_categories = ["Delq_Only", "Delq_Claim", "Declined"]

def extra_modelresult(b):
    def summarize(df, decile_col, crd_score_col, model_score_col):
        d = df.copy().rename(columns={decile_col: "Decile"})
        g = d.groupby("Decile", observed=False)

        out = g[cols_mean_base].mean()
        out["CRD_SCORE"] = g[crd_score_col].mean()

        def _extra(x):
            s = {}
            for k in categories:
                m = x["category"].eq(k)
                s[f"n_{k}"] = int(m.sum())
                s[f"avg_{k}"] = x.loc[m, model_score_col].mean()
                s[f"sum_{k}"] = x.loc[m, model_score_col].sum()

            s["n_total"] = len(x)
            s["n_Bad"] = x["category"].isin(bad_categories).sum()
            s["n_No_Event"] = x["category"].eq("No_Event").sum()

            s["pct_crd_score_ge_700"] = x[crd_score_col].ge(700).mean() * 100
            s["pct_actltv_ge_90"] = x["ACTLTV"].ge(90).mean() * 100
            s["pct_tdsr_ge_38"] = x["TDSR"].ge(38).mean() * 100
            s["pct_gdsr_ge_33"] = x["GDSR"].ge(33).mean() * 100

            return pd.Series(s)

        extra = g.apply(_extra).reset_index()
        out = out.reset_index().merge(extra, on="Decile", how="left")

        out["avg_Bad"] = (
            out["avg_Delq_Only"].fillna(0)
            + out["avg_Delq_Claim"].fillna(0)
            + out["avg_Declined"].fillna(0)
        ) / 3

        out["sum_Bad"] = (
            out["sum_Delq_Only"].fillna(0)
            + out["sum_Delq_Claim"].fillna(0)
            + out["sum_Declined"].fillna(0)
        )

        out["n_Bad"] = (
            out["n_Delq_Only"].fillna(0)
            + out["n_Delq_Claim"].fillna(0)
            + out["n_Declined"].fillna(0)
        )

        out = out.sort_values(
            "Decile",
            key=lambda x: x.astype(str).str.extract(r"(\d+)")[0].astype(int)
        ).reset_index(drop=True)

        count_cols = ["n_Delq_Only", "n_Delq_Claim", "n_Declined", "n_Bad", "n_No_Event"]
        for c in count_cols:
            total = out[c].sum()
            out[f"pct_{c}"] = out[c] / total * 100 if total > 0 else np.nan

        for k in ["Delq_Only", "Delq_Claim", "Declined", "Bad", "No_Event"]:
            n_col = f"n_{k}"
            out[f"cum_{n_col}"] = out[n_col].cumsum()
            total = out[n_col].sum()
            out[f"cum_pct_{k}"] = out[f"cum_{n_col}"] / total * 100 if total > 0 else np.nan

        out["cum_population"] = out["n_total"].cumsum()
        total_pop = out["n_total"].sum()
        out["cum_pct_population"] = out["cum_population"] / total_pop * 100 if total_pop > 0 else np.nan

        pop_share = out["cum_pct_population"] / 100
        bad_share = out["cum_pct_Bad"] / 100
        out["lift_Bad"] = np.where(pop_share > 0, bad_share / pop_share, np.nan)

        # ------------------------------
        # ORDERING (grouped cleanly)
        # ------------------------------
        base = ["Decile", "CRD_SCORE", "ACTLTV", "NUM_OF_BOR", "TDSR", "GDSR", "PRK_PROPERTY_AGE"]

        n_cols = ["n_total", "n_Delq_Only", "n_Delq_Claim", "n_Declined", "n_Bad", "n_No_Event"]
        avg_cols = ["avg_Delq_Only", "avg_Delq_Claim", "avg_Declined", "avg_No_Event", "avg_Bad"]
        sum_cols = ["sum_Delq_Only", "sum_Delq_Claim", "sum_Declined", "sum_No_Event", "sum_Bad"]

        pct_cols = ["pct_crd_score_ge_700", "pct_actltv_ge_90", "pct_tdsr_ge_38", "pct_gdsr_ge_33"] + [
            f"pct_{c}" for c in n_cols if f"pct_{c}" in out.columns
        ]

        cum_n_cols = [f"cum_n_{k}" for k in ["Delq_Only", "Delq_Claim", "Declined", "Bad", "No_Event"]]
        cum_pct_cols = [f"cum_pct_{k}" for k in ["Delq_Only", "Delq_Claim", "Declined", "Bad", "No_Event"]]

        tail = ["cum_population", "cum_pct_population", "lift_Bad"]

        ordered = base + n_cols + avg_cols + sum_cols + pct_cols + cum_n_cols + cum_pct_cols + tail
        ordered = [c for c in ordered if c in out.columns]

        return out[ordered].round(4)

    rr_map = {
        "SMI2": summarize(b, "rr_smi2_lr_decile", "CRD_SCORE_CV_SMI2", "SMI2_LR"),
        "SMI3_Funded": summarize(b, "rr_smi3_funded_lr_decile", "CRD_SCORE", "SMI3_LR_funded"),
        "SMI3_All": summarize(b, "rr_smi3_all_lr_decile", "CRD_SCORE", "SMI3_LR_all"),
    }

    for name, df_out in rr_map.items():
        print(f"\n=== Risk Ranking (basis=lr) for {name} ===")
        display(df_out)

    def _plot(ax, x, y, title, xlabel, ylabel, baseline=None):
        for name, df_plot in rr_map.items():
            ax.plot(df_plot[x], df_plot[y], marker="o", linewidth=1.8, markersize=4, label=name)
        if baseline is not None:
            ax.axhline(baseline, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 3, figsize=(25, 6))
    _plot(axes[0], "Decile", "pct_crd_score_ge_700", "Credit Score pct > 700", "Decile", "%")
    _plot(axes[1], "Decile", "lift_Bad", "Lift (lr)", "Decile", "Lift", 1.0)
    _plot(axes[2], "cum_pct_population", "cum_pct_Bad", "Gain (Bad)", "Cum % Pop", "Cum % Bad")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    _plot(axes[0], "cum_pct_population", "cum_pct_Delq_Only", "Delq_Only Capture", "Cum % Pop", "Cum %")
    _plot(axes[1], "cum_pct_population", "cum_pct_Delq_Claim", "Delq_Claim Capture", "Cum % Pop", "Cum %")
    _plot(axes[2], "cum_pct_population", "cum_pct_Declined", "Declined Capture", "Cum % Pop", "Cum %")
    _plot(axes[3], "cum_pct_population", "cum_pct_Bad", "Bad Capture", "Cum % Pop", "Cum %")
    fig.tight_layout()
    plt.show()

