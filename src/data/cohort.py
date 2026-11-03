"""Build the patient cohort: one slide, one label, one row per patient.

Applies the decisions in docs/methods.md:
  D4  patients with no JPG in the release are excluded (availability is read from the
      committed artifacts/slide_scale.csv, so the cohort never depends on local disk state);
  D5  each patient uses their lowest-numbered slide (numeric, not text order);
  D10 patients are grouped by SampleNumber only, so 8377886 is one patient;
  D11 label, age and sex come from the chosen slide.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CLASSES = ("In situ", "Lepidic", "Acinar", "Papillary", "Micropapillary", "Solid", "Cribriform")
SEXES = ("Female", "Male")


def _validate(meta: pd.DataFrame) -> None:
    """Fail loudly on anything the cohort rules do not expect."""
    required = {"SampleNumber", "WSI_ID", "Sex", "Age", "Tumor Subtype"}
    if missing := required - set(meta.columns):
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    if meta["WSI_ID"].duplicated().any():
        raise ValueError(f"duplicate WSI_ID: {meta.loc[meta.WSI_ID.duplicated(), 'WSI_ID'].tolist()}")
    if not meta["WSI_ID"].str.fullmatch(r"WSI-\d+").all():
        raise ValueError("WSI_ID values must look like 'WSI-<n>'")
    if bad := set(meta["Tumor Subtype"]) - set(CLASSES):
        raise ValueError(f"unknown labels: {sorted(bad)}")
    if bad := set(meta["Sex"]) - set(SEXES):
        raise ValueError(f"unknown sex values: {sorted(bad)}")
    if meta["Age"].isna().any():
        raise ValueError("missing Age values")


def build_cohort(meta: pd.DataFrame, slide_scale: pd.DataFrame,
                 images_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cohort, excluded): one row per patient, and the patients dropped with a reason.

    If images_dir is given, the JPGs on disk must match slide_scale.csv exactly.
    """
    _validate(meta)
    slides = meta.merge(slide_scale[["WSI_ID", "jpg_mpp"]], on="WSI_ID", how="left", validate="one_to_one")
    slides["wsi_num"] = slides["WSI_ID"].str[4:].astype(int)
    slides["has_image"] = slides["jpg_mpp"].notna()
    if images_dir is not None:
        on_disk = slides["WSI_ID"].map(lambda w: (Path(images_dir) / f"{w}.jpg").is_file())
        if mismatch := slides.loc[slides.has_image != on_disk, "WSI_ID"].tolist():
            raise ValueError(f"local images disagree with slide_scale.csv (download incomplete?): {mismatch}")

    usable = slides[slides.has_image]
    excluded_ids = sorted(set(slides.SampleNumber) - set(usable.SampleNumber))
    excluded = (slides[slides.SampleNumber.isin(excluded_ids)]
                .groupby("SampleNumber", as_index=False)
                .agg(WSI_IDs=("WSI_ID", lambda s: " ".join(sorted(s, key=lambda w: int(w[4:])))),
                     label=("Tumor Subtype", "first"))
                .assign(reason="no JPG in the Hugging Face release"))

    chosen = usable.sort_values("wsi_num").groupby("SampleNumber", as_index=False).first()
    stats = usable.groupby("SampleNumber").agg(n_slides=("WSI_ID", "size"),
                                               n_labels=("Tumor Subtype", "nunique"))
    cohort = (chosen.join(stats, on="SampleNumber")
              .assign(label_conflict=lambda d: d.n_labels > 1)
              .rename(columns={"Tumor Subtype": "label", "Age": "age", "Sex": "sex"})
              [["SampleNumber", "WSI_ID", "label", "age", "sex", "n_slides", "label_conflict", "jpg_mpp"]]
              .sort_values("SampleNumber", ignore_index=True))
    return cohort, excluded
