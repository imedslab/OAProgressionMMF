"""
Entry point to all available datasets, subsets, folds, and dataloaders.
"""

import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import sklearn.model_selection
from torch.utils.data import DataLoader, WeightedRandomSampler

from koafusion import preproc
from koafusion.various import StratifiedGroupKFold
from koafusion.datasets import DatasetOAI3d
from koafusion.datasets.oai import index_from_path_oai


logging.basicConfig()
logger = logging.getLogger("provider")
logger.setLevel(logging.DEBUG)


def sources_from_path(*, path_data_root, modals_all, target, fold_num, scheme_train_val,
                      seed_trainval_test, site_test, seed_train_val, ignore_cache=False):
    """

    Args:
        path_data_root (str):
        modals_all:
        target (str):
        fold_num (int):
            Number of folds.
        scheme_train_val:
        seed_trainval_test (int):
            Random state for the trainval/test splitting.
        site_test:
        seed_train_val (int):
            Random state for the train/val splitting.
        ignore_cache:

    Returns:

    """
    assert scheme_train_val in ("strat_target", "one_site_out")

    def _select_subjects_target(df_i):
        df_o = df_i.copy()
        logger.info(f"Using target: {target}!")

        if target.startswith("prog_kl_"):
            field_target = target
            field_sel = {
                "prog_kl_12": "panfilov_sel_kl_12",
                "prog_kl_24": "panfilov_sel_kl_24",
                "prog_kl_36": "panfilov_sel_kl_36",
                "prog_kl_48": "panfilov_sel_kl_48",
                "prog_kl_72": "panfilov_sel_kl_72",
                "prog_kl_96": "panfilov_sel_kl_96"}[field_target]
            # Output sample rejection reason if any
            field_reason = f"reason_kl_{field_target.split('_')[-1]}"
            print(df_o[("-", field_reason)].value_counts().sort_index(ascending=True))
            df_o[("-", "target")] = df_o[("-", field_target)]
            df_o = df_o[df_o[("-", field_sel)] == 1]
            df_o = df_o[df_o[("-", "target")] != -1]

        elif target in ("tiulpin2019_prog_bin",):
            df_o[("-", "target")] = df_o[("-", "tiulpin2019_prog")]
            df_o = df_o[df_o[("-", "tiulpin2019_sel")] == 1]
            df_o = df_o[df_o[("-", "target")] != -1]
            df_o.loc[df_o[("-", "target")] >= 1, ("-", "target")] = 1
            df_o[("-", "tiulpin2019_prog_bin")] = df_o[("-", "target")]

        else:
            raise ValueError(f"Unsupported target: {target}")
        return df_o

    def _exclude_corrupted_imaging(df_i):
        corr = [
            ("9004315", "000m", "RIGHT"),  # "SAG_3D_DESS"
            ("9522128", "000m", "RIGHT"),  # "SAG_3D_DESS"  [352, 352, 29]
            ("9560965", "000m", "RIGHT"),  # "SAG_3D_DESS"
            ("9594253", "000m", "LEFT"),  # "SAG_3D_DESS"  [352, 352, 124]
            ("9617608", "000m", "LEFT"),  # "SAG_3D_DESS"  [352, 288, 160]
            ("9637394", "000m", "RIGHT"),  # "SAG_3D_DESS"  [352, 352, 56]
            ("9176992", "000m", "RIGHT"),  # "SAG_T2_MAP"  NaNs
            ("9445104", "000m", "RIGHT"),  # "SAG_T2_MAP"  NaNs
            ("9481413", "000m", "RIGHT"),  # "SAG_T2_MAP"  NaNs
            ("9733288", "000m", "RIGHT"),  # "SAG_T2_MAP"  NaNs
            ("9952664", "000m", "RIGHT"),  # "SAG_T2_MAP"  NaNs
            ("9952817", "000m", "RIGHT"),  # "SAG_T2_MAP"  NaNs
            ("9006140", "000m", "RIGHT"),  # "SAG_T2_MAP"  zeroes in T2 map
            ("9594551", "000m", "RIGHT"),  # "SAG_T2_MAP"  zeroes in T2 map
            ("9641467", "000m", "RIGHT"),  # "SAG_T2_MAP"  zeroes in T2 map
            ("9700206", "000m", "LEFT"),  # "SAG_T2_MAP"  zeroes in T2 map
            ("9768219", "000m", "RIGHT"),  # "SAG_T2_MAP"  zeroes in T2 map
            ("9777471", "000m", "RIGHT"),  # "SAG_T2_MAP"  zeroes in T2 map
        ]

        df_o = df_i.copy()
        for c in corr:
            df_o = df_o[~(
                (df_o[("-", "patient")] == c[0]) &
                (df_o[("-", "visit_month")] == c[1]) &
                (df_o[("-", "side")] == c[2])
            )]
        return df_o

    def _exclude_missing_clin(df_i):
        df_o = df_i.copy()
        df_o = df_o.dropna(axis=0, subset=[("-", "P01BMI"), ])
        logger.info("  Removed samples with missing BMI:")
        _count_unique_subjects_knees(df_o)

        df_o = df_o[df_o[("-", "WOMTS-")] >= 0.0]
        logger.info("  Removed samples with missing WOMAC:")
        _count_unique_subjects_knees(df_o)

        df_o = df_o[df_o[("-", "P01INJ-")] != -1]
        logger.info(f"  Removed samples with missing Inj:")
        _count_unique_subjects_knees(df_o)

        df_o = df_o[df_o[("-", "P01KSURG-")] != -1]
        logger.info("  Removed samples with missing Surg:")
        _count_unique_subjects_knees(df_o)
        return df_o

    def _count_unique_subjects_knees(df_i, prefix="    "):
        logger.info(f"{prefix}num.subjects: {len(pd.unique(df_i[('-', 'patient')]))}")
        logger.info(f"{prefix}num.knees: {len(df_i)}")

    path_data_root = Path(path_data_root).resolve()

    t_df = dict()
    sources = dict()

    t_df["full_df"] = index_from_path_oai(path_root=path_data_root,
                                          modals_all=modals_all,
                                          ignore_cache=ignore_cache)
    logger.info("Created index:")
    _count_unique_subjects_knees(t_df["full_df"])

    # Select the specific subset
    t_df["sel_df"] = t_df["full_df"].copy()
    logger.info("After excl. - corrupted imaging:")
    t_df["sel_df"] = _exclude_corrupted_imaging(t_df["sel_df"])
    _count_unique_subjects_knees(t_df["sel_df"])

    logger.info("After excl. - missing clinical:")
    t_df["sel_df"] = _exclude_missing_clin(t_df["sel_df"])
    _count_unique_subjects_knees(t_df["sel_df"])

    logger.info("After excl. - missing target:")
    t_df["sel_df"] = _select_subjects_target(t_df["sel_df"])
    _count_unique_subjects_knees(t_df["sel_df"])

    logger.info("Selected before splitting:")
    _count_unique_subjects_knees(t_df["sel_df"])

    # Get trainval/test split
    t_df["trainval_df"] = t_df["sel_df"][t_df["sel_df"][("-", "V00SITE")] != site_test]
    t_df["test_df"] = t_df["sel_df"][t_df["sel_df"][("-", "V00SITE")] == site_test]
    logger.info("Made trainval-test split:")
    logger.info("  trainval:")
    _count_unique_subjects_knees(t_df["trainval_df"])
    logger.info("  test:")
    _count_unique_subjects_knees(t_df["test_df"])

    # Make train_val folds
    if scheme_train_val == "strat_target":
        t_gkf = StratifiedGroupKFold(n_splits=fold_num,
                                     shuffle=True,
                                     random_state=seed_train_val)
        t_grades = t_df["trainval_df"].loc[:, ("-", "target")].values
        t_groups = t_df["trainval_df"].loc[:, ("-", "patient")].values

        t_df["trainval_folds"] = t_gkf.split(X=t_df["trainval_df"],
                                             y=t_grades,
                                             groups=t_groups)
    elif scheme_train_val == "one_site_out":
        t_gkf = sklearn.model_selection.LeaveOneGroupOut()
        t_grades = t_df["trainval_df"].loc[:, ("-", "target")].values
        t_groups = t_df["trainval_df"].loc[:, ("-", "V00SITE")].values
        # Treat low-data sites A and E as one
        t_groups[t_groups == "E"] = "A"

        t_df["trainval_folds"] = t_gkf.split(X=t_df["trainval_df"],
                                             y=t_grades,
                                             groups=t_groups)

    sources["oai"] = t_df
    return sources


def prepare_datasets_loaders(config, fold_idx):
    """

    Returns:
        (datasets, loaders)
    """
    datasets = defaultdict(dict)
    loaders = defaultdict(dict)

    # Collect available sources and make splits
    sources = sources_from_path(
        path_data_root=config.path_data_root,
        modals_all=config.data.modals_all,
        target=config.data.target,
        fold_num=config.training.folds.num,
        # scheme_trainval_test=config.scheme_trainval_test,
        scheme_train_val=config.scheme_train_val,
        seed_trainval_test=config.seed_trainval_test,
        seed_train_val=config.seed_train_val,
        site_test=config.site_test,
        ignore_cache=config.data.ignore_cache,
    )

    # ds_names = [n for n in sources.keys()]
    ds_names = [d.name for _, d in config.data.sets.items()]

    # Use straightforward fold allocation strategy
    folds_seq = [sources[n]["trainval_folds"] for n in ds_names]
    folds_zip = list(zip(*folds_seq))
    # Select fold
    idcs_subsets = folds_zip[fold_idx]

    for idx, (_, ds) in enumerate(config.data.sets.items()):
        stats_classes = pd.value_counts(
            sources[ds.name]["sel_df"][("-", "target")]).to_dict()
        logger.info(f"Number of class occurrences in selected dataset: {stats_classes}")

        stats_classes = pd.value_counts(
            sources[ds.name]["trainval_df"][("-", "target")]).to_dict()
        logger.info(f"Number of class occurrences in trainval subset: {stats_classes}")

        sources[ds.name]["train_idcs"] = idcs_subsets[idx][0]
        sources[ds.name]["val_idcs"] = idcs_subsets[idx][1]

        sources[ds.name]["train_df"] = \
            sources[ds.name]["trainval_df"].iloc[sources[ds.name]["train_idcs"]]
        sources[ds.name]["val_df"] = \
            sources[ds.name]["trainval_df"].iloc[sources[ds.name]["val_idcs"]]

    # Select fraction of samples keeping balance of targets
    for idx, (_, ds) in enumerate(config.data.sets.items()):
        frac = ds.frac_classw
        if frac != 1.0:
            logger.warning(f"Sampled fraction of {frac} per target class")

            df_tmp = sources[ds.name]["train_df"]
            df_tmp = (df_tmp
                      .sort_values([("-", "target"), ])
                      .groupby(("-", "target"))
                      .sample(frac=frac, random_state=0))
            logger.warning(f"Selected only {len(df_tmp)} samples from train")
            sources[ds.name]["train_df"] = df_tmp

            df_tmp = sources[ds.name]["val_df"]
            df_tmp = (df_tmp
                      .sort_values([("-", "target"), ])
                      .groupby(("-", "target"))
                      .sample(frac=frac, random_state=0))
            logger.warning(f"Selected only {len(df_tmp)} samples from val")
            sources[ds.name]["val_df"] = df_tmp

        for n, s in sources.items():
            logger.info("Made {} train-val split, number of samples: {}, {}"
                        .format(n, len(s["train_df"]), len(s["val_df"])))
            logger.info("Test subset, number of samples: {}".format(len(s["test_df"])))

    # INFO: exclude Inj+/Surg+ from trainval
    for idx, (_, ds) in enumerate(config.data.sets.items()):
        for df_name in ("train_df", "val_df"):
            df_tmp = sources[ds.name][df_name]
            if config.data.exclude_inj:
                df_tmp = df_tmp[df_tmp[("-", "P01INJ-")] != 1]
                logger.warning(f"Excluded Inj+ from {df_name}!")
                logger.warning(f"Samples left: {len(df_tmp)}")
            if config.data.exclude_surg:
                df_tmp = df_tmp[df_tmp[("-", "P01KSURG-")] != 1]
                logger.warning(f"Excluded Surg+ from {df_name}!")
                logger.warning(f"Samples left: {len(df_tmp)}")
            sources[ds.name][df_name] = df_tmp

    # Initialize datasets
    for _, ds in config.data.sets.items():
        transfs = defaultdict(dict)  # (ds_modal, regime): []

        for idx, modal in enumerate(ds.modals):
            # Transforms and augmentations
            transfs["train"][modal] = []
            transfs["val"][modal] = []
            transfs["test"][modal] = []

            if (ds.name, modal) in (("oai", "sag_3d_dess"), ("oai", "cor_iw_tse")):
                transfs["train"][modal].extend([
                    preproc.RandomCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                    preproc.PTRotate3DInSlice(degree_range=[-15., 15.], prob=0.5),
                    preproc.PTGammaCorrection(gamma_range=(0.5, 2.0), prob=0.5, clip_to_unit=False),
                ])
            elif (ds.name, modal) == ("oai", "sag_t2_map"):
                transfs["train"][modal].extend([
                    preproc.RandomCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                    preproc.PTRotate3DInSlice(degree_range=[-15., 15.], prob=0.5),
                ])
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["train"][modal].extend([
                    preproc.RandomCrop(output_size=list(config.model.input_size[idx]), ndim=2),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                    preproc.PTRotate2D(degree_range=[-15., 15.], prob=0.5),
                    preproc.PTGammaCorrection(gamma_range=(0.5, 2.0), prob=0.5, clip_to_unit=False),
                ])
            elif (ds.name, modal) == ("oai", "clin"):
                pass
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["train"][modal].append(
                    preproc.PTNormalize(mean=[0.257, ], std=[0.235, ]))
            elif (ds.name, modal) == ("oai", "cor_iw_tse"):
                transfs["train"][modal].append(
                    preproc.PTNormalize(mean=[0.455, ], std=[0.290, ]))
            elif (ds.name, modal) == ("oai", "sag_t2_map"):
                transfs["train"][modal].append(
                    preproc.PTNormalize(mean=[0.259, ], std=[0.345, ]))
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["train"][modal].append(
                    preproc.PTNormalize(mean=[0.543, ], std=[0.296, ]))
            elif (ds.name, modal) == ("oai", "clin"):
                pass
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) in (("oai", "sag_3d_dess"), ("oai", "cor_iw_tse")):
                transfs["val"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            elif (ds.name, modal) == ("oai", "sag_t2_map"):
                transfs["val"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["val"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=2),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            elif (ds.name, modal) == ("oai", "clin"):
                pass
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["val"][modal].append(
                    preproc.PTNormalize(mean=[0.257, ], std=[0.235, ]))
            elif (ds.name, modal) == ("oai", "cor_iw_tse"):
                transfs["val"][modal].append(
                    preproc.PTNormalize(mean=[0.455, ], std=[0.290, ]))
            elif (ds.name, modal) == ("oai", "sag_t2_map"):
                transfs["val"][modal].append(
                    preproc.PTNormalize(mean=[0.259, ], std=[0.345, ]))
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["val"][modal].append(
                    preproc.PTNormalize(mean=[0.543, ], std=[0.296, ]))
            elif (ds.name, modal) == ("oai", "clin"):
                pass
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) in (("oai", "sag_3d_dess"), ("oai", "cor_iw_tse")):
                transfs["test"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            elif (ds.name, modal) == ("oai", "sag_t2_map"):
                transfs["test"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=3),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["test"][modal].extend([
                    preproc.CenterCrop(output_size=list(config.model.input_size[idx]), ndim=2),
                    preproc.NumpyToTensor(),
                    preproc.PTToUnitRange(),
                ])
            elif (ds.name, modal) == ("oai", "clin"):
                pass
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

            if (ds.name, modal) == ("oai", "sag_3d_dess"):
                transfs["test"][modal].append(
                    preproc.PTNormalize(mean=[0.257, ], std=[0.235, ]))
            elif (ds.name, modal) == ("oai", "cor_iw_tse"):
                transfs["test"][modal].append(
                    preproc.PTNormalize(mean=[0.455, ], std=[0.290, ]))
            elif (ds.name, modal) == ("oai", "sag_t2_map"):
                transfs["test"][modal].append(
                    preproc.PTNormalize(mean=[0.259, ], std=[0.345, ]))
            elif (ds.name, modal) == ("oai", "xr_pa"):
                transfs["test"][modal].append(
                    preproc.PTNormalize(mean=[0.543, ], std=[0.296, ]))
            elif (ds.name, modal) == ("oai", "clin"):
                pass
            else:
                raise ValueError(f"Unknown dataset/modality: {ds.name}/{modal}")

        if ds.name == "oai":
            cls = DatasetOAI3d
        else:
            raise ValueError(f"Unknown dataset {ds.name}")

        # Instantiate datasets
        datasets[ds.name]["sel"] = cls(
            df_meta=sources[ds.name]["sel_df"],
            modals=ds.modals,
            transforms=None)
        datasets[ds.name]["train"] = cls(
            df_meta=sources[ds.name]["train_df"],
            modals=ds.modals,
            transforms=transfs["train"])
        datasets[ds.name]["val"] = cls(
            df_meta=sources[ds.name]["val_df"],
            modals=ds.modals,
            transforms=transfs["val"])
        datasets[ds.name]["test"] = cls(
            df_meta=sources[ds.name]["test_df"],
            modals=ds.modals,
            transforms=transfs["test"])

        if config.data.debug:
            logger.info("---- Estimate intensity stats")
            t = cls(
                # df_meta=sources[ds.name]["trainval_df"],
                df_meta=sources[ds.name]["sel_df"],
                modals=ds.modals,
                transforms={ds.modals[0]: [
                    # preproc.CenterCrop(output_size=list(config.model.input_size[0]), ndim=2),
                    preproc.CenterCrop(output_size=list(config.model.input_size[0]), ndim=3),
                    # preproc.NumpyToTensor(),
                    # preproc.PTToUnitRange(),

                    # preproc.PTNormalize(mean=[0.259, ], std=[0.345, ])
                ]})
            t.describe()
            quit()

    # Initialize data loaders
    for _, ds in config.data.sets.items():
        # Configure samplers
        if config.training.sampler == "weighted":
            logger.info("Using frequency-based sampler for training subset")
            t_df = sources[ds.name]["train_df"]
            map_freqs = t_df[("-", "target")].value_counts(normalize=True).to_dict()
            sample_weights = [1.0 / map_freqs[e] for e in t_df[("-", "target")].tolist()]
            sampler_train = WeightedRandomSampler(weights=sample_weights,
                                                  num_samples=len(sample_weights),
                                                  replacement=True)
        elif config.training.sampler == "default":
            logger.info("Using default sampler for training subset")
            sampler_train = None
        else:
            raise ValueError(f"Invalid sampler {config.training.sampler}")

        # Instantiate dataloaders
        loaders[ds.name]["train"] = DataLoader(
            datasets[ds.name]["train"],
            batch_size=config.training.batch_size,
            sampler=sampler_train,
            num_workers=config.num_workers,
            drop_last=True)

        logger.warning("Validation balanced sampling is disabled!")
        loaders[ds.name]["val"] = DataLoader(
            datasets[ds.name]["val"],
            batch_size=config.validation.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=True)

        loaders[ds.name]["test"] = DataLoader(
            datasets[ds.name]["test"],
            batch_size=config.testing.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False)

    return datasets, loaders
