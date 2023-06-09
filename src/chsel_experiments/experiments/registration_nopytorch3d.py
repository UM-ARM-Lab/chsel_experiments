import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

import pytorch_volumetric.sdf
from pytorch_volumetric import voxel
from base_experiments import cfg
from pytorch_volumetric.sdf import sample_mesh_points


def saved_traj_dir_base(level, experiment_name="poke"):
    return os.path.join(cfg.DATA_DIR, f"{experiment_name}/{level.name}")


def saved_traj_dir_for_method(reg_method, experiment_name="poke"):
    name = reg_method.name.lower().replace('_', '-')
    return os.path.join(cfg.DATA_DIR, f"{experiment_name}/{name}")


def saved_traj_file(reg_method, level, seed, experiment_name="poke"):
    return f"{saved_traj_dir_for_method(reg_method, experiment_name=experiment_name)}/{level.name}_{seed}.txt"


def approximate_pose_file(level, experiment_name="poke"):
    # rotation saved as xyzw
    return f"{saved_traj_dir_base(level, experiment_name=experiment_name)}_approx_pose.txt"


def optimal_pose_file(level, seed, experiment_name="poke"):
    # rotation saved as xyzw
    return f"{saved_traj_dir_base(level, experiment_name=experiment_name)}_{seed}_optimal_pose.txt"


def read_offline_output(reg_method, level, seed: int, pokes: int, experiment_name="poke"):
    filepath = saved_traj_file(reg_method, level, seed, experiment_name=experiment_name)
    if not os.path.isfile(filepath):
        raise RuntimeError(f"Missing path, should run offline method first: {filepath}")

    T = []
    distances = []
    elapsed = None
    with open(filepath) as f:
        data = f.readlines()
        i = 0
        while i < len(data):
            header = data[i].split()
            this_poke = int(header[0])
            if this_poke < pokes:
                # keep going forward
                i += 5
                continue
            elif this_poke > pokes:
                # assuming the pokes are ordered, if we're past then there won't be anymore of this poke later
                break

            transform = torch.tensor([[float(v) for v in line.strip().split()] for line in data[i + 1:i + 5]])
            T.append(transform)
            batch = int(header[1])
            # lower is better
            rmse = float(header[2])
            distances.append(rmse)
            if len(header) > 3 and header[3] != "None":
                elapsed = float(header[3])
            i += 5

    # current_to_link transform (world to base frame)
    T = torch.stack(T)
    T = T.inverse()
    distances = torch.tensor(distances)
    return T, distances, elapsed


def build_model(obj_factory: pytorch_volumetric.sdf.ObjectFactory, vis, model_name, seed, num_points, pause_at_end=False,
                device="cpu", **kwargs):
    points, normals, cache = sample_mesh_points(obj_factory, num_points=num_points,
                                                seed=seed, clean_cache=True,
                                                name=model_name,
                                                device=device, **kwargs)
    print(f"finished building {model_name} {seed} {num_points}")
    if vis is not None:
        for i, pt in enumerate(points):
            vis.draw_point(f"mpt.{i}", pt, color=(0, 0, 1), length=0.003)
            vis.draw_2d_line(f"mn.{i}", pt, normals[i], color=(0, 0, 0), size=2., scale=0.03)

        if pause_at_end:
            input("paused for inspection")
        vis.clear_visualization_after("mpt", 0)
        vis.clear_visualization_after("mn", 0)
    return cache


def plot_sdf(obj_factory, target_sdf, vis, filter_pts=None):
    obj_factory.draw_mesh(vis, "objframe", ([0, 0, 0], [0, 0, 0, 1]), (0.3, 0.3, 0.3, 0.5),
                          object_id=vis.USE_DEFAULT_ID_FOR_NAME)
    s = target_sdf
    assert isinstance(s, pytorch_volumetric.sdf.CachedSDF)
    coords, pts = voxel.get_coordinates_and_points_in_grid(s.resolution, s.ranges, device=s.device)
    if filter_pts is not None:
        pts = filter_pts(pts)
    sdf_val, sdf_grad = s(pts)

    # color code them
    error_norm = matplotlib.colors.Normalize(vmin=sdf_val.min().cpu() * 0.8, vmax=sdf_val.max().cpu() * 0.8)
    color_map = matplotlib.cm.ScalarMappable(norm=error_norm)
    rgb = color_map.to_rgba(sdf_val.reshape(-1).cpu())
    rgb = rgb[:, :-1]

    vis.draw_points("sdf_pt", pts.cpu(), color=rgb, length=0.003, scale=1, cubes=True)
    # vis.draw_2d_lines("sdf_n", pts.cpu(), sdf_grad.cpu(), color=rgb, scale=0.005, size=0.1)
    input("finished")


# prettify labels for certain values
pretty_prints = {
    "plausible_diversity_q1.0": "Plausible Diversity (mm$^2$)",
    "plausibility_q1.0": "Plausibility (mm$^2$)",
    "qd_score": "QD Score",
    "elite_costs": "Lowest $|\hat{\mathcal{T}}_0|$ cell costs",
    "poke": "Probe number",
    "iterations": "$n_o$"
}


def plot_icp_results(filter=None, logy=True, logx=False, plot_median=True, x='points', y='chamfer_err',
                     hue="method",
                     style="name",
                     key_columns=("method", "name", "seed", "points", "batch"),
                     keep_lowest_y_quantile=0.5,
                     keep_lowest_y_wrt=None,
                     fmt='box',
                     save_path=None, show=True,
                     legend=True, x_starts_at_0=False,
                     leave_out_percentile=50, icp_res_file='icp_comparison.pkl'):
    fullname = os.path.join(cfg.DATA_DIR, icp_res_file)
    df = pd.read_pickle(fullname)

    # clean up the database by removing duplicates (keeping latest results)
    df = df.drop_duplicates(subset=key_columns, keep='last')
    # save this version to keep the size small and not waste the filtering work we just did
    df.to_pickle(fullname)
    df.reset_index(inplace=True)

    if filter is not None:
        df = filter(df)

    group = [x, "method", "name", "seed"]
    if "level" in key_columns:
        group.append("level")
    if keep_lowest_y_wrt is None:
        keep_lowest_y_wrt = y
    df = df[df[keep_lowest_y_wrt] <= df.groupby(group)[keep_lowest_y_wrt].transform('quantile', keep_lowest_y_quantile)]
    df.loc[df["method"].str.contains("ICP"), "name"] = "non-freespace baseline"
    df.loc[df["method"].str.contains("VOLUMETRIC"), "name"] = "ours"
    df.loc[df["method"].str.contains("CVO"), "name"] = "freespace baseline"
    df.loc[df["method"].str.contains("MEDIAL_CONSTRAINT"), "name"] = "freespace baseline"

    method_to_name = df.set_index(hue)[style].to_dict()
    # order the methods should be shown
    full_method_order = [
        "VOLUMETRIC_CMAMEGA",
        "VOLUMETRIC_CMAME",
        "VOLUMETRIC_SVGD",
        # sometimes labelled as this
        "CMAMEGA",
        "CMAME",
        # vanilla method
        "VOLUMETRIC",
        # variants of our method
        "VOLUMETRIC_ICP_INIT",
        "VOLUMETRIC_NO_FREESPACE",
        "VOLUMETRIC_LIMITED_REINIT",
        "VOLUMETRIC_LIMITED_REINIT_FULL",
        # variants with non-SGD optimization
        "VOLUMETRIC_CMAES",
        # baselines
        "ICP",
        # "ICP_REVERSE",
        "MEDIAL_CONSTRAINT",
        "CVO",
        "MEDIAL_CONSTRAINT_CMAME",
    ]
    # method_to_hue = {
    #     "VOLUMETRIC": "#6CB0D6",
    #     # variants of our method
    #     "VOLUMETRIC_ICP_INIT": "blue",
    #     "VOLUMETRIC_NO_FREESPACE": "blue",
    #     "VOLUMETRIC_LIMITED_REINIT": "blue",
    #     "VOLUMETRIC_LIMITED_REINIT_FULL": "blue",
    #     # variants with non-SGD optimization
    #     "VOLUMETRIC_CMAES": "blue",
    #     "VOLUMETRIC_CMAME": "#089099",
    #     "VOLUMETRIC_SVGD": "green",
    #     "VOLUMETRIC_CMAMEGA": "#045275",
    #     # baselines
    #     "ICP": "#dc3977",
    #     "ICP_REVERSE": "black",
    #     "CVO": "black",
    #     "MEDIAL_CONSTRAINT": "#7c1d6f",
    # }
    # order the categories should be shown
    methods_order = [m for m in full_method_order if m in method_to_name]
    full_category_order = ["ours", "non-freespace baseline", "freespace baseline"]
    category_order = [m for m in full_category_order if m in method_to_name.values()]
    # colors = [method_to_hue[method] for method in methods_order]
    fig = plt.figure()
    if fmt == 'scatter':
        res = sns.scatterplot(data=df, x=x, y=y, hue=hue, style=style, alpha=0.5)
    elif fmt == 'line':
        if len(category_order) > 1:
            res = sns.lineplot(data=df, x=x, y=y, hue=hue,
                               style=style,  # palette=colors,
                               estimator=np.median if plot_median else np.mean,
                               hue_order=methods_order,
                               style_order=category_order,
                               errorbar=("pi", 100 - leave_out_percentile) if plot_median else ("ci", 95)
                               )
        else:
            res = sns.lineplot(data=df, x=x, y=y, hue=hue,
                               estimator=np.median if plot_median else np.mean,
                               hue_order=methods_order,
                               errorbar=("pi", 100 - leave_out_percentile) if plot_median else ("ci", 95)
                               )
    elif fmt == 'box':
        res = sns.boxplot(data=df, x=x, y=y, hue=hue,
                          hue_order=methods_order,
                          )
    else:
        raise RuntimeError(f"Unrecognized plot format {fmt}")
    if logy:
        res.set(yscale='log')
    else:
        res.set(ylim=(0, None))
    if logx:
        res.set(xscale='log')
    else:
        if x_starts_at_0:
            res.set(xlim=(0, None))

    if fmt == 'line' and len(category_order) > 1:
        # combine hue and styles in the legend
        handles, labels = res.get_legend_handles_labels()
        next_title_index = labels.index(style)
        style_dict = {label: (handle.get_linestyle(), handle.get_marker(), None)
                      for handle, label in zip(handles[next_title_index:], labels[next_title_index:])}

        for handle, label in zip(handles[1:next_title_index], labels[1:next_title_index]):
            handle.set_linestyle(style_dict[method_to_name[label]][0])
            handle.set_marker(style_dict[method_to_name[label]][1])
            dashes = style_dict[method_to_name[label]][2]
            if dashes is not None:
                handle.set_dashes(dashes)
        # create a legend only using the items
        res.legend(handles[1:next_title_index], labels[1:next_title_index], title='method', framealpha=0.4)

    if hue == "qd_method":
        res.get_legend().set_title("QD method")
    # pokes are integer
    res.xaxis.set_major_locator(MaxNLocator(integer=True))

    if y in pretty_prints:
        res.set_ylabel(pretty_prints[y])
    if x in pretty_prints:
        res.set_xlabel(pretty_prints[x])

    if legend:
        # change to our method name
        for text in res.get_legend().get_texts():
            # text.set_text(text.get_text().replace("VOLUMETRIC", "CHSEL"))
            t = text.get_text()
            t = t.replace("VOLUMETRIC", r"$\tilde{C}(\mathcal{X},\mathbf{T})$")
            if "_" in t and "MEDIAL" not in t:
                t = t.replace("_", " with ")
            if "CMAMEGA" in t:
                t += " (CHSEL)"
            text.set_text(t)
    else:
        res.get_legend().remove()

    # plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path[:-4]}_{fmt}{save_path[-4:]}")
        plt.close(fig)
    if show:
        plt.show()


def plot_poke_chamfer_err(args, level, obj_factory, key_columns, db_prefix="poking"):
    def filter(df):
        # show each level individually or marginalize over all of them
        if args.marginalize:
            df = df[(df["level"].str.contains(level.name))]
        else:
            df = df[(df["level"] == level.name)]
        return df

    def filter_single(df):
        # df = df[(df["level"] == level.name) & (df["seed"] == 0) & (df["method"] == "VOLUMETRIC")]
        df = df[(df["level"] == level.name) & (df["seed"] == args.seed[0])]
        return df

    plot_icp_results(filter=filter, icp_res_file=f"{db_prefix}_{obj_factory.name}.pkl",
                     key_columns=key_columns,
                     logy=True, keep_lowest_y_wrt="rmse",
                     save_path=os.path.join(cfg.DATA_DIR, f"img/{level.name.lower()}.png"),
                     show=not args.no_gui,
                     plot_median=False, x='poke', y='chamfer_err')


def plot_poke_plausible_diversity(args, level, obj_factory, key_columns, quantile=1.0, db_prefix="poking", **kwargs):
    """Choose quantile from {0.50, 0.75, 1.0} which indicate when the plausible diversity is computed from a
    subset of the estimated transform set that includes only transforms above a certain quantile of loss"""

    def filter(df):
        if args.marginalize:
            df = df[(df["level"].str.contains(level.name))]
        else:
            df = df[(df["level"] == level.name)]
        df = df[df.batch == 0]
        df = df[df['plausibility_q1.0'].notnull()]
        df = df[df.name == args.name]
        df = df[df.seed.isin(args.seed)]
        return df

    for y in ['plausibility', 'coverage', 'plausible_diversity']:
        plot_icp_results(filter=filter, icp_res_file=f"{db_prefix}_{obj_factory.name}.pkl",
                         key_columns=key_columns,
                         logy=True, keep_lowest_y_quantile=1.0,
                         save_path=os.path.join(cfg.DATA_DIR, f"img/{level.name.lower()}__{y}.pdf"),
                         show=not args.no_gui,
                         plot_median=True, x='poke', y=f'{y}_q{quantile}', **kwargs)


def plot_qd_exploration(args, level, key_columns, res_file, x="iterations", y="qd_score", **kwargs):
    # what method was used to initialize the archive
    def filter(df):
        df = df[df.method.str.lower() == args.registration]
        df = df[df.seed.isin(args.seed)]
        df = df[df.poke.isin(args.poke)]
        return df

    save_path = os.path.join(cfg.DATA_DIR, f"img/{x}_{y}_{level.name.lower()}_{args.poke}.pdf")
    plot_icp_results(filter=filter, icp_res_file=res_file, key_columns=key_columns, logy=y != "qd_score", logx=False,
                     keep_lowest_y_quantile=1.0, x=x, y=y, hue="qd_method", fmt="line", x_starts_at_0=True,
                     save_path=save_path, **kwargs)

    # additionally plot the ratio per seed
    fullname = os.path.join(cfg.DATA_DIR, res_file)
    df = filter(pd.read_pickle(fullname))
    # for each seed, batch, and poke, get the ratio of CMAMEGA / CMAME value
    d1 = df[df.qd_method == "CMAME"]
    d2 = df[df.qd_method == "CMAMEGA"]
    keys = ["method", "seed", "poke", "iterations"]
    d3 = d2.merge(d1, on=keys, how='left')
    # x is CMAMEGA values
    d3[y] = d3[f"{y}_x"] / d3[f"{y}_y"]

    fig = plt.figure(figsize=(5, 3))
    res = sns.lineplot(data=d3, x=x, y=y, estimator=np.median, errorbar=("pi", 50), )
    if y in pretty_prints:
        res.set_ylabel(pretty_prints[y])
    if x in pretty_prints:
        res.set_xlabel(pretty_prints[x])
    # draw the equality line
    res.axhline(1.0, ls='--', c='grey', alpha=0.5)

    res.set(xlim=(0, 975))
    res.legend(["CMAMEGA / CMAME ratio"])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f"{save_path[:-4]}_ratio{save_path[-4:]}")
    plt.close(fig)
