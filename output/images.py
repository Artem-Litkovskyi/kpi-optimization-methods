import os.path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm


DPI = 200
IMAGES_DIR = 'images'


def search_path(
        func, real_target, path_points, title, subdir, filename,
        levels_n=7, pixels_per_unit=256,
        pad_big_x=0.25, pad_big_y=0.25, pad_small_x=0.05, pad_small_y=0.05,
        constraints=None, constrained_target=None
):
    path_xs = [p[0] for p in path_points]
    path_ys = [p[1] for p in path_points]

    # Calculate plot limits
    big_xlim = (min(*path_xs, real_target[0]) - pad_big_x, max(*path_xs, real_target[0]) + pad_big_x)
    big_ylim = (min(*path_ys, real_target[1]) - pad_big_y, max(*path_ys, real_target[0]) + pad_big_y)
    if constrained_target is not None:
        big_xlim = (min(big_xlim[0], constrained_target[0] - pad_big_x), max(big_xlim[1], constrained_target[0] + pad_big_x))
        big_ylim = (min(big_ylim[0], constrained_target[1] - pad_big_y), max(big_ylim[1], constrained_target[1] + pad_big_y))

    approx_target = path_points[-1]
    small_xlim=(approx_target[0] - pad_small_x, approx_target[0] + pad_small_x)
    small_ylim=(approx_target[1] - pad_small_y, approx_target[1] + pad_small_y)

    # Make data
    resolution_x = int((big_xlim[1] - big_xlim[0]) * pixels_per_unit)
    resolution_y = int((big_ylim[1] - big_ylim[0]) * pixels_per_unit)
    x, y = np.meshgrid(np.linspace(*big_xlim, resolution_x), np.linspace(*big_ylim, resolution_y))
    z = func(x, y)
    levels = np.linspace(np.min(z), np.max(z), levels_n)

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for i in range(2):
        # Contour
        ax[i].contour(x, y, z, levels=levels, alpha=0.25)

        # Feasible region
        if constraints is not None:
            ax[i].imshow(
                np.logical_not(np.logical_and.reduce([c(x, y) <= 0 for c in constraints])).astype(int),
                extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='Greys', alpha=0.3
            )

        # Search path
        ax[i].plot(path_xs, path_ys, marker='.', color='k')

        # Targets
        ax[i].plot(*real_target, 'rx')
        ax[i].plot(*approx_target, marker='+', mec='limegreen', ms=8)
        if constrained_target is not None:
            ax[i].plot(*constrained_target, 'bx')

        # Zoom for left plot
        if i == 0:
            ax[i].set_aspect('equal', 'box')
        # Zoom for right plot
        else:
            ax[i].set_aspect('equal')
            ax[i].set(xlim=small_xlim, ylim=small_ylim)

    # Decorations
    plt.suptitle(title)
    handles = [Line2D([], [], color='r', marker='x', linestyle='None', label='Minimum')]
    if constrained_target is not None:
        handles.append(Line2D([], [], color='b', marker='x', linestyle='None', label='Minimum inside the region'))
    handles.append(Line2D([], [], color='limegreen', marker='+', ms=8, linestyle='None', label='Approximated minimum'))
    plt.legend(handles=handles)

    # save and close
    dir_path = os.path.join(IMAGES_DIR, subdir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, 'path_' + filename), dpi=DPI)

    plt.close()


def surface(func, start_point, real_target, xlim, ylim, title, subdir, filename):
    # make data
    x, y = np.meshgrid(
        np.linspace(*xlim, num=256),
        np.linspace(*ylim, num=256)
    )

    z = func(x, y)

    # plot
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={'projection': '3d', 'computed_zorder': False})
    ax.plot_surface(x, y, z, cmap=cm.inferno, zorder=1)

    start_f = func(*start_point)
    ax.scatter(start_point[0], start_point[1], zs=start_f, marker='.', color='k', zorder=2)
    ax.text(start_point[0], start_point[1], start_f+0.2, 'Start', zorder=2)

    real_f = func(*real_target)
    ax.plot([real_target[0], real_target[0]], ylim, zs=[real_f, real_f], color='r', zorder=0)
    ax.plot(xlim, [real_target[1], real_target[1]], zs=[real_f, real_f], color='r', zorder=0)
    ax.text(
        real_target[0], real_target[1]-0.3, real_f, 'Minimum', color='r',
        horizontalalignment='right', verticalalignment='top', zorder=0
    )
    ax.set(xlim=xlim, ylim=ylim, zlim=(real_f, max(map(max, z))))
    ax.view_init(elev=30, azim=20)

    fig.tight_layout()

    # decorations
    plt.suptitle(title)

    # save and close
    dir_path = os.path.join(IMAGES_DIR, subdir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, 'surface_' + filename), dpi=DPI)

    plt.close()


def calls_and_deviation(
        values, calls, deviations,
        change_param, old_v, new_v,
        title, subdir, filename,
        min_calls=None, max_calls=None, min_dev=None, max_dev=None
):
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax[0].plot(values, calls, color='black')
    ax[0].vlines((old_v, new_v), min(calls), max(calls), colors=['gray', 'green'], linestyles=['--', '-'])
    ax[0].set_xlabel(change_param)
    ax[0].set_ylabel('calls')

    if min_calls is not None:
        ax[0].set_ylim(bottom=min_calls)

    if max_calls is not None:
        ax[0].set_ylim(top=max_calls)

    ax[1].plot(values, deviations, color='black')
    ax[1].vlines((old_v, new_v), min(deviations), max(deviations), colors=['gray', 'green'], linestyles=['--', '-'])
    ax[1].set_xlabel(change_param)
    ax[1].set_ylabel('$|| f(X_{min}) - f(X^*) ||$')

    if min_dev is not None:
        ax[0].set_ylim(bottom=min_dev)

    if max_dev is not None:
        ax[1].set_ylim(top=max_dev)

    # decorations
    plt.suptitle(title)
    ax[1].legend(
        [Line2D([0], [0], color='gray'),
         Line2D([0], [0], color='green')],
        ['old', 'new'],
        loc='upper right'
    )

    # save and close
    dir_path = os.path.join(IMAGES_DIR, subdir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig(os.path.join(dir_path, 'plot_' + filename), dpi=DPI)

    plt.close()