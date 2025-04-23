import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm


DPI = 200
IMAGES_DIR = 'images'


if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)


def search_path(
        func, real_target, path_points, title, filename,
        levels_n=7, pad_big=0.25, pad_small=0.05
):
    path_xs = [p[0] for p in path_points]
    path_ys = [p[1] for p in path_points]

    # make data for contour
    x, y = np.meshgrid(
        np.linspace(min(path_xs)-pad_big, max(path_xs)+pad_big, 256),
        np.linspace(min(path_ys)-pad_big, max(path_ys)+pad_big, 256)
    )

    z = func(x, y)
    levels = np.linspace(np.min(z), np.max(z), levels_n)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for i in range(2):
        ax[i].contour(x, y, z, levels=levels, alpha=0.25)
        ax[i].plot(path_xs, path_ys, marker='.', color='k')
        ax[i].plot(*real_target, 'rx')
        if i == 0:
            ax[i].set_aspect('equal', 'box')
        else:
            ax[i].set_aspect('equal')
            ax[i].set(
                xlim=(real_target[0]-pad_small, real_target[0]+pad_small),
                ylim=(real_target[1]-pad_small, real_target[1]+pad_small)
            )

    # decorations
    plt.suptitle(title)
    plt.legend(handles=[
        mlines.Line2D([], [], color='r', marker='x', linestyle='None', label='Actual minimum')
    ])

    plt.savefig(os.path.join(IMAGES_DIR, 'path_' + filename), dpi=DPI)


def surface(func, start_point, real_target, xlim, ylim, title, filename):
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

    plt.savefig(os.path.join(IMAGES_DIR, 'surface_' + filename), dpi=DPI)


def calls_and_deviation(
        values, calls, deviations,
        change_param, old_v, new_v,
        title, filename
):
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax[0].plot(values, calls, color='black')
    ax[0].vlines((old_v, new_v), min(calls), max(calls), colors=['gray', 'green'], linestyles=['--', '-'])
    ax[0].set_xlabel(change_param)
    ax[0].set_ylabel('calls')

    ax[1].plot(values, deviations, color='black')
    ax[1].vlines((old_v, new_v), min(deviations), max(deviations), colors=['gray', 'green'], linestyles=['--', '-'])
    ax[1].set_xlabel(change_param)
    ax[1].set_ylabel('deviation')

    # decorations
    plt.suptitle(title)
    ax[1].legend(
        [mlines.Line2D([0], [0], color='gray'),
         mlines.Line2D([0], [0], color='green')],
        ['old', 'new'],
        loc='upper right'
    )

    plt.savefig(os.path.join(IMAGES_DIR, 'plot_' + filename), dpi=DPI)