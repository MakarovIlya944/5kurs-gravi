import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
  """
  Create a heatmap from a numpy array and two lists of labels.

  Parameters
  ----------
  data
      A 2D numpy array of shape (N, M).
  row_labels
      A list or array of length N with the labels for the rows.
  col_labels
      A list or array of length M with the labels for the columns.
  ax
      A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
      not provided, use current axes or create a new one.  Optional.
  cbar_kw
      A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
  cbarlabel
      The label for the colorbar.  Optional.
  **kwargs
      All other arguments are forwarded to `imshow`.
  """

  if not ax:
    ax = plt.gca()

  # Plot the heatmap
  im = ax.imshow(data, **kwargs)

  # Create colorbar
  cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
  cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

  # We want to show all ticks...
  ax.set_xticks(np.arange(data.shape[1]))
  ax.set_yticks(np.arange(data.shape[0]))
  # ... and label them with the respective list entries.
  ax.set_xticklabels(col_labels)
  ax.set_yticklabels(row_labels)

  # Let the horizontal axes labeling appear on top.
  ax.tick_params(top=True, bottom=False,
                  labeltop=True, labelbottom=False)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
            rotation_mode="anchor")

  # Turn spines off and create white grid.
  for edge, spine in ax.spines.items():
      spine.set_visible(False)

  ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
  ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
  ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
  ax.tick_params(which="minor", bottom=False, left=False)

  return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
  """
  A function to annotate a heatmap.

  Parameters
  ----------
  im
      The AxesImage to be labeled.
  data
      Data used to annotate.  If None, the image's data is used.  Optional.
  valfmt
      The format of the annotations inside the heatmap.  This should either
      use the string format method, e.g. "$ {x:.2f}", or be a
      `matplotlib.ticker.Formatter`.  Optional.
  textcolors
      A list or array of two color specifications.  The first is used for
      values below a threshold, the second for those above.  Optional.
  threshold
      Value in data units according to which the colors from textcolors are
      applied.  If None (the default) uses the middle of the colormap as
      separation.  Optional.
  **kwargs
      All other arguments are forwarded to each call to `text` used to create
      the text labels.
  """

  if not isinstance(data, (list, np.ndarray)):
    data = im.get_array()

  # Normalize the threshold to the images color range.
  if threshold is not None:
    threshold = im.norm(threshold)
  else:
    threshold = im.norm(data.max())/2.

  # Set default alignment to center, but allow it to be
  # overwritten by textkw.
  kw = dict(horizontalalignment="center",
            verticalalignment="center")
  kw.update(textkw)

  # Get the formatter in case a string is supplied
  if isinstance(valfmt, str):
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

  # Loop over the data and create a `Text` for each "pixel".
  # Change the text's color depending on the data.
  texts = []
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
      text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
      texts.append(text)

  return texts

def func(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

def heatmaps(coords, true, pred, reverse, label='value', save_filename=None):
  num = (1 if not true is None else 0) + (1 if not pred is None else 0) + (1 if not reverse is None else 0)
  ax_all = []
  if num == 1:
    fig, ax = plt.subplots()
    ax_all.append(ax)
  elif num == 2:
    fig, ((ax),(ax2)) = plt.subplots(1, 2)
    ax_all.append(ax)
    ax_all.append(ax2)
  elif num == 3:
    fig, ((ax),(ax2),(ax3)) = plt.subplots(1, 3)
    ax_all.append(ax)
    ax_all.append(ax2)
    ax_all.append(ax3)
  vmax = -1E-10
  vmin = 1E+10
  kx, ky = coords['kx'], coords['ky']
  coords['x'] = range(coords['x'].start,coords['x'].stop,coords['x'].step * kx)
  coords['y'] = range(coords['y'].start,coords['y'].stop,coords['y'].step * ky)

  vmax = [0,0,0]
  vmin = [0,0,0]
  if not true is None:
    vmax[0] = np.max(true)
    vmin[0] = np.min(true)
  if not pred is None:
    vmax[1] = np.max(pred)
    vmin[1] = np.min(pred)
  if not reverse is None:
    vmax[2] = np.max(reverse)
    vmin[2] = np.min(reverse)
  vmax = min(vmax)
  vmin = max(vmin)

  ax_index = 0
  if not true is None:
    vmax = np.max(true)
    vmin = np.min(true)
    im, _ = heatmap(true, coords['x'], coords['y'], ax=ax_all[ax_index],
                  cmap="PuOr", vmin=vmin, vmax=vmax,
                  cbarlabel="true " + label)
    ax_index += 1
    # annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
  if not pred is None:
    vmax = np.max(pred)
    vmin = np.min(pred)
    im, _ = heatmap(pred, coords['x'], coords['y'], ax=ax_all[ax_index],
                  cmap="PuOr", vmin=vmin, vmax=vmax,
                  cbarlabel="pred " + label)
    ax_index += 1
    # annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)
  if not reverse is None:
    vmax = np.max(reverse)
    vmin = np.min(reverse)
    im, _ = heatmap(reverse, coords['x'], coords['y'], ax=ax_all[ax_index],
                  cmap="PuOr", vmin=vmin, vmax=vmax,
                  cbarlabel="reverse " + label)
    # annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

  fig.tight_layout()
  if save_filename:
    plt.savefig(save_filename,dpi=300)
  else:
    plt.show()