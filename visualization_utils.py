import string
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np


def output_to_binary(prediction, threshold, mode='sample'):
    if mode == 'sample':
        axis = 0
    elif mode == 'batch':
        axis = 1
    else:
        raise ValieError('Wrong mode argument!')
        
    e = prediction
    e = e - e.max(axis=axis, keepdims=True)
    e = np.exp(e)
    e = e / np.sum(e, axis=axis, keepdims=True)
    
    if axis == 0:
        return np.where(e[1] > threshold, 1, 0).astype(np.uint8)
    elif axis == 1:
        return np.where(e[:, 1] > threshold, 1, 0).astype(np.uint8)

    
def make_colored_diff(gt, pred, threshold=None, path=None):
    if np.issubdtype(pred.dtype, np.floating):
        pred = output_to_binary(pred, threshold)
         
    red_mask = np.where((gt == 1) & (pred == 0), 255, 0)
    blue_mask = np.where((gt == 0) & (pred == 1), 255, 0)
    valid_mask = np.where((gt == 1) & (pred == 1), 255, 0)[:, :, np.newaxis]

    colored_image = np.concatenate([valid_mask, valid_mask, valid_mask], axis=2)
    colored_image[:, :, 0] += red_mask
    colored_image[:, :, 2] += blue_mask

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(colored_image)

    if path is not None:
        plt.savefig(path)
    


def plot_sample(sample, mask, predicted, threshold, metrics, fig_path=None):
    predicted_mask = output_to_binary(predicted, threshold)
        
    plt.figure(figsize=(30,10))
    
    plt.subplot(1, 3, 1)
    plt.title('image')
    plt.imshow(sample, 'gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('pred')
    plt.imshow(predicted_mask, 'gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('gt')
    plt.imshow(mask, 'gray')
    plt.axis('off')
    
    if fig_path is not None:
        plt.savefig(fig_path)
    
    plt.show()
    for k, v in metrics.items():
        print('{:12}: {}'.format(k, v(mask[None, :], predicted[None, :])))
        
        
def plot_polar(
        title, 
        models, 
        stacks, 
        metrics_th, 
        data,
        stacks_name_mapping,
        models_name_mapping,
        markers,
        h, w):
    
    plt.style.use('seaborn-white')
    subplot_kw = {'projection': 'polar'}
    gridspec_kw = {'left': 0.1, 'right': 0.9, 'bottom': 0.05, 'top': 0.95,
                   'wspace': 0.1, 'hspace': 0.075}

    fig_kw = {'figsize': (10 * w, 10 * h)}
    title_properties = {'weight': 'semibold', 'size': 30}
    legend_properties = {'weight': 'semibold', 'size': 30}
    ylabels_properties = {'weight': 'semibold', 'size': 16}
    xlabels_properties = {'weight': 'semibold', 'size': 22}
    
    fig, axes = plt.subplots(h, w, #sharey='row', #sharex='col',
                             subplot_kw=subplot_kw,
                             gridspec_kw=gridspec_kw,
                             **fig_kw)
    
    axes = list(chain(*axes))
    
    for i, (stack, axis) in enumerate(zip(sorted(stacks, key=lambda item: stacks_name_mapping[item]), axes)):
        
        results = data[data['stack'] == stack]
        
        x = list(np.linspace(0, 2 * np.pi, len(results) + 1))
        axis.set_xticks(x)
        labels = [models_name_mapping[item] for item in results['model']]
#         axis.set_thetagrids(np.linspace(0, 360, len(results) + 1), frac=1.5)
        axis.tick_params(axis='x', pad=10)
        axis.set_xticklabels(labels, 
                             fontdict=xlabels_properties)
        
        for j, metric_name in enumerate(metrics_th):
            y = list(results[metric_name])
#             x.append(x[0])
            axis.plot(x, y + [y[0]],
                      linestyle='-',
                      marker=markers[j],
                      label=metric_name, 
                      markersize=15,
                      linewidth=4) #, label=stacks_name_mapping[k], linewidth=3)
#             items = sorted(results.items(), key=lambda item: stacks_name_mapping[item[0]])
#             for k, v in items:
#                 x, y = v[metric_name]
#                 axis.plot(x, y, label=stacks_name_mapping[k], linewidth=3)
#             axis.set_xlim([0, 1])
#             axis.set_ylim([0, 1.05])

#             plt.xticks(np.arange(len(y) - 1), results['Stack'])
           
#             axis.set_rticks(np.arange(0.8, 1.0, 21))
            
            
            
#             axis.set_yticks([0] + list(np.arange(0.05, 0.951, 0.1)) + [1])
#             axis.tick_params(labelsize=18)
#             axis.set_xlabel('Threshold', fontsize=22)
#             axis.set_ylabel(metric_name[0].upper() + metric_name[1:], fontsize=22)
        axis.set_rmin(0.7)
        axis.set_rmax(1)
        axis.set_rticks(np.linspace(0.8, 1, 9))
        axis.set_yticklabels([round(x, 3) for x in np.linspace(0.8, 1, 9)], 
                             fontdict=ylabels_properties)
        if i == len(stacks) - 1:
            axis.legend(loc='lower right', 
                        bbox_to_anchor=(0, 0, 1.8, 1.8), 
                        prop=legend_properties)
#         else:
#             axis.legend(loc='center', bbox_to_anchor=(0, 0, 0.7, 0.7), fontsize=16)
        axis.set_title('{}) Models quality on {}'.format(string.ascii_letters[i], 
                                                         stacks_name_mapping[stack]),
                       fontdict=legend_properties)

    if len(stacks) < len(axes):
        axes[-1].set_visible(False)

#     fig.suptitle('Models quality on diffenent stacks', fontdict=title_properties) 
#     fig.text(0.5, 0.025, 'Threshold', ha='center', va='center', fontsize=40)
#     fig.text(0.05, 0.5, metric_name.upper(), 
#              ha='center', va='center', rotation=90, fontsize=40)

    plt.savefig('./article_results/polar_plot_{}.jpg'.format(title))
    plt.show()
    

def plot_thresholded_metric(models, metrics_th, h, w):
    plt.style.use('seaborn-white')
    fig_kw = {'figsize': (10 * w, 10 * h)}
    gridspec_kw = {#'nrows': h, 'ncols': 2 * w,
                   'left': 0.1, 'right': 0.9, 'bottom': 0.05, 'top': 0.95,
                   'wspace': 0.025, 'hspace': 0.1}
    suptitle_properties = {'weight': 'semibold', 'size': 50}
    title_properties = {'weight': 'semibold', 'size': 40}
    legend_properties = {'weight': 'semibold', 'size': 30}
    labels_properties = {'weight': 'semibold', 'size': 20}
#     gs = gridspec.GridSpec()
#     gs.update(left=0.1, right=0.9, bottom=0.05, top=0.95)
    
#     fig = plt.figure(figsize=(10 * w , 10 * h))
    fig, axes = plt.subplots(h, w, sharey='row', #sharex='col', 
                             gridspec_kw=gridspec_kw, **fig_kw)
    
    axes = list(chain(*axes))
    for i, (fname, axis) in enumerate(zip(models, axes)):
        results = load('../output_data/' + fname)
            
        axis.set_xlim([0, 1])
        axis.set_ylim([0, 1.05])
        axis.set_xticks([round(item, 2) for item in np.arange(0.05, 0.951, 0.1)])
        axis.set_xticklabels([round(item, 2) for item in np.arange(0.05, 0.951, 0.1)],
                             fontdict=labels_properties)
        axis.set_yticks([0] + [round(item, 2) for item in np.arange(0.05, 0.951, 0.1)] + [1])
        axis.set_yticklabels([0] + [round(item, 2) for item in np.arange(0.05, 0.951, 0.1)] + [1],
                             fontdict=labels_properties)
#             axis.tick_params(fontdict=ylabels_properties)
        axis.set_title(models_name_mapping[fname.split('.')[0]], fontsize=24)
    
        for metric_name in metrics_th:
            items = sorted(results.items(), key=lambda item: stacks_name_mapping[item[0]])
#             print(type(items))
            for k, v in items:
#                 print(k, v)
                x, y = v[metric_name]
                axis.plot(x, y, 
                          label=stacks_name_mapping[k], 
                          linewidth=4)
        
        if i == len(models) - 1:
            axis.legend(loc='lower right', 
                        bbox_to_anchor=(0, 0, 1.8, 1.8), 
                        prop=legend_properties)
        axis.set_title('{}) {} {} from threshold'.format(string.ascii_letters[i], 
                                       models_name_mapping[fname.split('.')[0]],
                                                        metrics_th[0]),
                       fontdict=legend_properties)
#             axis.set_xlabel('Threshold', fontsize=22)
#             axis.set_ylabel(metric_name[0].upper() + metric_name[1:], fontsize=22)
    if len(models) < len(axes):
        axes[-1].set_visible(False)
#     gs.tight_layout(fig)

#     fig=axes[0,0].figure\
#     fig = axes[0].figure

#     fig.suptitle('IOU from threshold dependence for all models', 
#                  fontdict=suptitle_properties) 
    fig.text(0.5, 0.025, 'Threshold', 
             fontdict=title_properties,
             ha='center', va='center')
    fig.text(0.05, 0.5, metric_name.upper(), 
             fontdict=title_properties,
             ha='center', va='center', rotation=90)

    plt.savefig('../output_data/results/metrics_plot_{}_{}.jpg'
                .format('_'.join(models), '_'.join(metrics_th)))
    plt.show()