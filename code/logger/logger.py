import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import os
import jittor as jt 

def plot_loss_log(loss_log, epoch, loss_dir):
    # 改进点：使用数组实际长度，防止维度不匹配导致崩溃
    for key in loss_log.keys():
        current_data = np.array(loss_log[key])
        actual_epochs = len(current_data)
        axis = np.linspace(1, actual_epochs, actual_epochs)
        
        label = '{} Loss'.format(key)
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, current_data, label=label)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        # 保持原逻辑存为 pdf
        plt.savefig(os.path.join(loss_dir, 'loss_{}.pdf'.format(key)))
        plt.close(fig)

def plot_psnr_log(psnr_log, epoch, psnr_dir):
    # psnr_log 通常是一个 list 或者 numpy array
    current_data = np.array(psnr_log)
    actual_epochs = len(current_data)
    axis = np.linspace(1, actual_epochs, actual_epochs)
    
    label = 'PSNR'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, current_data, label=label)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(psnr_dir, 'psnr.pdf'))
    plt.close(fig)