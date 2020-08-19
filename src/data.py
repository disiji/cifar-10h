from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from models import TemperatureScaling, TemperatureScalingSoftLabels
from plot import plot_reliability_diagram, plot_reliability_diagram_with_soft_labels

DIR = '/home/disij/projects/cifar-10h/data/'

class Dataset:
    def __init__(self,
                 y: np.ndarray,
                 human_counts: np.ndarray,
                 s_human: np.ndarray,
                 logits_model: np.ndarray,
                 dataset_name: str) -> None:
        self.y = y
        self.human_counts = human_counts
        self.s_human = s_human
        self.logits_model = logits_model
        self.s_model = softmax(logits_model, axis=1)

        self.y_human = np.argmax(s_human, axis=1)
        self.y_model = np.argmax(self.s_model, axis=1)
        self.dataset_name = dataset_name
        self.n, self.k = self.s_model.shape
        self.indices = np.arange(self.n)

        self.calibrated_logits_human = None
        self.calibrated_logits_model = None
        self.calibrated_s_human = None
        self.calibrated_s_model = None

        self.train_idx = None
        self.eval_idx = None

        
    def shuffle(self, random_state=0) -> None:
        shuffle_ids = np.arange(self.n)
        shuffle_ids = shuffle(shuffle_ids, random_state=random_state)
        self.y = self.y[shuffle_ids]
        self.human_counts = self.human_counts[shuffle_ids]
        self.s_human = self.s_human[shuffle_ids]
        self.s_model = self.s_model[shuffle_ids]
        self.indices = self.indices[shuffle_ids]
        self.y_human = self.y_human[shuffle_ids]
        self.y_model = self.y_model[shuffle_ids]

    def split_train_eval(self, ratio=0.8, random_state=0):
        shuffle_ids = np.arange(self.n)
        shuffle_ids = shuffle(shuffle_ids, random_state=random_state)
        val = int(self.n * ratio)
        self.train_idx = shuffle_ids[:val]
        self.eval_idx = shuffle_ids[val:]
        
    @classmethod
    def load_from_text(cls, dataset_name: str) -> 'Dataset':
        if dataset_name == 'cifar10':
            human_counts = np.load(DIR + 'cifar10h-counts.npy')
            s_human = np.load(DIR + 'cifar10h-probs.npy')
            
            array = np.genfromtxt(DIR + 'cifar10_resnet_small_logits.txt', delimiter=',')
            logits_model = array[:, 1:].astype(np.float)
            y = array[:, 0].astype(np.int)
  
        return cls(y, human_counts, s_human, logits_model, dataset_name)


    def calibration(self, calibration_type):
        if calibration_type == 'temperature_scaling':
            print('\nTraining and apply temperature scaling to human predictor...')
            temperature_model = TemperatureScaling()
            temperature_model.set_temperature(np.log(self.s_human+1e-10)[self.train_idx], 
                                                                  self.y[self.train_idx])
            self.calibrated_logits_human = temperature_model(np.log(self.s_human)).cpu().data.numpy() 
            self.calibrated_s_human = softmax(self.calibrated_logits_human, axis=1)
            
            print('Training and apply temperature scaling to model predictor...')
            temperature_model = TemperatureScaling()
            temperature_model.set_temperature(self.logits_model[self.train_idx], 
                                                         self.y[self.train_idx])
            self.calibrated_logits_model = temperature_model(self.logits_model).cpu().data.numpy() 
            self.calibrated_s_model = softmax(self.calibrated_logits_model, axis=1)

            
    def calibration_to_distribution(self, calibration_type):
        
        if calibration_type == 'temperature_scaling':
            print('\nTraining and apply temperature scaling to human predictor...')
            temperature_model = TemperatureScalingSoftLabels()
            temperature_model.set_temperature(np.log(self.s_human+1e-10)[self.train_idx], 
                                                            self.s_human[self.train_idx])
            self.calibrated_logits_human = temperature_model(np.log(self.s_human)).cpu().data.numpy() 
            self.calibrated_s_human = softmax(self.calibrated_logits_human, axis=1)
            
            print('Training and apply temperature scaling to model predictor...')
            temperature_model = TemperatureScalingSoftLabels()
            temperature_model.set_temperature(self.logits_model[self.train_idx], 
                                                   self.s_human[self.train_idx])
            self.calibrated_logits_model = temperature_model(self.logits_model).cpu().data.numpy() 
            self.calibrated_s_model = softmax(self.calibrated_logits_model, axis=1)


    def plot_reliability(self, plot_split:str='eval'):
        # plot_split = train, eval, full
        idx = {
            'train': self.train_idx,
            'eval': self.eval_idx,
            'full': np.arange(self.n)
        }[plot_split]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1 = plot_reliability_diagram(ax1, self.s_human[idx], self.y[idx])
        ax1.set_title('a group of humans', fontsize=12)
        ax2 = plot_reliability_diagram(ax2, self.s_model[idx], self.y[idx])
        ax2.set_title('a small ResNet', fontsize=12)

        ax1 = plot_reliability_diagram_with_soft_labels(ax1, self.s_human[idx], self.s_human[idx])
        ax1.set_title('a group of humans', fontsize=12)
        ax2 = plot_reliability_diagram_with_soft_labels(ax2, self.s_model[idx], self.s_human[idx])
        ax2.set_title('a small ResNet', fontsize=12)

        ax1.legend(fontsize=12, loc = 'upper left')
        ax2.legend(fontsize=12, loc = 'upper left')   
        
        return fig, (ax1, ax2)
    
    def plot_calibrated_reliability(self, plot_split:str='eval'):
        # plot_split = train, eval, full
        idx = {
            'train': self.train_idx,
            'eval': self.eval_idx,
            'full': np.arange(self.n)
        }[plot_split]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        ax1 = plot_reliability_diagram(ax1, self.calibrated_s_human[idx], self.y[idx])
        ax1.set_title('a group of humans', fontsize=12)
        ax2 = plot_reliability_diagram(ax2, self.calibrated_s_model[idx], self.y[idx])
        ax2.set_title('a small ResNet', fontsize=12)

        ax1 = plot_reliability_diagram_with_soft_labels(ax1, self.calibrated_s_human[idx], self.s_human[idx])
        ax1.set_title('a group of humans', fontsize=12)
        ax2 = plot_reliability_diagram_with_soft_labels(ax2, self.calibrated_s_model[idx], self.s_human[idx])
        ax2.set_title('a small ResNet', fontsize=12)

        ax1.legend(fontsize=12, loc = 'upper left')
        ax2.legend(fontsize=12, loc = 'upper left')
        
        return fig, (ax1, ax2)