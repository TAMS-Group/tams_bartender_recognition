#!/usr/bin/env python

import numpy as np
import glob
import os
from shutil import copy2

train_dir = 'labels_train/'
test_dir = 'labels_test/'
split_ratio = 0.3

if __name__=="__main__":

    # iterate labels
    for label_dir in glob.glob('labels/*'):
        label = label_dir.split('/')[1]
        samples = glob.glob(label_dir+'/*.png')
        
        # split randomly into test set
        test_samples = np.random.choice(len(samples), int(split_ratio * len(samples)), replace=False)

        # copy samples to target directories
        for i, sample in enumerate(samples):
            target_dir = test_dir if i in test_samples else train_dir
            target_dir += label + '/'

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            copy2(sample, target_dir)
