import pandas as pd
import numpy as np

# value calculator
from collections import Counter

class Node:
    """
    A single node in a decision tree accounting for
    > Gini Impurity Score
    > Number of observations
    > Number of observations belonging to binary target classes {0, 1}
    > Feature matrix X
    """
    def __init__(self, Y: list, X:pd.DataFrame, min_samples_split=None, max_depth=None, depth=None, node_type=None, rule=None):
        # Local instances of param data
        self.Y = Y
        self.X = X

        # Local hyper params
        self.min_samples_split = min_samples_split if min_samples_split else 20 # Defaults to 20 if not specified by operator
        self.max_depth = max_depth if max_depth else 5 # Defaults to 5 if not specified by operator

        # Current depth
        self.depth = depth if depth else 0

        # Extracting features
        self.features = list(self.X.columns) # Extract actual strings for column titles
        
        # Type of node
        self.node_type = node_type if node_type else "root"

        # Rule for splitting
        self.rule = rule if rule else ""

        # Calculate occurences of Y distribution in node
        self.counts = Counter(Y)

        # GINI impurity on Y
        self.gini_impurity = self.get_GINI()

        # Sorting counts and saving final prediction
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))
