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

        # Get most repeated class
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        # Saving to object, predict class with most frequency
        self.yhat = yhat

        # Save number of observations in node
        self.n = len(Y)

        # Set left and right nodes as empty
        self.left = None
        self.right = None

        # Default splits
        self.best_feature = None
        self.best_value = None

        @staticmethod
        def GINI_impurity(y1_count: int, y2_count: int) -> float:
            """
            Calculate GINI impurity for a given observation belonging to a binary class
            """
            # Count check
            if y1_count is None:
                y1_count = 0
            if y2_count is None:
                y2_count = 0

            # Getting total obs
            n = y1_count + y2_count

            # If n is 0 then the lowest possible gini impurity is returned 0.0
            if n == 0:
                return 0.0

            # Get probability pertaining to each class
            # Observation / total number of observations
            p1 = y1_count / n
            p2 = y2_count / n

            # Return GINI score
            return 1 - (p1 ** 2 + p2 ** 2)

