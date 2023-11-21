"""Decision Tree Helpers."""
import math
import graphviz
from colorama import Style, Fore
from helpers import is_numeric
from load_data import HEADERS

class Question:
    """A Question is used to partition a dataset."""
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        """Compare the feature value in an example to the feature value in this question."""
        val = example[self.column]
        return val >= self.value if is_numeric(val) else val == self.value

    def __repr__(self):
        condition = ">=" if is_numeric(self.value) else "=="
        return f"Is {HEADERS[self.column]} {condition} {str(self.value)}"

class Leaf:
    """ A Leaf node classifies data. """
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class DecisionNode:
    """ A Decision Node asks a question. """
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}
    for row in rows:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    return counts

def partition(rows, question):
    """Partitions a dataset."""
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def classify(row, node):
    """Classifies a row using the tree."""
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    return classify(row, node.false_branch)

def entropy(rows):
    """Returns the entropy of the rows."""
    counts = class_counts(rows)
    impurity = 0
    base = float(len(rows))
    if base <= 1:
        return 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / base
        impurity -= prob_of_lbl * math.log(prob_of_lbl, 2)
    return impurity

def info_gain(left, right, current_uncertainty):
    """Returns Information Gain."""
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

def info_gain_ratio(left, right, current_uncertainty):
    """Returns Information Gain Ratio."""
    p = float(len(left)) / (len(left) + len(right))
    return (current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)) / intrinsic_value(left, right)

def intrinsic_value(left, right):
    """Returns the intrinsic value."""
    total = len(left) + len(right)
    p_left = len(left) / total
    p_right = len(right) / total
    return -(p_left * math.log(p_left, 2) + p_right * math.log(p_right, 2))

def find_best_split(rows, igr):
    """Find the best question to ask by iterating over every feature / value and calculate the information gain or information gain ratio."""
    best_gain = 0
    best_question = None
    current_uncertainty = entropy(rows)
    for col in range(len(rows[0]) - 2):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            if igr:
                gain = info_gain_ratio(true_rows, false_rows, current_uncertainty)
            else:
                gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

def build_tree(rows, igr):
    """Builds the tree."""
    gain, question = find_best_split(rows, igr)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows, igr)
    false_branch = build_tree(false_rows, igr)

    return DecisionNode(question, true_branch, false_branch)

def print_leaf(counts):
    """Prints the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {lbl: f"{int(counts[lbl] / total * 100)}%" for lbl in counts}
    return probs

def print_tree(node, spacing=""):
    """Prints the tree to the console."""
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + str(node.question))

    print(f"{Style.DIM}{spacing}'--> True:'{Style.NORMAL}")
    print_tree(node.true_branch, spacing + "  ")

    print(f"{Style.DIM}{spacing}'--> False:'{Style.NORMAL}")
    print_tree(node.false_branch, spacing + "  ")

def draw_tree(node, dot=None, parent=None, edge_label=None):
    """Returns a graphviz Digraph object with the tree drawn."""
    if dot is None:
        dot = graphviz.Digraph(comment='Decision Tree')

    if isinstance(node, Leaf):
        predictions = print_leaf(node.predictions)
        for lbl, prob in predictions.items():
            dot.node(str(id(node)), f"{lbl}\n{prob}")
        if parent is not None:
            dot.edge(str(id(parent)), str(id(node)), label=edge_label)
        return dot

    dot.node(str(id(node)), str(node.question))
    if parent is not None:
        dot.edge(str(id(parent)), str(id(node)), label=edge_label)

    draw_tree(node.true_branch, dot, node, 'True')
    draw_tree(node.false_branch, dot, node, 'False')

    return dot

def export_tree(dot):
    """Exports the tree to a png file."""
    dot.render('output/tree', format='png')
    print(f"{Style.BRIGHT}{Fore.GREEN}!!! Tree exported to output/tree.png !!!{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.GREEN}!!! Tree exported to output/tree.png !!!{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{Fore.GREEN}!!! Tree exported to output/tree.png !!!{Style.RESET_ALL}")
