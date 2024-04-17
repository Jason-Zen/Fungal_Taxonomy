import os
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_curve, auc, \
    ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize, LabelEncoder


class Evaluation:
    """
    General methods of model evaluation.


    Parameters:
        - clf: Classifier

        - x: pandas.DataFrame
            Training data.

        - y: pandas.DataFrame
            Target value relative to X.

        - algorithm: Name for the classifier. Default is the name of the classifier

    Methods:
        __call__(self, para, path): Pipeline.

        - feature_selection: Select features by SequentialFeatureSelector().

        - grid_search: Select parameters for the classifier by GridSearchCV().

        - cross_validation: Calculate cross validation score by cross_val_score().

        - eva_metrics: Calculate accuracy, precision, recall and F1 score of the model.

    Examples:
        Return the best parameter, accuracy, precision, recall, f1 score and best features
        for RandomForestClassifier fitted with X and y, and generate a confusion matrix to
        the current directory.

            >> evaluator = Evaluation(RandomForestClassifier(), X, y, "RF")
            >> para = {'criterion': ['gini', 'entropy', 'log_loss'], 'min_samples_leaf': [1, 2, 3, 5, 10]}
            >> clf, acc, pre, rec, f1, fea = evaluator(para, os.getcwd())
    """
    def __init__(
            self,
            clf,
            x: pd.DataFrame,
            y,
            algorithm: str = None
    ):
        self.clf = clf
        self.X = x
        self.y = y

        if algorithm is None:
            self.name = clf.__class__.__name__
        else:
            self.name = algorithm

        # Check if it needs numerical label
        self.num_label_model = ["XGBClassifier"]
        if self.clf.__class__.__name__ in self.num_label_model:
            label_encoder = LabelEncoder()
            self.y = pd.Series(label_encoder.fit_transform(y))

    def __call__(self, para, path):
        # Select features
        fea = "Unselected"  # if manually selected with correlation coefficient, feature_selection() is not necessary
        # Grid search
        best_clf = self.clf = self.grid_search(para=para)
        # Metrics
        acc, pre, rec, f1 = self.eva_metrics(path)

        return best_clf, acc, pre, rec, f1, fea

    def feature_selection(self):
        print(" <]>- Selecting features ...")
        fea_1 = self.X.columns.values
        _sfs = SequentialFeatureSelector(
            self.clf,
            n_features_to_select='auto',
            tol=0.01
        )
        fea_2 = _sfs.fit_transform(self.X, self.y).columns.values

        fea_out = [item for item in fea_1 if item not in fea_2]
        return fea_out  # header names

    def grid_search(self, para):
        print(" <]>- Grid searching ...")
        _gs = GridSearchCV(
            self.clf,
            para,
            verbose=0,  # the number of displaying information
            scoring='f1_weighted',  # the name for scores
            cv=5  # the number of folds for StratifiedKFold (if int)
        )
        _gs.fit(self.X, self.y)
        return _gs.best_estimator_  # trained model

    def cross_validation(self):
        print(" <]>- Cross validating ...")
        _cvs = cross_val_score(
            self.clf,
            self.X,
            self.y,
            cv=5  # the number of folds for StratifiedKFold (if int)
        )
        return _cvs  # list of accuracy

    def eva_metrics(
            self,
            path,
            n_splits: int = 10,  # Number of re-shuffling & splitting iterations
            test_size: float = 0.2,  # Proportion of the dataset in the test split, between (0, 1)
            fold: int = 0,  # The index of shuffling & splitting iterations, between (0, n_splits)
            average: str = 'weighted'  # ['weighted', 'micro', 'macro']
    ):
        print(" <]>- Evaluating model metrics ...")
        # Split training and testing sets
        _sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=None)

        _train_index = []
        _test_index = []
        for i_train, i_test in _sss.split(self.X, self.y):
            _train_index.append(i_train)
            _test_index.append(i_test)

        # Create lists to save results from different shuffles
        y_t = []
        y_predict = []
        # y_prob = []

        accuracy = []
        precision = []
        recall = []
        f1 = []

        for i in range(n_splits):
            x_train, x_test = self.X.iloc[_train_index[i]], self.X.iloc[_test_index[i]]  # Pandas Series
            y_train, y_test = self.y.iloc[_train_index[i]], self.y.iloc[_test_index[i]]  # Pandas Series

            # fit model
            self.clf.fit(x_train, y_train)
            _y_predict = self.clf.predict(x_test)
            # _y_prob = self.clf.predict_proba(x_test)

            accuracy.append(accuracy_score(y_test, _y_predict))
            precision.append(precision_score(y_test, _y_predict, average=average, zero_division=np.nan))
            recall.append(recall_score(y_test, _y_predict, average=average, zero_division=np.nan))
            f1.append(f1_score(y_test, _y_predict, average=average, zero_division=np.nan))

            y_t.append(y_test)
            y_predict.append(_y_predict)
            # y_prob.append(_y_prob)

        # Generate diagrams
        _path_cm = os.path.join(path, "Peformance")
        os.makedirs(_path_cm, exist_ok=True)
        fig = Graphics(_path_cm)
        fig.cm(self.name, self.y.unique().tolist(), y_t[fold], y_predict[fold])
        # fig.roc(self.name, self.clf, y_t[fold], y_prob[fold])

        return accuracy, precision, recall, f1


class Graphics:
    """
    Diagrams to illustrate evaluation of machine learning models.

    Parameters:
        - path: Path of the output directory.

    Methods:
        - roc: Under development.

        - cm(name, y, y_predict): Generate confusion matrix.

        - metric(df): Generate box plot to show metrics of different classifiers.
    """
    def __init__(
            self,
            path: str
    ):
        self.path = path

        # 字体
        self.font_config = {
            "font.family": 'Times New Roman',
            "font.size": '6'
        }
        plt.rcParams.update(self.font_config)

    def cm(
            self,
            name,
            labels,
            y_true,
            y_predict
    ):
        plt.figure(num=f"Confusion matrix {name}")

        _cm = confusion_matrix(y_true=y_true, y_pred=y_predict, labels=labels)
        ConfusionMatrixDisplay(_cm, display_labels=labels).plot(xticks_rotation='vertical')

        plt.savefig(os.path.join(self.path, f"Confusion matrix - {name}.pdf"))  # 保存图片
        plt.close()

    def metric(self, df: pd.DataFrame):
        plt.figure(num='Metric')

        # import data
        _ac = {}
        _pr = {}
        _re = {}
        _f1 = {}
        for index, row in df.iterrows():
            _ac[row['Classifier']] = row['Accuracy']
            _pr[row['Classifier']] = row['Precision']
            _re[row['Classifier']] = row['Recall']
            _f1[row['Classifier']] = row['F1']

        df_ac = pd.DataFrame(_ac)
        df_pr = pd.DataFrame(_pr)
        df_re = pd.DataFrame(_re)
        df_f1 = pd.DataFrame(_f1)

        plt.subplot(2, 2, 1)
        seaborn.boxplot(data=df_ac).set(xlabel='Classifier', ylabel='Accuracy')

        plt.subplot(2, 2, 2)
        seaborn.boxplot(data=df_f1).set(xlabel='Classifier', ylabel='F1 Score')

        plt.subplot(2, 2, 3)
        seaborn.boxplot(data=df_pr).set(xlabel='Classifier', ylabel='Precision')

        plt.subplot(2, 2, 4)
        seaborn.boxplot(data=df_re).set(xlabel='Classifier', ylabel='Recall')

        plt.savefig(os.path.join(self.path, f"Metrics of classifiers.pdf"))
        print(" <]>- Metrics of classifiers calculated -<[>")
        plt.close()

    def roc(
            self,
            name,
            clf,
            y_test,  # Pandas Series
            y_probs
    ):
        plt.figure(num=f"ROC curve {name}", figsize=(8, 8))

        # Calculate ROC metrics
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(clf.classes_)

        y_test_bin = label_binarize(y_test, classes=clf.classes_)

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # plot
        _cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
        _cmap_values = np.linspace(0, 1, n_classes)
        for i, j in enumerate(_cmap_values):
            plt.plot(fpr[i], tpr[i], color=_cmap(j), lw=2,
                     label='ROC curve (area = {:.2f}) for class {}'.format(roc_auc[i], i))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend(loc='lower right')

        plt.savefig(os.path.join(self.path, f"ROC curve - {name}.pdf"))
        print(" ROC curve generated ")
        plt.close()
