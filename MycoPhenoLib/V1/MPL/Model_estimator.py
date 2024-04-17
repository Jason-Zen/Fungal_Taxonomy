import os
import statistics
import pandas as pd

from .Model_evaluation import Evaluation


class Classification:
    """
    Procedures and algorithms for classification issues.

    Methods:
        - test_clf(x, y, path, opt=None): Test the performance of selected machine learning classifiers.

        - select_clf(df_models, para): Select the best classifier from tested ones in *test_clf()*.

    Algorithms:
        - logistic_regression: Perform logistic regression by LogisticRegression().

        - k_nearest_neighbors: Perform k-nearest neighbors by KNeighborsClassifier().

        - support_vector_machines: Perform support vector machines by SVC().

        - neural_networks: Perform neural networks by MLPClassifier().

        - decision_trees: Perform decision trees by DecisionTreeClassifier().

        - extra_trees: Perform extra trees by ExtraTreesClassifier().

        - random_forests: Perform random forests by RandomForestClassifier().

        - extremely_randomized_trees: Perform extremely randomized trees by ExtraTreeClassifier().

        - gradient_boosting: Perform gradient boosting by GradientBoostingClassifier(),
            or HistGradientBoostingClassifier(). **Only for binary classification in new version**

        - adaboost: Perform adaboost by AdaBoostClassifier().

        - extreme_gradient_boosting: Perform extreme gradient boosting by XGBClassifier().
    """

    @staticmethod
    def test_clf(  # 测试各算法参数、准确度
            x,
            y,
            path: str,
            opt: list[str] = None,
            stop_signal: callable = None
    ):
        """
        Test the performance of machine learning classifiers.

        Parameters:
            - x: {array-like, sparse matrix} of shape (n_samples, n_features).
                Training data.

            - y: array-like of shape (n_samples,)
                Target value relative to X.

            - path: Path of the output directory.

            - opt: list of algorithm initials.
                Options of classifiers to be tested from *sl_models*.
                Default = ['DT', 'RF', 'ET', 'ERT'].

                Support  'LR' = logistic_regression(),
                        'KNN' = k_nearest_neighbors(),
                        'SVM' = support_vector_machines(),
                         'NN' = neural_networks(),
                         'DT' = decision_trees(),
                         'ET' = extra_trees(),
                         'RF' = random_forests(),
                        'ERT' = extremely_randomized_trees(),
                         'GB' = gradient_boosting(),
                         'AB' = adaboost(),
                        'XGB' = extreme_gradient_boosting()

        Returns:
            Pandas dataframe with metrics of tested classifiers.
        """
        print("\n============ Testing models ============")

        if opt is None:
            opt = ['DT', 'RF', 'ET', 'ERT']

        sl_model = {
            'LR': 'logistic_regression',
            'KNN': 'k_nearest_neighbors',
            'SVM': 'support_vector_machines',
            'NN': 'neural_networks',
            'DT': 'decision_trees',
            'RF': 'random_forests',
            'ET': 'extra_trees',
            'ERT': 'extremely_randomized_tree',
            'GB': 'gradient_boosting',
            'AB': 'adaboost',
            'XGB': 'extreme_gradient_boosting'
        }

        opt_model = {}
        for item in opt:
            try:
                opt_model[item] = sl_model[item]
            except KeyError:
                print(f"***** {item} is not a valid option *****")

        # 将每个模型的准确率、标准差和对应参数记录到数据框中
        sl = Supervised(x, y, path)
        models_data = []
        for k, v in opt_model.items():
            # 终止信号检测
            if stop_signal():
                return

            _method = getattr(sl, v)
            _clf, _acc, _pre, _rec, _f1, _fea = _method()

            row_dat = [  # 与下方columns=对应
                k,  # Name of the classifier
                _clf,  # Classifier with best parameter
                _fea,  # Selected features
                _acc,  # Accuracy
                statistics.mean(_acc),  # Mean accuracy
                statistics.stdev(_acc),  # Standard deviation of accuracy
                _pre,  # Precision
                statistics.mean(_pre),  # Mean precision
                statistics.stdev(_pre),  # Standard deviation of precision
                _rec,  # Recall
                statistics.mean(_rec),  # Mean recall
                statistics.stdev(_rec),  # Standard deviation of recall
                _f1,  # F1 score
                statistics.mean(_f1),  # Mean F1 score
                statistics.stdev(_f1)  # Standard deviation of F1 score
            ]
            models_data.append(row_dat)

        models = pd.DataFrame(
            models_data,
            columns=[
                'Classifier',
                'Best_parameter',
                'Selected_features',
                'Accuracy',
                'Accuracy_mean',
                'Accuracy_SD',
                'Precision',
                'Precision_mean',
                'Precision_SD',
                'Recall',
                'Recall_mean',
                'Recall_SD',
                'F1',
                'F1_mean',
                'F1_SD'
            ]
        )

        # write to Excel
        path_excel = os.path.join(path, "Results.xlsx")  # output Excel file
        with pd.ExcelWriter(path_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as file_excel:
            models.to_excel(file_excel, index=False, sheet_name='Metrics')  # 输出至Excel

        return models

    @staticmethod
    def select_clf(
            df_models: pd.DataFrame,
            metric: str = 'F1'
    ):
        """
        Select the best classifier tested in *test_clf()*.

        Parameters:
            - df_models: Pandas dataframe with the best parameter of the classifier,
                as well as columns of 'Accuracy', 'Precision', 'Recall' and 'F1'.

            - metric: Column header of 'Accuracy', 'Precision', 'Recall' or 'F1', on which
                the selection will be based.

        Returns:
            The best classifier selected based on the highest mean value of *para*.
        """
        print("\n======= Selecting the best model =======")

        _para_mean = metric + "_mean"
        _para_sd = metric + "_SD"

        _max = df_models[_para_mean].max()
        _max_index = df_models[df_models[_para_mean] == _max].index  # corresponding row

        # Select the model with the highest accuracy and lowest SD
        max_sd = {}
        for i in _max_index:
            max_sd[i] = df_models.loc[i, _para_sd]
        _best_sd_index = min(max_sd, key=max_sd.get)  # Best model row
        best_clf = df_models.loc[_best_sd_index, 'Best_parameter']  # Best model parameter

        return best_clf


class Supervised:
    """
    Estimators of supervised machine learning algorithms.

    Parameters:
        - x: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training features.

        - y: array-like of shape (n_samples,)
            Target labels relative to X.

        - path: Path of the output directory.

    Returns:
        Estimator, accuracy, precision, recall, F1 score, selected features
    """
    def __init__(self, x, y, path):
        self.X = x
        self.y = y
        self.path = path

    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        _algorithm = 'Logistic Regression'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = LogisticRegression(
                max_iter=5000,
                class_weight='balanced',
                multi_class='multinomial'
            )

        _para = {
            'C': [
                0.001,
                0.01,
                0.1,
                1,
                10,
                100
            ],
            'solver': [
                'lbfgs',
                'newton-cg',
                'sag',
                'saga'
            ]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def k_nearest_neighbors(self):
        from sklearn.neighbors import KNeighborsClassifier
        _algorithm = 'k-Nearest Neighbors'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = KNeighborsClassifier()

        _para = {
            'n_neighbors': [
                i for i in range(2, 15)
            ],
            'weights': [
                'uniform',
                'distance'
            ],
            'p': [1, 2]  # p=1:manhattan-distance, p=2:euclidean_distance
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def support_vector_machines(self):
        from sklearn.svm import SVC
        _algorithm = 'Support Vector Machines'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = SVC(probability=True)

        _para = {
            'C': [
                1,
                10,
                100,
                1000
            ],
            'kernel': [
                'poly',
                'rbf'
            ]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def neural_networks(self):
        from sklearn.neural_network import MLPClassifier
        _algorithm = 'Neural Networks'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = MLPClassifier(max_iter=2000, early_stopping=True)

        _para = {
            'activation': [
                'identity',
                'logistic',
                'tanh',
                'relu'
            ],
            'solver': [
                'lbfgs',
                'sgd',
                'adam'
            ],
            'learning_rate': [
                'constant',
                'invscaling',
                'adaptive'
            ]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def decision_trees(self):
        from sklearn.tree import DecisionTreeClassifier
        _algorithm = 'Decision Trees'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = DecisionTreeClassifier()

        _para = {
            'criterion': [
                'gini',
                'entropy',
                'log_loss'
            ],
            'min_samples_leaf': [
                1,
                2,
                3,
                5,
                10
            ],
            'min_impurity_decrease': [
                0,
                0.1,
                0.2,
                0.5
            ]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def extra_trees(self):
        from sklearn.ensemble import ExtraTreesClassifier
        _algorithm = 'Extra Trees'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = ExtraTreesClassifier()

        _para = {
            'criterion': [
                'gini',
                'entropy',
                'log_loss'
            ],
            'min_samples_leaf': [
                1,
                2,
                3,
                5,
                10
            ]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    # 集成学习算法
    def random_forests(self):
        from sklearn.ensemble import RandomForestClassifier
        _algorithm = 'Random Forests'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = RandomForestClassifier()

        _para = {
            'criterion': [
                'gini',
                'entropy',
                'log_loss'
            ],
            'min_samples_leaf': [
                1,
                2,
                3,
                5,
                10
            ]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def extremely_randomized_tree(self):
        from sklearn.tree import ExtraTreeClassifier
        _algorithm = 'Extremely Randomized Tree'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = ExtraTreeClassifier()

        _para = {
            'criterion': [
                'gini',
                'entropy',
                'log_loss'
            ],
            'min_samples_leaf': [
                1,
                2,
                3,
                5,
                10
            ]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def gradient_boosting(self):  # Support only binary classification in new version
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
        _algorithm = 'Gradient Boosting'
        print(f"><><>< Performing {_algorithm} ><><><")

        if len(self.X) < 10000:
            _estimator = GradientBoostingClassifier(n_estimators=50)
        else:
            _estimator = HistGradientBoostingClassifier(max_iter=50)

        # from sklearn.multiclass import OneVsRestClassifier
        # _estimator = OneVsRestClassifier(_estimator)  # Use one-vs-the-rest

        _para = {
            'learning_rate': [  # 每次更新迭代权重时的步长。值越小训练越慢
                0.001,
                0.01,
                0.015,
                0.025,
                0.05,
                0.1
            ],
            'max_depth': [  # 值越大，模型学习的更加具体
                None,
                3,
                5,
                6,
                7,
                9,
                12
            ]
            # 'min_impurity_decrease': [0, 0.1, 0.2, 0.5]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def adaboost(self):
        from sklearn.ensemble import AdaBoostClassifier
        _algorithm = 'AdaBoost'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = AdaBoostClassifier(
                n_estimators=100,
                random_state=None,
                algorithm='SAMME'
            )

        _para = {
            'learning_rate': [  # 每次更新迭代权重时的步长。值越小训练越慢
                0.001,
                0.01,
                0.015,
                0.025,
                0.05,
                0.1
            ]
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea

    def extreme_gradient_boosting(self):
        import xgboost
        _algorithm = 'eXtreme Gradient Boosting'
        print(f"><><>< Performing {_algorithm} ><><><")

        _estimator = xgboost.XGBClassifier(
                n_estimators=100,  # 总迭代次数
                objective="multi:softprob",  # 多分类返回概率
                random_state=None
            )

        _para = {
            'learning_rate': [  # 每次更新迭代权重时的步长。值越小训练越慢
                0.001,
                0.01,
                0.015,
                0.025,
                0.05,
                0.1
            ],
            'gamma': [  # 指定叶节点进行分支所需的损失减少的最小值。设置的值越大，模型就越保守
                0,
                0.05,
                0.1,
                0.3,
                0.5,
                0.7,
                0.9,
                1
            ],
            'max_depth': [  # 值越大，模型学习的更加具体
                None,
                3,
                5,
                6,
                7,
                9,
                12
            ],
        }

        evaluator = Evaluation(
            _estimator,
            self.X,
            self.y,  # Label should be numbers
            _algorithm
        )

        clf, acc, pre, rec, f1, fea = evaluator(_para, self.path)

        print("<><><><><><><><><><><><><><><><><><><><>\n")
        return clf, acc, pre, rec, f1, fea
