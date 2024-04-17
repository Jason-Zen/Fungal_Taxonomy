import os
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance


class Interpret:
    """
    Interpret glass-box models.

    Parameters:
        - path: Path of the output directory.

        - clf: A fitted classifier.

        - x_train: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training features.

        - y_train: array-like of shape (n_samples,)
            Target labels relative to X.

    Methods:
        - __call__: Auto detection and interpretation.

        - lr: Illustrate feature importance for logistic regression.

        - gini_perm: Illustrate Gini and permutation importance for tree-based models.

        - xgb: Illustrate feature importance for extreme gradient boosting. (unverified)

    Examples:
        Output standard diagrams interpreting random forest with X and y to the current directory.

            >> inter = Interpret(os.getcwd(), RandomForestClassifier(), x_train, y_train)
            >> inter()
    """

    def __init__(
            self,
            path: str,
            clf,
            x_train,
            y_train,
    ):

        self.path = path
        self.clf = clf
        self.name = clf.__class__.__name__
        self.X = x_train
        self.y = y_train

        # 字体
        self.font_config = {
            "font.family": 'Times New Roman',  # 字体
            "font.size": '8'  # 字体大小
        }
        plt.rcParams.update(self.font_config)

    def __call__(self):
        if self.name in [
            'LogisticRegression',
            'DecisionTreeClassifier',
            'RandomForestClassifier',
            'ExtraTreesClassifier',
            'GradientBoostingClassifier',
            'HistGradientBoostingClassifier',
            'AdaBoostClassifier',
            'XGBClassifier'
        ]:
            # self.clf.fit(self.X, self.y)
            if self.name == 'LogisticRegression':
                self.lr()
            elif self.name == 'XGBClassifier':
                self.xgb()
            else:
                self.gini_perm()
        else:
            _explain = Explain(self.path, self.clf, self.X, self.y)
            _explain()

    def lr(self):
        # Plot
        plt.figure(num=f"Feature importance - {self.name}")

        if self.X.shape[1] / 6 > 4:
            height = self.X.shape[1] / 6
        else:
            height = 4
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, height), gridspec_kw={'width_ratios': [2, 3]})
        plt.subplots_adjust(wspace=0.4)

        # Coefficient
        avg_importance = np.mean(np.abs(self.clf.coef_), axis=0)
        df_fi = pd.DataFrame(avg_importance, index=self.X.columns)
        df_fi.sort_values(by=0).plot.barh(ax=ax1)
        ax1.set_xlabel("Feature importance")

        # Permutation importance
        result = permutation_importance(
            self.clf,
            self.X,
            self.y,
            n_repeats=20,
            random_state=None,
            n_jobs=1  # Number of parallel run
        )
        perm_sorted_idx = result.importances_mean.argsort()

        ax2.boxplot(
            result.importances[perm_sorted_idx].T,
            vert=False,
            labels=self.X.columns[perm_sorted_idx],
        )
        ax2.axvline(x=0, color='k', linestyle=':')  # '--' dashed line
        ax2.set_xlabel("Decrease in accuracy score")

        plt.savefig(os.path.join(self.path, f"Feature importance - LR.pdf"))
        plt.close()

    def gini_perm(self):  # for "DT", "RF", "ET", "ERT", "GB", "AB"
        """
        Gini Importance or Mean Decrease in Impurity (MDI):
            Count the times each feature is used to split a node, weighted by the
            number of samples it splits.

        Permutation importance:
            Measures the contribution of each feature to a fitted model’s statistical
            performance on a given tabular dataset
        """

        # Plot
        plt.figure(num=f"Feature importance - {self.name}")

        if self.X.shape[1] / 6 > 4:
            height = self.X.shape[1] / 6
        else:
            height = 4
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, height), gridspec_kw={'width_ratios': [2, 3]})
        plt.subplots_adjust(wspace=0.4)

        # Gini Importance (Mean Decrease in Impurity)
        mdi_importance = pd.Series(self.clf.feature_importances_, index=self.X.columns)
        mdi_importance.sort_values().plot.barh(ax=ax1)
        ax1.set_xlabel("Gini importance")
        # Permutation importance
        result = permutation_importance(
            self.clf,
            self.X,
            self.y,
            n_repeats=20,
            random_state=None,
            n_jobs=1  # Number of parallel run
        )
        perm_sorted_idx = result.importances_mean.argsort()

        ax2.boxplot(
            result.importances[perm_sorted_idx].T,
            vert=False,
            labels=self.X.columns[perm_sorted_idx],
        )
        ax2.axvline(x=0, color='k', linestyle=':')  # '--' dashed line
        ax2.set_xlabel("Decrease in accuracy score")

        plt.savefig(os.path.join(self.path, f"Feature importance - {self.name}.pdf"))
        plt.close()

    def xgb(self):
        import xgboost
        plt.figure(num=f"Plot feature importance")
        xgboost.plot_importance(self.clf, importance_type='weight')
        plt.savefig(os.path.join(self.path, f"Feature importance - XGB.pdf"))
        plt.close()


class Explain:
    """
    Explain impact of each feature on the model's prediction of both black- and glass-box models.

    Parameters:
        - path: Path of the output directory.

        - clf: A fitted classifier.

        - x_train: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training features.

        - y_train: array-like of shape (n_samples,)
            Labels of the training features.

        - x_target: pandas.DataFrame
            Target features with exactly the same header of training features.

        - x_target: pandas.DataFrame
            Labels of *x_target*. If *x_target* is not None, this has to be provided.

    Methods:
        - __call__(cls, index): Standard pipelines.

        - summary_plot: For global feature importance and single class.

        - force_plot(cls, index): For single class or instance.

    Examples:
        Output standard diagrams interpreting random forest model to the current directory.

            >> exp = Explain(os.getcwd(), RandomForestClassifier(), x_train, y_train, x_target, y_target)
            >> exp()
            >> exp(0)
            >> exp(0, 1)
    """

    def __init__(
            self,
            path: str,
            clf,
            x_train,
            y_train,
            x_target: pd.DataFrame = None,
            y_target: pd.DataFrame = None
    ):
        if x_target is None:
            x_target = x_train
            y_target = y_train
        else:
            if not (x_target.columns == x_train.columns).all():
                raise ValueError("Features of target data should be identical to the training data")
            if y_target is None:
                raise ValueError("Labels of target features are needed")

        self.path = path
        self.clf_name = clf.__class__.__name__
        self.X = x_target
        self.y = y_target
        # self.callable_model = partial(clf.predict_proba(self.X), clf)

        if self.clf_name in [
            'DecisionTreeClassifier',
            'RandomForestClassifier',
            'ExtraTreesClassifier',
            'GradientBoostingClassifier',
            'HistGradientBoostingClassifier',
            'XGBClassifier'
        ]:
            _explainer = shap.TreeExplainer(clf)
        elif self.clf_name in [
            'SVC',
            'LogisticRegression',
            'KNeighborsClassifier',
            'MLPClassifier',
            'AdaBoostClassifier',
        ]:
            _k = len(self.y.value_counts())
            background_data = shap.sample(self.X, _k*5)
            _explainer = shap.KernelExplainer(clf.predict_proba, data=background_data)
        else:
            _explainer = shap.Explainer(clf)  # choose explainer automatically

        self.expected_values = _explainer.expected_value
        self.shap_values = _explainer.shap_values(self.X)  # shap_values: list
        # self.shap_interaction_values = explainer.shap_interaction_values(self.X)  # shap_values: list

        if self.clf_name == 'XGBClassifier':
            self.class_name = y_train.unique().tolist()
        else:
            self.class_name = clf.classes_

        # 字体
        self.font_config = {
            "font.family": 'Times New Roman',  # 字体
            "font.size": '10'  # 字体大小
        }
        plt.rcParams.update(self.font_config)

    def __call__(
            self,
            cls: int = None,  # specific class
            index: int = None  # specific instance
    ):
        if cls is None:  # global
            self.summary_plot()
        else:
            if index is None:  # local
                self.summary_plot(cls)
                # self.force_plot(cls)
                # self.decision_plot(cls)
            else:  # instance
                self.force_plot(cls, index)

    def summary_plot(self, cls: int = None):  # 全局条形图
        if cls is None:  # global impact
            file_name = f"Global impact - {self.clf_name}.pdf"
            plt.figure(num=file_name)
            _cmap = plt.get_cmap("tab10" if len(self.class_name) <= 10 else "tab20")  # custom  color
            shap.summary_plot(
                self.shap_values,
                features=self.X,
                class_names=self.class_name,
                color=_cmap,
                show=False
            )
            plt.savefig(os.path.join(self.path, file_name))
            print(f" <]>- SHAP Summary generated -<[>")

        else:  # local impact
            file_name = f"Beeswarmplot of {self.class_name[cls]}.pdf"
            plt.figure(num=file_name)
            shap.summary_plot(
                self.shap_values[cls],
                features=self.X,
                feature_names=self.X.columns.values,
                title=file_name + " - " + self.clf_name,
                color='coolwarm',
                show=False
            )

            _pathout = os.path.join(self.path, "Local_impact")
            os.makedirs(_pathout, exist_ok=True)
            plt.savefig(os.path.join(_pathout, file_name))
            print(f" <]>- SHAP Beeswarm generated -<[>")

        plt.close()

    def force_plot(
            self,
            cls: int = 0,
            index: int = None
    ):
        if index is None:  # local impact
            file_name = f"Forceplot of {self.class_name[cls]} - {self.clf_name}.html"
            plt.figure(num=file_name)

            force = shap.force_plot(
                self.expected_values[cls],  # base value（average prediction）
                self.shap_values[cls],  # shap value
                self.X,
                link='logit',  # "logit" will change log-odds numbers into probabilities
                figsize=(20, 3),
                matplotlib=False,
                show=False
            )
            _pathout = os.path.join(self.path, "Local_impact")
            os.makedirs(_pathout, exist_ok=True)
            shap.save_html(os.path.join(_pathout, file_name), force)

        else:  # specific sample impact
            file_name = f"Forceplot of {self.y[index]} on {self.class_name[cls]} - {self.clf_name}.html"
            plt.figure(num=file_name)

            force = shap.force_plot(
                self.expected_values[cls],  # base value（average prediction）
                self.shap_values[cls][index, :],  # shap value
                self.X.iloc[index, :],
                link='logit',  # "logit" will change log-odds numbers into probabilities
                contribution_threshold=0.05,
                figsize=(10, 3),
                matplotlib=False,
                show=False
            )
            _pathout = os.path.join(self.path, "Sample_impact")
            os.makedirs(_pathout, exist_ok=True)
            shap.save_html(os.path.join(_pathout, file_name), force)
            # plt.savefig(os.path.join(self.path, file_name))

        print(" <]>- Force plot generated -<[>")
        plt.close()
