import os
import pickle
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from . import Model_evaluation
from . import Preprocess
from .Model_estimator import Classification
from .Model_interpretation import Explain, Interpret


class LoadData:
    """
    Load datasets and models.

    Methods:
        - load_excel: Assign the path of the input Excel file.

        - load_model: Load a machine learning model from a file.
    """

    def __init__(self):
        self.excel_path = None
        self.path_out = None
        self.model = None

    def load_excel(self, path: str = None):
        """
        Choose the input Excel file and set the output directory.
        :return: Path of the input Excel file, output directory
        """
        if path is None:
            while True:
                self.excel_path = input("[Please enter the path of the Excel file]: ")  # *** user input ***
                if os.path.exists(self.excel_path):
                    break
                else:
                    print("!!!!! Incorrect path !!!!!\n")
        else:
            self.excel_path = path

        self.path_out = os.path.join(os.path.dirname(self.excel_path), "Out_ML")
        os.makedirs(self.path_out, exist_ok=True)

        return self.excel_path, self.path_out

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(self.path_out, 'Model.pkl')

        try:
            with open(model_path, 'rb') as file_model:
                self.model = pickle.load(file_model)
        except FileNotFoundError as e:
            print(f"***** {e} *****")
        except EOFError as e:
            print(f"***** {e} *****")
        except Exception as e:
            print(f"***** {e} *****")


class ExcelParser:
    """
    Parse datasets in an Excel file.

    Parameters:
        - ib: Whether to perform under- ('Under') or over-sampling ('Over') for imbalanced data.
        - excel_in: Input Excel file. If None, will be determined while running.
        - sheet_pred: Name of the worksheet with data to be predicted. (Default: TD)

    Methods:
        - train: Test different classifiers with data in the 1st worksheet of the Excel file.
            Save the best, trained model according to *criterion* ('Accuracy', 'Precision',
            'Recall' or 'F1') to the file 'Model.pkl'.

        - prediction: Predict classes based on features in the worksheet *sheet_pred* with the model in
            the '*.pkl' file or the result from *train_test()*. If the 1st column in *sheet_pred* is true
            labels, please set *val*=True so that a confusion matrix will be generated.

        - explanation: Illustrate a specific prediction of a specific class.

    Examples:
        1. Test and select the best model for the dataset in an Excel file.
            >> parser = ExcelParser()
            >> parser.train()
        2. Predict entries in the worksheet *sheet_pred*.
            >> parser.prediction()
        3. Illustrate the prediction of the first entry to the first class
            >> parser.explanation()
            >> 1, 1
    """

    def __init__(self, ib: str = None, excel_in: str = None, sheet_pred: str = 'TD'):
        self.model_clf = None
        self.best_model = None
        self.loader = LoadData()
        self.excel_in, self.path_dir = self.loader.load_excel(excel_in)
        self.sheet_pred = sheet_pred
        self.numerical_label = False
        self.label_dict = {}

        print("\n======== Parsing the Excel file ========")

        # Verify Excel data
        self.df1 = pd.read_excel(self.excel_in, sheet_name=0, dtype={0: str}, header=0)
        self.df1, self.feature = Preprocess.verification(self.df1, self.path_dir)  # verified data, features
        # Data normalization
        x_df1 = self.df1.iloc[:, 1:]
        y_df1 = self.df1.iloc[:, 0]
        x_df1_norm = Preprocess.normalization(x_df1, y_df1, self.path_dir)

        # for imbalanced data
        if ib is None:
            self.x_train, self.y_train = x_df1_norm, y_df1
        elif ib == 'Under':
            enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=2, kind_sel='all')  # under-sampling
            self.x_train, self.y_train = enn.fit_resample(x_df1_norm, y_df1)
        elif ib == 'Over':
            smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=4)  # over-sampling
            self.x_train, self.y_train = smote.fit_resample(x_df1_norm, y_df1)
        else:
            raise ValueError("Unsupported sampling")

        if ib is not None:
            df_resampled = pd.concat([self.y_train, self.x_train], axis=1)  # convert to dataframe
            excel_out = os.path.join(self.path_dir, "Results.xlsx")
            with pd.ExcelWriter(excel_out, engine='openpyxl', mode='a', if_sheet_exists='replace') as file_excel:
                df_resampled.to_excel(file_excel, index=False, sheet_name='Resampled')

        # df2
        self.x_df2 = None
        self.x_df2_norm = None
        self.y_df2 = None

    def train(self, opt: list[str] = None, metric: str = "F1", stop_signal: callable = None):
        # Test different classifiers
        df_models = Classification.test_clf(
            self.x_train,  # x
            self.y_train,  # y
            self.path_dir,  # Path
            opt,
            stop_signal
        )
        if stop_signal():
            return

        model_evaluator = Model_evaluation.Graphics(self.path_dir)
        model_evaluator.metric(df_models)

        # Select the best model
        self.best_model = Classification.select_clf(
            df_models,
            metric
        )

        # Save the best model
        _path_model = os.path.join(self.path_dir, 'Model.pkl')
        with open(_path_model, 'wb') as file_model:
            pickle.dump(self.best_model, file_model)

        inter = Interpret(
            self.path_dir,
            self.best_model,
            self.x_train,
            self.y_train
        )
        inter()

        _exp_test_train = Explain(
            self.path_dir,
            self.best_model,
            self.x_train,
            self.y_train
        )
        _exp_test_train()
        for _cls in range(len(self.y_train.unique().tolist())):
            _exp_test_train(cls=_cls)  # Local impact

    def predict(self, val: bool = False, model_path: str = None):
        try:
            df2 = pd.read_excel(self.excel_in, sheet_name=self.sheet_pred, dtype={0: str}, header=0)
            if df2.columns.equals(self.feature):
                # Data normalization
                self.x_df2 = df2.iloc[:, 1:]
                self.y_df2 = df2.iloc[:, 0]
                self.x_df2_norm = Preprocess.normalization(self.df1.iloc[:, 1:], self.df1.iloc[:, 0], self.path_dir,
                                                           self.x_df2, self.y_df2)
            else:
                raise ValueError("Features of predicting data should be identical to the training data\n")
        except Exception as e:
            raise ValueError(f"Errors when verifying worksheet {self.sheet_pred}: {e}\n")

        print("\n======= Predicting data in ‘TD' ========")

        self.loader.load_model(model_path=model_path)
        if self.loader.model is None:
            if self.best_model is not None:
                self.model_clf = self.best_model
            else:
                raise ValueError("The model is unavailable from either .pkl file or tested results")
        else:
            self.model_clf = self.loader.model

        y_df2_predict = self.model_clf.predict(self.x_df2_norm)
        y_label = self.y_train.unique().tolist()

        if self.model_clf.__class__.__name__ == "XGBClassifier" and not self.numerical_label:
            from sklearn.preprocessing import LabelEncoder
            self.numerical_label = True
            label_encoder = LabelEncoder()
            y_labelled = label_encoder.fit_transform(self.y_train)
            self.model_clf.fit(self.x_train, y_labelled)
            self.label_dict = {encoded_label: self.y_train for encoded_label, self.y_train in
                               zip(y_labelled, self.y_train)}
            y_df2_predict = pd.Series(y_df2_predict).map(self.label_dict)
        else:
            self.model_clf.fit(self.x_train, self.y_train)

        # Output predicting results
        y_df2_predict = pd.DataFrame({'Predicted_Class': y_df2_predict})
        df2_out = pd.concat([y_df2_predict, self.y_df2, self.x_df2], axis=1)

        # write to Excel
        excel_out = os.path.join(self.path_dir, "Results.xlsx")
        with pd.ExcelWriter(excel_out, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer_ac:
            df2_out.to_excel(writer_ac, index=False, sheet_name='Prediction')

        # Generate confusion matrix
        if val:
            fig = Model_evaluation.Graphics(self.path_dir)
            fig.cm(self.model_clf.__class__.__name__, y_label, self.y_df2, y_df2_predict)

    def explain(self, cls=None, idx=None):
        if self.x_df2_norm is None:
            input(f"!!!!! Requires valid worksheet {self.sheet_pred} !!!!!")
            return

        print("\n======= Explaining data in ‘TD' ========")

        if self.numerical_label:
            cls_info = np.array(self.y_train.unique().tolist())
        else:
            cls_info = self.model_clf.classes_

        # Check feature importance of specific prediction
        explainer = Explain(
            self.path_dir,
            self.model_clf,
            self.x_train,
            self.y_train,
            self.x_df2_norm,
            self.y_df2
        )

        if (cls is None) or (idx is None):
            while True:
                for i, cls_name in enumerate(cls_info):
                    print(f"{i + 1} - {cls_name}")
                cls = input("(Please choose the digits of target class): ")
                idx = input(f"(Please enter the number of an entry in the worksheet {self.sheet_pred}): ")

                try:
                    explainer(cls=int(cls) - 1, index=int(idx) - 1)
                except Exception as e:
                    print(e)

                while True:
                    _exit = input("Continue checking? (y/n): ")
                    if _exit == 'n':
                        return
                    elif _exit == 'y':
                        break
        else:
            if cls in cls_info:
                idx_cls = np.where(cls_info == cls)[0]
                print(f"Explaining feature importance of entry {idx_cls+1} to {cls} ...")
                if 0 < int(idx) <= len(self.x_df2_norm):
                    explainer(cls=int(idx_cls), index=int(idx) - 1)
                else:
                    raise ValueError("The index to be explained is invalid")
            else:
                raise ValueError("The class to be explained is invalid")


if __name__ == '__main__':
    parser = ExcelParser(
        # ib=None (default), 'Under' or 'Over',
        # excel_in="",
        # sheet_pred=""
    )

    parser.train(
        # opt=None (default) or in ['LR', 'KNN', 'SVM', 'NN', 'DT', 'RF', 'ET', 'ERT', 'AB', 'XGB']
        # criterion="F1" or 'Accuracy', 'Precision' or 'Recall'
    )

    parser.predict(
        # val=False (default) or True,
        # model_path=''
    )

    parser.explain()
