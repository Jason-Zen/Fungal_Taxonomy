"""
Major dependencies for the development environment:

    - numpy (v1.26.4)
        Required-by: Bottleneck, contourpy, imbalanced-learn, matplotlib, mkl-fft, mkl-random,
            numba, numexpr, pandas, scikit-learn, scipy, seaborn, shap, xgboost

    - scipy (v1.12.0)
        Requires: numpy
        Required-by: imbalanced-learn, scikit-learn, shap, xgboost

    - scikit-learn (v1.4.1.post1)
        Requires: joblib, numpy, scipy, threadpoolctl
        Required-by: imbalanced-learn, shap

    - xgboost (v2.0.3)
        Requires: numpy, scipy

    - pandas (v2.2.1)
        Requires: numpy, python-dateutil, pytz, tzdata
        Required-by: seaborn, shap

    - openpyxl (v3.1.2)
        Requires: et-xmlfile

    - Jinja2 (v3.1.3)
        Requires: MarkupSafe

    - imbalanced-learn (v0.12.2)
        Requires: joblib, numpy, scikit-learn, scipy, threadpoolctl
        Required-by: imblearn

    - shap (v0.42.1)
        Requires: cloudpickle, numba, numpy, packaging, pandas, scikit-learn, scipy, slicer, tqdm

    - matplotlib v3.8.0
        Requires: contourpy, cycler, fonttools, kiwisolver, numpy, packaging, pillow, pyparsing, python-dateutil
        Required-by: seaborn

    - seaborn (v0.12.2)
        Requires: matplotlib, numpy, pandas
"""


class MPLInfo:
    version = 'v1.0.0'

    date = '2024/04/01'

    dependency = [
        'numpy',
        'scipy',
        'sklearn',
        'xgboost',
        'pandas',
        'openpyxl',
        'jinja2',
        'imblearn',
        'shap',
        'matplotlib',
        'seaborn'
    ]


missing_library = []

for lib in MPLInfo.dependency:
    try:
        __import__(lib)
    except ImportError:
        missing_library.append(lib)

if len(missing_library) != 0:
    input(f"The following library is missing:\n{missing_library}")
