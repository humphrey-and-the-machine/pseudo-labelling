# pseudo-labelling scripts from Humphrey et al. (2022, MNRAS, submitted)
Scripts used to create results in our MNRAS paper on application of pseudo-labelling to estimation of galaxy properties

If you're not using a GPU, you'll need to modify a keyword for CatBoost, LightGBM, and XGBoost:

- for CatBoost, remove or change the `task_type` keyword (if not defined, it will default to CPU);

- for XGBoost, change `tree_method` to `hist` or `exact`. The latter can be a lot slower, but often gives slightly higher quality results; 

- for LighGBM, remove or change the `device` keyword (if not defined, it will default to CPU).

Depending on the computer you use, you may also need to reduce the `max_depth` keyword for some of the learning algorithms. 

If you don't have the Intel extension to Sxikit-Learn installed, or don't have an Intel CPU, you'll also need to delete/comment out these lines:

`from sklearnex import patch_sklearn`  
`patch_sklearn("knn_regressor")`  
`patch_sklearn('random_forest_regressor')`  

If you do have an Intel CPU, I really recommend installing the Intel `sklearnex` package (and an Intel version of Python 3) due to the dramatic speed increases you'll get in some functions.
