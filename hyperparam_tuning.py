from sklearnex import patch_sklearn
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import NuSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import pandas as pd
import numpy as np
patch_sklearn()

class Hyperparam_Tuner():

    def __init__(self, input, output, model='RF'):
        self.X = input
        self.y = output
        self.model_name = model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42)

    def run(self, random_model=None):
        if random_model == None:
            self.y_pred_random, self.r2_random, self.mae_random, self.best_random = self.run_random()
        else:
            print('Reading tuned random model from {}'.format(random_model))
            random = load(random_model)
            self.best_random = random.get_params()
        self.y_pred_dense, self.r2_dense, self.mae_dense, self.best_grid = self.run_dense(self.best_random)
        return self.y_pred_dense, self.r2_dense, self.mae_dense, self.best_grid

    def run_random(self):
        if self.model_name == 'RF':
            model = RandomForestRegressor()
            random_grid = {'n_estimators': [100, 200, 500],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [10,20,50,100,None],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]
              }
        if self.model_name == 'MLP':
            model = MLPRegressor()
            random_grid = {'hidden_layer_sizes': [(5,),(10,),(50,),(5,5),(10,10),(50,50)],
               'activation': ['identity','logistic','tanh','relu'],
               'solver': ['lbfgs','sgd','adam'],
               'alpha': [0.0001,0.001,0.01,0.1],
               'learning_rate': ['constant','invscaling','adaptive'],
               'learning_rate_init': [0.0001,0.001,0.01,0.1],
               'power_t': [0.1, 0.5],
               'momentum': [0.1, 0.5, 0.9],
               'max_iter': [500]
              }
        if self.model_name == 'SVR':
            model = NuSVR()
            random_grid = {'nu':[0.25,0.5,0.75],
               'C':[1.0,10.0,100.0],
               'kernel':['rbf','poly','sigmoid','linear'],
               'degree':[1,2,3,4,5],
               'gamma': [0.1, 0.01, 0.001, 0.0001, 'scale', 'auto'],
               'shrinking': [True, False]
              }

        random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
        random.fit(self.X_train,self.y_train)
        best_params = random.best_params_
        best_estimator = random.best_estimator_
        y_pred = best_estimator.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        dump(best_estimator, 'random_search_trained_model.joblib')
        return y_pred, r2, mae, best_params

    def run_dense(self,best_random):
        if self.model_name == 'RF':
            model = RandomForestRegressor()
            old_n_estimators = best_random['n_estimators']
            if old_n_estimators == 100:
                n_estimators = [100, 200, 300]
            elif old_n_estimators == 500:
                n_estimators = [300, 400, 500]
            else:
                n_estimators = [old_n_estimators - 100, old_n_estimators, old_n_estimators + 100]

            old_max_depth = best_random['max_depth']
            max_depth_vec = [10,20,50,100,None]
            if old_max_depth == 10:
                max_depth = [10,20]
            elif old_max_depth == None:
                max_depth = [100,None]
            else:
                index = max_depth_vec.index( old_max_depth )
                max_depth = max_depth_vec[ index-1 : index+2 ]

            dense_grid = {'n_estimators': n_estimators,
               'max_features': [best_random['max_features']],
               'max_depth': max_depth,
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]
              }

        if self.model_name == 'MLP':
            model = MLPRegressor()
            if len(best_random['hidden_layer_sizes']) == 1:
                hidden_layer_sizes = [(5,),(10,),(50,)]
            else:
                hidden_layer_sizes = [(5,5),(10,10),(50,50)]

            old_alpha = best_random['alpha']
            alpha_vec = [0.0001,0.001,0.01,0.1]
            if old_alpha == 0.0001:
                alpha = [0.0001,0.001]
            elif old_alpha == 0.1:
                alpha = [0.01, 0.1]
            else:
                index = alpha_vec.index( old_alpha )
                alpha = alpha_vec[ index-1 : index+2 ]

            old_learning_rate_init = best_random['learning_rate_init']
            learning_rate_init_vec = [0.0001,0.001,0.01,0.1]
            if old_learning_rate_init == 0.0001:
                learning_rate_init = [0.0001,0.001]
            elif old_learning_rate_init == 0.1:
                learning_rate_init = [0.01, 0.1]
            else:
                index = learning_rate_init_vec.index( old_learning_rate_init )
                learning_rate_init = learning_rate_init_vec[ index-1 : index+2 ]

            dense_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'activation': ['identity','logistic','tanh','relu'],
               'solver': ['lbfgs','sgd','adam'],
               'alpha': alpha,
               'learning_rate': ['constant','invscaling','adaptive'],
               'learning_rate_init': learning_rate_init,
               'power_t': [0.1, 0.5],
               'momentum': [0.1, 0.5, 0.9],
               'max_iter': [500]
              }

        if self.model_name == 'SVR':
            model = NuSVR()
            old_nu = best_random['nu']
            if old_nu == 0.25:
                nu = [0.1,0.2,0.3]
            elif old_nu == 0.5:
                nu = [0.4,0.5,0.6]
            else:
                nu = [0.7,0.8,0.9]

            old_C = best_random['C']
            if old_C == 1.0:
                C = [1.0,10.0]
            elif old_C == 100.0:
                C = [50.0,100.0]
            else:
                C = [10.0,50.0]

            old_degree = best_random['degree']
            degree_vec = [1,2,3,4,5]
            if old_degree == 1:
                degree = [1,2,3]
            elif old_degree == 5:
                degree = [3,4,5]
            else:
                index = degree_vec.index( old_degree )
                degree = degree_vec[ index-1 : index+2 ]

            old_gamma = best_random['gamma']
            gamma_vec = [0.1, 0.01, 0.001, 0.0001, 'scale', 'auto']
            if old_gamma == 0.1:
                gamma = [0.1, 0.01]
            elif old_gamma == 0.0001:
                gamma = [0.001, 0.0001]
            elif old_gamma == 'scale':
                gamma = ['scale']
            elif old_gamma == 'auto':
                gamma = ['auto']
            else:
                index = gamma_vec.index( old_gamma )
                gamma = gamma_vec[ index-1 : index+2 ]

            dense_grid = {'C':C,
               'kernel':['rbf','poly','sigmoid','linear'],
               'degree':degree,
               'gamma':gamma,
               'shrinking': [True, False]
              }
        grid = GridSearchCV(estimator = model, param_grid = dense_grid, cv = 5, verbose=2,  n_jobs = -1)
        grid.fit(self.X_train,self.y_train)
        best_params = grid.best_params_
        best_estimator = grid.best_estimator_
        y_pred = best_estimator.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        dump(best_estimator, 'grid_search_trained_model.joblib')
        pd.DataFrame( np.column_stack( (self.y_test, y_pred) ), columns=['y_test','y_pred'] ).to_csv('y_pred.csv',index=False) 
        return y_pred, r2, mae, best_params

