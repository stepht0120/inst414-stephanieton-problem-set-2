'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot

def main():
    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.etl_process()

    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.preprocess_data()

    # PART 3: Call functions/instanciate objects from logistic_regression
    logistic_regression_model, df_arrests_test_lr = logistic_regression.train_logistic_regression(df_arrests)

    # PART 4: Call functions/instanciate objects from decision_tree
    decision_tree_model, df_arrests_test_dt = decision_tree.train_decision_tree(df_arrests)

    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.plot_calibration_curve(df_arrests_test_lr, logistic_regression_model, 'Logistic Regression')
    calibration_plot.plot_calibration_curve(df_arrests_test_dt, decision_tree_model, 'Decision Tree')

if __name__ == "__main__":
    main()