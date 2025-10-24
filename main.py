# This file serves as an example for configuring a runtime function. 
# Here is the execution functions of the external behavior model trained in 2024-06

import sys
import pytz
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
from spectralcrop.models import Predict
from spectralcrop.utils import PathManager

#--- Time zone for date handling
timezone_colombia = pytz.timezone('America/Bogota')

# Setting variables to model and save predictions
model_name = "lr2_scorecard_cat_no_grp_porc_endeuda_sf_pipe_grid_model"
translate = [0, 430, 485, 640, 690, 718, 800, 813, 850, 900, 1000]

def run_prediction(save_to_impala: bool = True, **kwargs):
    """
    Run predictions for customer defaults within the specified date.

    Parameters
    ----------
    save_to_impala : bool, optional
        If True, save prediction results to Impala. Default is True.
    **kwargs : dict
        Additional keyword arguments:
        - start_date : str, optional
            Start date for prediction processing. If provided, it overrides the default start date.
        - end_date : str, optional
            End date for prediction processing. If provided, it overrides the default end date.

    Raises
    ------
    SystemExit
        If the model has already been executed for the specified date 
        range (last_date_model_execution matches current_date_str).

    Notes
    -----
    This function runs predictions for customer defaults based on the specified 
    date range. It checks if the model has already been executed for the date 
    and raises a SystemExit exception if so. Otherwise, it iterates over the dates
    range and executes predictions for each month within the range, using the 
    `make_prediction` function.

    The results of the predictions can be optionally saved to Impala 
    if `save_to_impala` is True.
    """

    global model_name, translate

    try:
        start_date = kwargs['start_date']
    except:
        #--- Time zone for date handling
        timezone_colombia = pytz.timezone('America/Bogota')
        current_date = datetime.now(timezone_colombia) - relativedelta(months=1)
        start_date = current_date.strftime("%Y-%m")

    model_name = kwargs.get('model_name', model_name)
    end_date = kwargs.get('end_date', start_date)

    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    dates = [d.strftime("%Y-%m") for d in dates]
    predictor_object = Predict(dates=dates)
    predictions = predictor_object.make_prediction(
                                        model_name=model_name,
                                        lims_translate=translate,
                                        save_predictions=True
                                        )

    # Control clients
    sd = start_date.replace('-', '')
    ed = end_date.replace('-', '')
    if ed == sd:
        file_name = 'control_clients_pred_{}.log'.format(sd)
    else:
        file_name = 'control_clients_pred_{}_{}.log'.format(sd, ed)

    f_path = PathManager().get_abs_path_folder('logs') + file_name
    control_clients = [32244023, 1048019199, 1048019951, 1037624693, 1037631295, 1037657056]
    with open(f_path, "w") as f:
        print(predictions.loc[predictions['id'].isin(control_clients), ['fa', 'id', 'Score', 'T_Externa']].sort_values(by=['id', 'fa']), file=f)

def run_generate_features_to_train():
    pass

def run_train_model():
    pass

def run_make_backtesting():
    pass

if __name__ == '__main__':

    # Dictionary to select the process(function) to be executed

    process_to_run = {
        "prediction": run_prediction,
        "features_to_train": run_generate_features_to_train,
        "train_model": run_train_model,
        "make_backtesting": run_make_backtesting
    }[sys.argv[1]]

    try:
        kwargs = {value.split('=')[0]:value.split('=')[1] for value in sys.argv[2:]}
    except:
        raise ValueError('A value that does not specify the parameter to configure has been entered. It should have the form key=value')
    
    start_execution = datetime.now(timezone_colombia).strftime('%Y-%m-%d %H:%M:%S')
    print(f'-------- Process started at {start_execution} Colombian time --------\n')

    process_to_run(**kwargs)

    # python main.py prediction
    # python main.py prediction start_date=2024-04
    # python main.py prediction start_date=2024-03 end_date=2024-05