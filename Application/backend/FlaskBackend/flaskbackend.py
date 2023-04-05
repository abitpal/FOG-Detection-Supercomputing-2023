from flask import Flask
from flask import request
import tsfresh
import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal
from scipy.signal import find_peaks
import json
from joblib import load
import pickle
import sys
import csv

annotation_checking = True


f = open('Scalers/activityrecogscaler.pkl', 'rb')
scaler1 = pickle.load(f)

f1 = open('Scalers/fogscaler.pkl', 'rb')
scaler2 = pickle.load(f1)

dataCollectionFile = open('dataCollection/session3_annotation_test_walking.csv', 'w')
writer = csv.writer(dataCollectionFile)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)



app = Flask(__name__)

model = load("Models/activityrecognition50.joblib")

@app.route("/activityrec", methods = ['POST'])
def activityRecognitionPostRequest():

    x_list=request.json[0]
    y_list=request.json[1]
    z_list=request.json[2]

    x = featureengineeringforactivityrec(x_list, y_list, z_list)

    prediction = model.predict(x)

    annotation = ['Downstairs', 'Jogging', 'Still', 'Still', 'Upstairs', 'Walking']

    if (annotation_checking==True):
        dataCollection_annotation(prediction)
    
    print(annotation[prediction[0]], file=sys.stderr)

    return {
        'data':json.dumps(prediction, cls=NpEncoder)

    }


def dataCollection_annotation(prediction):
    writer.writerow(prediction)

@app.route("/fogrec", methods = ['POST'])
def ret2():

    x_list=request.json[0]
    y_list=request.json[1]
    z_list=request.json[2]

    x = featureengineeringforfog(x_list, y_list, z_list)

    return {
        'data':x
    }


def featureengineeringforactivityrec(x_list, y_list, z_list):
    x = []

    #mean
    x.append(np.mean(x_list))
    x.append(np.mean(y_list))
    x.append(np.mean(z_list))

    #std dev
    x.append(np.std(x_list))
    x.append(np.std(y_list))
    x.append(np.std(z_list))

    #avg absolute deviation
    x.append(np.mean(np.absolute(x_list - np.mean(x_list))))
    x.append(np.mean(np.absolute(y_list - np.mean(y_list))))
    x.append(np.mean(np.absolute(z_list - np.mean(z_list))))

    #minimum value
    x.append(np.min(x_list))
    x.append(np.min(y_list))
    x.append(np.min(z_list))

    #maximum value
    x.append(np.max(x_list))
    x.append(np.max(y_list))
    x.append(np.max(z_list))

    #difference of maximum and minimum values
    x.append(np.max(x_list) - np.min(x_list))
    x.append(np.max(y_list) - np.min(y_list))
    x.append(np.max(z_list) - np.min(z_list))

    #median
    x.append(np.median(x_list))
    x.append(np.median(y_list))
    x.append(np.median(z_list))

    #median absolute deviation
    x.append(np.median(np.absolute(x_list - np.median(x_list))))
    x.append(np.median(np.absolute(y_list - np.median(y_list))))
    x.append(np.median(np.absolute(z_list - np.median(z_list))))

    #interquartile range
    x.append(np.percentile(x_list, 75) - np.percentile(x_list, 25))
    x.append(np.percentile(y_list, 75) - np.percentile(y_list, 25))
    x.append(np.percentile(z_list, 75) - np.percentile(z_list, 25))

    #negative values count
    x.append(np.sum(np.array(x_list) < 0))
    x.append(np.sum(np.array(y_list) < 0))
    x.append(np.sum(np.array(z_list) < 0))

    #positive values count
    x.append(np.sum(np.array(x_list) > 0))
    x.append(np.sum(np.array(y_list) > 0))
    x.append(np.sum(np.array(z_list) > 0))

    #number of values above mean
    x.append(np.sum(x_list > np.mean(x_list)))
    x.append(np.sum(y_list > np.mean(y_list)))
    x.append(np.sum(z_list > np.mean(z_list)))

    #number of peaks
    x.append(len(find_peaks(x_list)[0]))
    x.append(len(find_peaks(y_list)[0]))
    x.append(len(find_peaks(z_list)[0]))

    #stats.skewness
    x.append(stats.skew(x_list))
    x.append(stats.skew(y_list))
    x.append(stats.skew(z_list))

    #stats.kurtosis
    x.append(stats.kurtosis(x_list))
    x.append(stats.kurtosis(y_list))
    x.append(stats.kurtosis(z_list))

    #energy
    x.append(np.sum(np.square(x_list))/len(x_list))
    x.append(np.sum(np.square(y_list))/len(y_list))
    x.append(np.sum(np.square(z_list))/len(z_list))

    #average resultant acceleration
    x.append(np.mean(np.sqrt(np.square(x_list) + np.square(y_list) + np.square(z_list))))

    #signal magnitude area
    x.append(np.sum(np.absolute(x_list)/len(x_list)) + np.sum(np.absolute(y_list)/len(y_list)) + np.sum(np.absolute(z_list)/len(z_list)))

    x_list_fft = np.abs(np.fft.fft(x_list))[1: int(((len(x_list))/2)+1)]
    y_list_fft = np.abs(np.fft.fft(y_list))[1: int((len(x_list))/2 +1)]
    z_list_fft = np.abs(np.fft.fft(z_list))[1: int((len(x_list))/2 +1)]

    #mean
    x.append(np.mean(x_list_fft))
    x.append(np.mean(y_list_fft))
    x.append(np.mean(z_list_fft))

    #std dev
    x.append(np.std(x_list_fft))
    x.append(np.std(y_list_fft))
    x.append(np.std(z_list_fft))

    #avg absolute deviation
    x.append(np.mean(np.absolute(x_list_fft - np.mean(x_list_fft))))
    x.append(np.mean(np.absolute(y_list_fft - np.mean(y_list_fft))))
    x.append(np.mean(np.absolute(z_list_fft - np.mean(z_list_fft))))

    #minimum value
    x.append(np.min(x_list_fft))
    x.append(np.min(y_list_fft))
    x.append(np.min(z_list_fft))

    #maximum value
    x.append(np.max(x_list_fft))
    x.append(np.max(y_list_fft))
    x.append(np.max(z_list_fft))

    #difference of maximum and minimum values
    x.append(np.max(x_list_fft) - np.min(x_list_fft))
    x.append(np.max(y_list_fft) - np.min(y_list_fft))
    x.append(np.max(z_list_fft) - np.min(z_list_fft))

    #median
    x.append(np.median(x_list_fft))
    x.append(np.median(y_list_fft))
    x.append(np.median(z_list_fft))

    #median absolute deviation
    x.append(np.median(np.absolute(x_list_fft - np.median(x_list_fft))))
    x.append(np.median(np.absolute(y_list_fft - np.median(y_list_fft))))
    x.append(np.median(np.absolute(z_list_fft - np.median(z_list_fft))))

    #interquartile range
    x.append(np.percentile(x_list_fft, 75) - np.percentile(x_list_fft, 25))
    x.append(np.percentile(y_list_fft, 75) - np.percentile(y_list_fft, 25))
    x.append(np.percentile(z_list_fft, 75) - np.percentile(z_list_fft, 25))


    #number of values above mean
    x.append(np.sum(x_list_fft > np.mean(x_list_fft)))
    x.append(np.sum(y_list_fft > np.mean(y_list_fft)))
    x.append(np.sum(z_list_fft > np.mean(z_list_fft)))

    #number of peaks
    x.append(len(find_peaks(x_list_fft)[0]))
    x.append(len(find_peaks(y_list_fft)[0]))
    x.append(len(find_peaks(z_list_fft)[0]))

    #stats.skewness
    x.append(stats.skew(x_list_fft))
    x.append(stats.skew(y_list_fft))
    x.append(stats.skew(z_list_fft))

    #stats.kurtosis
    x.append(stats.kurtosis(x_list_fft))
    x.append(stats.kurtosis(y_list_fft))
    x.append(stats.kurtosis(z_list_fft))

    #energy
    x.append(np.sum(np.square(x_list_fft))/(len(x_list)/2))
    x.append(np.sum(np.square(y_list_fft))/(len(y_list)/2))
    x.append(np.sum(np.square(z_list_fft))/(len(z_list)/2))

    #average resultant acceleration
    x.append(np.mean(np.sqrt(np.square(x_list_fft) + np.square(y_list_fft) + np.square(z_list_fft))))

    #signal magnitude area
    x.append(np.sum(np.absolute(x_list_fft)/(len(x_list)/2)) + np.sum(np.absolute(y_list_fft)/(len(y_list)/2)) + np.sum(np.absolute(z_list_fft)/(len(z_list)/2)))

    # index of max value in time domain
    x.append(np.argmax(x_list))
    x.append(np.argmax(y_list))
    x.append(np.argmax(z_list))


    # index of min value in time domain
    x.append(np.argmin(x_list))
    x.append(np.argmin(y_list))
    x.append(np.argmin(z_list))

    # absolute difference between above indices
    x.append(abs(np.argmax(x_list) - np.argmin(x_list)))
    x.append(abs(np.argmax(y_list) - np.argmin(y_list)))
    x.append(abs(np.argmax(z_list) - np.argmin(z_list)))

    # index of max value in frequency domain
    x.append(np.argmax(x_list_fft))
    x.append(np.argmax(y_list_fft))
    x.append(np.argmax(z_list_fft))

    # index of min value in frequency domain
    x.append(np.argmin(x_list_fft))
    x.append(np.argmin(y_list_fft))
    x.append(np.argmin(z_list_fft))


    # absolute difference between above indices
    x.append(abs(np.argmax(x_list_fft) - np.argmin(x_list_fft)))
    x.append(abs(np.argmax(y_list_fft) - np.argmin(y_list_fft)))
    x.append(abs(np.argmax(z_list_fft) - np.argmin(z_list_fft)))
    

   
    _, x_list_psd = signal.periodogram(x_list_fft)
    _, y_list_psd = signal.periodogram(y_list_fft)
    _, z_list_psd = signal.periodogram(z_list_fft)

    
    #mean
    x.append(np.mean(x_list_psd))
    x.append(np.mean(y_list_psd))
    x.append(np.mean(z_list_psd))

    #std dev
    x.append(np.std(x_list_psd))
    x.append(np.std(y_list_psd))
    x.append(np.std(z_list_psd))

    #avg absolute deviation
    x.append(np.mean(np.absolute(x_list_psd - np.mean(x_list_psd))))
    x.append(np.mean(np.absolute(y_list_psd - np.mean(y_list_psd))))
    x.append(np.mean(np.absolute(z_list_psd - np.mean(z_list_psd))))

    #minimum value
    x.append(np.min(x_list_psd))
    x.append(np.min(y_list_psd))
    x.append(np.min(z_list_psd))

    #maximum value
    x.append(np.max(x_list_psd))
    x.append(np.max(y_list_psd))
    x.append(np.max(z_list_psd))

    #difference of maximum and minimum values
    x.append(np.max(x_list_psd) - np.min(x_list_psd))
    x.append(np.max(y_list_psd) - np.min(y_list_psd))
    x.append(np.max(z_list_psd) - np.min(z_list_psd))

    #median
    x.append(np.median(x_list_psd))
    x.append(np.median(y_list_psd))
    x.append(np.median(z_list_psd))

    #median absolute deviation
    x.append(np.median(np.absolute(x_list_psd - np.median(x_list_psd))))
    x.append(np.median(np.absolute(y_list_psd - np.median(y_list_psd))))
    x.append(np.median(np.absolute(z_list_psd - np.median(z_list_psd))))

    #interquartile range
    x.append(np.percentile(x_list_psd, 75) - np.percentile(x_list_psd, 25))
    x.append(np.percentile(y_list_psd, 75) - np.percentile(y_list_psd, 25))
    x.append(np.percentile(z_list_psd, 75) - np.percentile(z_list_psd, 25))


    #number of values above mean
    x.append(np.sum(x_list_psd > np.mean(x_list_psd)))
    x.append(np.sum(y_list_psd > np.mean(y_list_psd)))
    x.append(np.sum(z_list_psd > np.mean(z_list_psd)))

    #number of peaks
    x.append(len(find_peaks(x_list_psd)[0]))
    x.append(len(find_peaks(y_list_psd)[0]))
    x.append(len(find_peaks(z_list_psd)[0]))

    #skewness
    x.append(stats.skew(x_list_psd))
    x.append(stats.skew(y_list_psd))
    x.append(stats.skew(z_list_psd))

    #kurtosis
    x.append(stats.kurtosis(x_list_psd))
    x.append(stats.kurtosis(y_list_psd))
    x.append(stats.kurtosis(z_list_psd))

    #energy
    x.append(np.sum(np.square(x_list_psd))/(len(x_list)/2))
    x.append(np.sum(np.square(y_list_psd))/(len(y_list)/2))
    x.append(np.sum(np.square(z_list_psd))/(len(z_list)/2))

    #average resultant acceleration
    x.append(np.mean(np.sqrt(np.square(x_list_psd) + np.square(y_list_psd) + np.square(z_list_psd))))

    #signal magnitude area
    x.append(np.sum(np.absolute(x_list_psd)/(len(x_list)/2)) + np.sum(np.absolute(y_list_psd)/(len(y_list)/2)) + np.sum(np.absolute(z_list_psd)/(len(z_list)/2)))

    
    xfinal = scaler1.transform([x]) 
    return xfinal


featurelist = ['UHF__sum_values', 'UHF__mean_abs_change', 'UHF__median',
       'UHF__mean', 'UHF__standard_deviation', 'UHF__variance',
       'UHF__skewness', 'UHF__kurtosis', 'UHF__absolute_sum_of_changes',
       'UHF__longest_strike_above_mean', 'UHF__count_above_mean',
       'UHF__count_below_mean', 'UHF__first_location_of_maximum',
       'UHF__last_location_of_minimum',
       'UHF__percentage_of_reoccurring_values_to_all_values',
       'UHF__percentage_of_reoccurring_datapoints_to_all_datapoints',
       'UHF__sum_of_reoccurring_values',
       'UHF__sum_of_reoccurring_data_points',
       'UHF__ratio_value_number_to_time_series_length', 'UHF__maximum',
       'UHF__absolute_maximum', 'UHF__minimum',
       'UHF__benford_correlation',
       'UHF__time_reversal_asymmetry_statistic__lag_2',
       'UHF__time_reversal_asymmetry_statistic__lag_3', 'UHF__c3__lag_1',
       'UHF__c3__lag_2', 'UHF__c3__lag_3', 'UHF__cid_ce__normalize_True',
       'UHF__cid_ce__normalize_False', 'UHF__symmetry_looking__r_0.05',
       'UHF__symmetry_looking__r_0.1',
       'UHF__large_standard_deviation__r_0.15000000000000002',
       'UHF__large_standard_deviation__r_0.2',
       'UHF__large_standard_deviation__r_0.25', 'UHF__quantile__q_0.1',
       'UHF__quantile__q_0.2', 'UHF__quantile__q_0.3',
       'UHF__quantile__q_0.4', 'UHF__quantile__q_0.6',
       'UHF__quantile__q_0.7', 'UHF__quantile__q_0.8',
       'UHF__quantile__q_0.9', 'UHF__autocorrelation__lag_1',
       'UHF__partial_autocorrelation__lag_1',
       'UHF__partial_autocorrelation__lag_2',
       'UHF__number_cwt_peaks__n_1', 'UHF__number_peaks__n_1',
       'UHF__number_peaks__n_3', 'UHF__number_peaks__n_5',
       'UHF__number_peaks__n_10', 'UHF__binned_entropy__max_bins_10',
       'UHF__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_3__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_3__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_5__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_6__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_8__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_9__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_9__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_10__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_12__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_13__w_20__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)',
       'UHF__cwt_coefficients__coeff_14__w_20__widths_(2, 5, 10, 20)',
       'UHF__spkt_welch_density__coeff_5',
       'UHF__spkt_welch_density__coeff_8',
       'UHF__ar_coefficient__coeff_0__k_10',
       'UHF__ar_coefficient__coeff_1__k_10',
       'UHF__ar_coefficient__coeff_2__k_10',
       'UHF__ar_coefficient__coeff_3__k_10',
       'UHF__ar_coefficient__coeff_9__k_10',
       'UHF__ar_coefficient__coeff_10__k_10',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.2__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.0',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0',
       'UHF__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2',
       'UHF__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.2',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
       'UHF__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
       'UHF__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2',
       'UHF__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.4',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
       'UHF__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
       'UHF__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
       'UHF__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
       'UHF__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
       'UHF__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
       'UHF__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
       'UHF__fft_coefficient__attr_"real"__coeff_0',
       'UHF__fft_coefficient__attr_"abs"__coeff_0',
       'UHF__fft_coefficient__attr_"abs"__coeff_1',
       'UHF__fft_coefficient__attr_"abs"__coeff_2',
       'UHF__fft_coefficient__attr_"abs"__coeff_3',
       'UHF__fft_coefficient__attr_"abs"__coeff_5',
       'UHF__fft_coefficient__attr_"abs"__coeff_6',
       'UHF__fft_coefficient__attr_"abs"__coeff_7',
       'UHF__fft_coefficient__attr_"abs"__coeff_8',
       'UHF__fft_coefficient__attr_"abs"__coeff_9',
       'UHF__fft_coefficient__attr_"abs"__coeff_10',
       'UHF__fft_coefficient__attr_"abs"__coeff_11',
       'UHF__fft_coefficient__attr_"abs"__coeff_12',
       'UHF__fft_coefficient__attr_"abs"__coeff_13',
       'UHF__fft_coefficient__attr_"abs"__coeff_14',
       'UHF__fft_coefficient__attr_"abs"__coeff_15',
       'UHF__fft_coefficient__attr_"abs"__coeff_16',
       'UHF__fft_coefficient__attr_"abs"__coeff_17',
       'UHF__fft_coefficient__attr_"abs"__coeff_18',
       'UHF__fft_coefficient__attr_"abs"__coeff_19',
       'UHF__fft_coefficient__attr_"abs"__coeff_20',
       'UHF__fft_coefficient__attr_"abs"__coeff_21',
       'UHF__fft_coefficient__attr_"abs"__coeff_22',
       'UHF__fft_coefficient__attr_"abs"__coeff_23',
       'UHF__fft_coefficient__attr_"abs"__coeff_24',
       'UHF__fft_coefficient__attr_"abs"__coeff_25',
       'UHF__fft_aggregated__aggtype_"centroid"',
       'UHF__fft_aggregated__aggtype_"variance"',
       'UHF__fft_aggregated__aggtype_"skew"',
       'UHF__fft_aggregated__aggtype_"kurtosis"',
       'UHF__range_count__max_0__min_-1000000000000.0',
       'UHF__range_count__max_1000000000000.0__min_0',
       'UHF__approximate_entropy__m_2__r_0.1',
       'UHF__approximate_entropy__m_2__r_0.3',
       'UHF__approximate_entropy__m_2__r_0.5',
       'UHF__approximate_entropy__m_2__r_0.7',
       'UHF__approximate_entropy__m_2__r_0.9',
       'UHF__linear_trend__attr_"intercept"',
       'UHF__linear_trend__attr_"stderr"',
       'UHF__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
       'UHF__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
       'UHF__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"var"',
       'UHF__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"max"',
       'UHF__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
       'UHF__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
       'UHF__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
       'UHF__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"',
       'UHF__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
       'UHF__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
       'UHF__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"',
       'UHF__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"',
       'UHF__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
       'UHF__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
       'UHF__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"var"',
       'UHF__augmented_dickey_fuller__attr_"teststat"__autolag_"AIC"',
       'UHF__number_crossing_m__m_0', 'UHF__number_crossing_m__m_-1',
       'UHF__number_crossing_m__m_1', 'UHF__ratio_beyond_r_sigma__r_0.5',
       'UHF__ratio_beyond_r_sigma__r_1.5',
       'UHF__ratio_beyond_r_sigma__r_2.5',
       'UHF__ratio_beyond_r_sigma__r_3', 'UHF__ratio_beyond_r_sigma__r_5',
       'UHF__count_above__t_0', 'UHF__count_below__t_0',
       'UHF__lempel_ziv_complexity__bins_2',
       'UHF__lempel_ziv_complexity__bins_3',
       'UHF__lempel_ziv_complexity__bins_5',
       'UHF__lempel_ziv_complexity__bins_10',
       'UHF__lempel_ziv_complexity__bins_100',
       'UHF__fourier_entropy__bins_3', 'UHF__fourier_entropy__bins_5',
       'UHF__fourier_entropy__bins_10', 'UHF__fourier_entropy__bins_100',
       'UHF__permutation_entropy__dimension_3__tau_1',
       'UHF__permutation_entropy__dimension_4__tau_1',
       'UHF__permutation_entropy__dimension_5__tau_1',
       'UHF__permutation_entropy__dimension_6__tau_1',
       'UHF__mean_n_absolute_max__number_of_maxima_7', 'UV__sum_values',
       'UV__abs_energy', 'UV__mean_abs_change', 'UV__median', 'UV__mean',
       'UV__standard_deviation', 'UV__variation_coefficient',
       'UV__variance', 'UV__skewness', 'UV__kurtosis',
       'UV__root_mean_square', 'UV__absolute_sum_of_changes',
       'UV__count_above_mean', 'UV__count_below_mean',
       'UV__last_location_of_maximum', 'UV__first_location_of_maximum',
       'UV__first_location_of_minimum',
       'UV__percentage_of_reoccurring_values_to_all_values',
       'UV__percentage_of_reoccurring_datapoints_to_all_datapoints',
       'UV__sum_of_reoccurring_values',
       'UV__ratio_value_number_to_time_series_length', 'UV__maximum',
       'UV__absolute_maximum', 'UV__benford_correlation',
       'UV__time_reversal_asymmetry_statistic__lag_1', 'UV__c3__lag_1',
       'UV__c3__lag_2', 'UV__c3__lag_3', 'UV__cid_ce__normalize_True',
       'UV__cid_ce__normalize_False', 'UV__symmetry_looking__r_0.05',
       'UV__symmetry_looking__r_0.1',
       'UV__large_standard_deviation__r_0.15000000000000002',
       'UV__large_standard_deviation__r_0.2',
       'UV__large_standard_deviation__r_0.25', 'UV__quantile__q_0.1',
       'UV__quantile__q_0.2', 'UV__quantile__q_0.3',
       'UV__quantile__q_0.4', 'UV__quantile__q_0.6',
       'UV__quantile__q_0.7', 'UV__quantile__q_0.8',
       'UV__quantile__q_0.9', 'UV__autocorrelation__lag_1',
       'UV__autocorrelation__lag_3', 'UV__autocorrelation__lag_4',
       'UV__autocorrelation__lag_5', 'UV__autocorrelation__lag_6',
       'UV__autocorrelation__lag_7', 'UV__autocorrelation__lag_8',
       'UV__partial_autocorrelation__lag_1',
       'UV__partial_autocorrelation__lag_2',
       'UV__partial_autocorrelation__lag_4',
       'UV__partial_autocorrelation__lag_6', 'UV__number_cwt_peaks__n_1',
       'UV__number_peaks__n_1', 'UV__number_peaks__n_5',
       'UV__number_peaks__n_10',
       'UV__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_3__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_3__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_4__w_5__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_5__w_5__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_5__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_6__w_5__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_6__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_7__w_5__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_8__w_5__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_8__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_9__w_5__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_9__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_9__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_10__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_12__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_13__w_20__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)',
       'UV__cwt_coefficients__coeff_14__w_20__widths_(2, 5, 10, 20)',
       'UV__spkt_welch_density__coeff_2',
       'UV__spkt_welch_density__coeff_5',
       'UV__spkt_welch_density__coeff_8',
       'UV__ar_coefficient__coeff_0__k_10',
       'UV__ar_coefficient__coeff_1__k_10',
       'UV__ar_coefficient__coeff_2__k_10',
       'UV__ar_coefficient__coeff_4__k_10',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2',
       'UV__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.2',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
       'UV__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.4',
       'UV__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
       'UV__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
       'UV__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8',
       'UV__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
       'UV__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
       'UV__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
       'UV__fft_coefficient__attr_"real"__coeff_0',
       'UV__fft_coefficient__attr_"imag"__coeff_12',
       'UV__fft_coefficient__attr_"abs"__coeff_0',
       'UV__fft_coefficient__attr_"abs"__coeff_3',
       'UV__fft_coefficient__attr_"abs"__coeff_4',
       'UV__fft_coefficient__attr_"abs"__coeff_5',
       'UV__fft_coefficient__attr_"abs"__coeff_6',
       'UV__fft_coefficient__attr_"abs"__coeff_7',
       'UV__fft_coefficient__attr_"abs"__coeff_8',
       'UV__fft_coefficient__attr_"abs"__coeff_9',
       'UV__fft_coefficient__attr_"abs"__coeff_10',
       'UV__fft_coefficient__attr_"abs"__coeff_11',
       'UV__fft_coefficient__attr_"abs"__coeff_12',
       'UV__fft_coefficient__attr_"abs"__coeff_13',
       'UV__fft_coefficient__attr_"abs"__coeff_14',
       'UV__fft_coefficient__attr_"abs"__coeff_15',
       'UV__fft_coefficient__attr_"abs"__coeff_16',
       'UV__fft_coefficient__attr_"abs"__coeff_17',
       'UV__fft_coefficient__attr_"abs"__coeff_18',
       'UV__fft_coefficient__attr_"abs"__coeff_19',
       'UV__fft_coefficient__attr_"abs"__coeff_20',
       'UV__fft_coefficient__attr_"abs"__coeff_21',
       'UV__fft_coefficient__attr_"abs"__coeff_22',
       'UV__fft_coefficient__attr_"abs"__coeff_23',
       'UV__fft_coefficient__attr_"abs"__coeff_24',
       'UV__fft_coefficient__attr_"abs"__coeff_25',
       'UV__fft_aggregated__aggtype_"centroid"',
       'UV__fft_aggregated__aggtype_"variance"',
       'UV__fft_aggregated__aggtype_"skew"',
       'UV__fft_aggregated__aggtype_"kurtosis"',
       'UV__approximate_entropy__m_2__r_0.1',
       'UV__approximate_entropy__m_2__r_0.3',
       'UV__approximate_entropy__m_2__r_0.5',
       'UV__approximate_entropy__m_2__r_0.7',
       'UV__approximate_entropy__m_2__r_0.9',
       'UV__linear_trend__attr_"pvalue"',
       'UV__linear_trend__attr_"intercept"',
       'UV__linear_trend__attr_"stderr"',
       'UV__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"max"',
       'UV__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
       'UV__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
       'UV__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"var"',
       'UV__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"max"',
       'UV__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
       'UV__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
       'UV__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"',
       'UV__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
       'UV__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
       'UV__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"',
       'UV__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"',
       'UV__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
       'UV__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"var"',
       'UV__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"',
       'UV__augmented_dickey_fuller__attr_"usedlag"__autolag_"AIC"',
       'UV__ratio_beyond_r_sigma__r_0.5', 'UV__ratio_beyond_r_sigma__r_1',
       'UV__ratio_beyond_r_sigma__r_1.5',
       'UV__ratio_beyond_r_sigma__r_2.5', 'UV__ratio_beyond_r_sigma__r_3',
       'UV__ratio_beyond_r_sigma__r_5',
       'UV__lempel_ziv_complexity__bins_2',
       'UV__lempel_ziv_complexity__bins_3',
       'UV__lempel_ziv_complexity__bins_5',
       'UV__lempel_ziv_complexity__bins_100',
       'UV__fourier_entropy__bins_100',
       'UV__mean_n_absolute_max__number_of_maxima_7', 'UHL__sum_values',
       'UHL__abs_energy', 'UHL__mean_abs_change', 'UHL__median',
       'UHL__mean', 'UHL__standard_deviation', 'UHL__skewness',
       'UHL__kurtosis', 'UHL__root_mean_square',
       'UHL__absolute_sum_of_changes', 'UHL__last_location_of_minimum',
       'UHL__percentage_of_reoccurring_values_to_all_values',
       'UHL__percentage_of_reoccurring_datapoints_to_all_datapoints',
       'UHL__sum_of_reoccurring_data_points',
       'UHL__ratio_value_number_to_time_series_length', 'UHL__minimum',
       'UHL__benford_correlation',
       'UHL__time_reversal_asymmetry_statistic__lag_2',
       'UHL__time_reversal_asymmetry_statistic__lag_3', 'UHL__c3__lag_1',
       'UHL__c3__lag_2', 'UHL__c3__lag_3', 'UHL__cid_ce__normalize_True',
       'UHL__cid_ce__normalize_False', 'UHL__symmetry_looking__r_0.05',
       'UHL__symmetry_looking__r_0.1',
       'UHL__large_standard_deviation__r_0.2', 'UHL__quantile__q_0.1',
       'UHL__quantile__q_0.2', 'UHL__quantile__q_0.3',
       'UHL__quantile__q_0.4', 'UHL__autocorrelation__lag_1',
       'UHL__autocorrelation__lag_2', 'UHL__autocorrelation__lag_4',
       'UHL__autocorrelation__lag_5', 'UHL__autocorrelation__lag_6',
       'UHL__autocorrelation__lag_7', 'UHL__autocorrelation__lag_8',
       'UHL__autocorrelation__lag_9',
       'UHL__agg_autocorrelation__f_agg_"var"__maxlag_40',
       'UHL__partial_autocorrelation__lag_1',
       'UHL__partial_autocorrelation__lag_2',
       'UHL__partial_autocorrelation__lag_3',
       'UHL__number_cwt_peaks__n_1', 'UHL__number_peaks__n_1',
       'UHL__number_peaks__n_5', 'UHL__number_peaks__n_10',
       'UHL__binned_entropy__max_bins_10',
       'UHL__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_3__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_5__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_6__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_8__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_9__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_12__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_13__w_20__widths_(2, 5, 10, 20)',
       'UHL__cwt_coefficients__coeff_14__w_20__widths_(2, 5, 10, 20)',
       'UHL__ar_coefficient__coeff_0__k_10',
       'UHL__ar_coefficient__coeff_1__k_10',
       'UHL__ar_coefficient__coeff_2__k_10',
       'UHL__ar_coefficient__coeff_10__k_10',
       'UHL__change_quantiles__f_agg_"mean"__isabs_False__qh_0.2__ql_0.0',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.0',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0',
       'UHL__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2',
       'UHL__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.2',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2',
       'UHL__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.4',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.4',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6',
       'UHL__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
       'UHL__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
       'UHL__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
       'UHL__fft_coefficient__attr_"real"__coeff_0',
       'UHL__fft_coefficient__attr_"abs"__coeff_0',
       'UHL__fft_coefficient__attr_"abs"__coeff_2',
       'UHL__fft_coefficient__attr_"abs"__coeff_3',
       'UHL__fft_coefficient__attr_"abs"__coeff_4',
       'UHL__fft_coefficient__attr_"abs"__coeff_5',
       'UHL__fft_coefficient__attr_"abs"__coeff_6',
       'UHL__fft_coefficient__attr_"abs"__coeff_8',
       'UHL__fft_coefficient__attr_"abs"__coeff_9',
       'UHL__fft_coefficient__attr_"abs"__coeff_10',
       'UHL__fft_coefficient__attr_"abs"__coeff_11',
       'UHL__fft_coefficient__attr_"abs"__coeff_12',
       'UHL__fft_coefficient__attr_"abs"__coeff_13',
       'UHL__fft_coefficient__attr_"abs"__coeff_14',
       'UHL__fft_coefficient__attr_"abs"__coeff_16',
       'UHL__fft_coefficient__attr_"abs"__coeff_17',
       'UHL__fft_coefficient__attr_"abs"__coeff_18',
       'UHL__fft_coefficient__attr_"abs"__coeff_19',
       'UHL__fft_coefficient__attr_"abs"__coeff_20',
       'UHL__fft_coefficient__attr_"abs"__coeff_21',
       'UHL__fft_coefficient__attr_"abs"__coeff_22',
       'UHL__fft_coefficient__attr_"abs"__coeff_23',
       'UHL__fft_coefficient__attr_"abs"__coeff_24',
       'UHL__fft_coefficient__attr_"abs"__coeff_25',
       'UHL__fft_coefficient__attr_"angle"__coeff_0',
       'UHL__fft_aggregated__aggtype_"centroid"',
       'UHL__fft_aggregated__aggtype_"skew"',
       'UHL__approximate_entropy__m_2__r_0.1',
       'UHL__approximate_entropy__m_2__r_0.3',
       'UHL__approximate_entropy__m_2__r_0.5',
       'UHL__approximate_entropy__m_2__r_0.7',
       'UHL__approximate_entropy__m_2__r_0.9',
       'UHL__linear_trend__attr_"intercept"',
       'UHL__linear_trend__attr_"stderr"',
       'UHL__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
       'UHL__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
       'UHL__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
       'UHL__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
       'UHL__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
       'UHL__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
       'UHL__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
       'UHL__augmented_dickey_fuller__attr_"teststat"__autolag_"AIC"',
       'UHL__augmented_dickey_fuller__attr_"usedlag"__autolag_"AIC"',
       'UHL__number_crossing_m__m_0', 'UHL__number_crossing_m__m_-1',
       'UHL__number_crossing_m__m_1', 'UHL__ratio_beyond_r_sigma__r_1',
       'UHL__ratio_beyond_r_sigma__r_1.5',
       'UHL__ratio_beyond_r_sigma__r_3', 'UHL__ratio_beyond_r_sigma__r_5',
       'UHL__lempel_ziv_complexity__bins_2',
       'UHL__lempel_ziv_complexity__bins_3',
       'UHL__lempel_ziv_complexity__bins_5',
       'UHL__lempel_ziv_complexity__bins_10',
       'UHL__lempel_ziv_complexity__bins_100',
       'UHL__fourier_entropy__bins_2', 'UHL__fourier_entropy__bins_3',
       'UHL__fourier_entropy__bins_5', 'UHL__fourier_entropy__bins_10',
       'UHL__fourier_entropy__bins_100',
       'UHL__permutation_entropy__dimension_3__tau_1',
       'UHL__permutation_entropy__dimension_4__tau_1',
       'UHL__permutation_entropy__dimension_5__tau_1',
       'UHL__permutation_entropy__dimension_6__tau_1',
       'UHL__permutation_entropy__dimension_7__tau_1',]

def featureengineeringforfog(x_list, y_list, z_list):
    
    df = pd.DataFrame({'Time': range(1, len(x_list)+1), 'UHF': x_list, 'UV': y_list, 'UHL': z_list, 'dummy': [69420] * 100})
    features = tsfresh.extract_features(df, column_id="dummy", column_sort="Time")
    features = features.dropna(axis=1, how='any')
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how='any')
    df2 = features[featurelist]
    finallist = df2.iloc[0].tolist()

    
    fogfinal = scaler2.transform(np.array(finallist))
    return fogfinal

                      





if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)


