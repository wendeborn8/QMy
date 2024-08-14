import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from scipy import stats
import math
import copy

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn import linear_model

#%%

def parse_data(file, data, x, y, ye, x_col, y_col, ye_col, sep = ','):
    
    if x is not None and y is not None and ye is None:
        if ye is None:
            ye = np.array([0 for _ in y])
            
    elif x is not None and y is not None and ye is not None:
        np.array(x), np.array(y), np.array(ye)
    
    else:
        if data is None and isinstance(file, str):
            data = pd.read_csv(file, sep = sep)
        
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            
        if len(data.columns) < 2:
            print(f'Less than 2 columns found in {file}. Cannot extract at least x and y columns.')
            return
        else:
            columns = np.array(data.columns)
            
            if x_col in columns:
                x = data[x_col]
            else:
                x = data.iloc[:, 0]
                
            if y_col in columns:
                y = data[y_col]
            else:
                y = data.iloc[:, 1]
            
            if ye_col in columns:
                ye = data[ye_col]
            elif len(columns) >= 3:
                ye = data.iloc[:, 2]
            else:
                ye = [0 for _ in y]
                
    # Convert to numpy arrays
    x, y, ye = np.array(x), np.array(y), np.array(ye)
    
    # Remove nans
    nan_mask = np.array([False if any(pd.isna([_x, _y])) else True for _x, _y in zip(x, y)])
    x, y, ye = x[nan_mask], y[nan_mask], ye[nan_mask]
    ye = np.nan_to_num(ye, nan = np.nanmedian(ye))
    
    # Sort by x
    indices_sorted = np.argsort(x)
    x, y, ye = x[indices_sorted], y[indices_sorted], ye[indices_sorted]
        
    return x, y, ye

###

def polyfit_div(x, y, ye, degree):
    
    if degree == 'ransac':
        ransac = linear_model.RANSACRegressor(max_trials = 5000, stop_probability = 0.9999999)
        dye = 0.1 * np.nanpercentile(ye, 25)
        ransac.fit(x.reshape(-1, 1), y.reshape(-1, 1), sample_weight = 1 / (ye + dye))
        y_pred = np.array(ransac.predict(x.reshape(-1, 1))[:, 0])
        
    else:
        y_pred = np.poly1d(np.polyfit(x, y, deg = degree))(x)
        
    y /= (y_pred / np.nanmedian(y_pred))
        
    return y, ye

###

def GP_M(x, y, ye):
    
    amplitude = 1
    length_scale = 0.5
    rbf_kernel = amplitude**2 * RBF(length_scale = length_scale, length_scale_bounds = 'fixed')
    white_noise_kernel = WhiteKernel(noise_level=0.01, noise_level_bounds = (0.001, 0.1))
    
    # kernel = rbf_kernel
    kernel = rbf_kernel + white_noise_kernel
    
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 9)
    gp.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    
    return gp

###

def GP_Q(phase, y, ye):
    
    amplitude, alpha = 1, 1
    noise_level, noise_level_bounds = 0.1, 'fixed'
    length_scale, length_scale_bounds = int(len(phase) / 3), 'fixed'
    periodicity, periodicity_bounds = 1, 'fixed'
    
    rbf_kernel = RBF(length_scale=0.1)
    periodic_kernel = ExpSineSquared(length_scale = length_scale, periodicity = periodicity, length_scale_bounds = length_scale_bounds, periodicity_bounds = periodicity_bounds)
    
    kernel = amplitude**2 * periodic_kernel
        
    gp_Q = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp_Q.fit(phase.reshape(-1, 1), y.reshape(-1, 1))
    y_pred, y_pred_err = gp_Q.predict(phase.reshape(-1, 1), return_std = True)
    
    return y_pred, y_pred_err
    
###

def M(data = None, file = None, x = None, y = None, ye = None,
      x_col = 'mjd', y_col = 'flux', ye_col = 'flux_err', sep = ',',
      sigma = 5, fine = 1, skip = 1, polyfit_degree = 1, label = ''):
    
    fig, ((ax_lc, ax_resid), (ax_gp, ax_pct)) = plt.subplots(2, 2, figsize = (10, 6), sharex = True)
    
    # Get the x, y, and y_error data
    x, y, ye = parse_data(file = file, data = data, x = x, y = y, ye = ye, x_col = x_col, y_col = y_col, ye_col = ye_col, sep = sep)
    # x /= 10
    
    # Skip
    if skip > 1:
        x, y, ye = x[::skip], y[::skip], ye[::skip]
        
    # Fine (for sparsely sampled data)
    if fine > 1:
        x_fine = np.linspace(min(x), max(x), fine*len(x))
    else:
        x_fine = copy.deepcopy(x)
    
    # Normalize
    y_median = np.nanmedian(y)
    ye /= y_median
    y /= y_median
         
    # Divide off the long-term trend from the full light curve
    if polyfit_degree > 0:
        ax_lc.scatter(x, y, color = 'grey', alpha = 0.25, s = 1, zorder = 0) # Light curve before global trend division
        y, ye = polyfit_div(x = x, y = y, ye = ye, degree = polyfit_degree)
    
    
    # Fit a Guassian Process to the full Light Curve, Panel B
    gp = GP_M(x = x, y = y, ye = ye)
    y_pred, y_pred_err = gp.predict(x.reshape(-1, 1), return_std = True)
    y_pred_fine, y_pred_err_fine = gp.predict(x_fine.reshape(-1, 1), return_std = True)
    
    
    # Calculate the residuals and determine what data are good, Panel C
    y_resid, y_resid_up, y_resid_low = y - y_pred, y - (y_pred + y_pred_err), y - (y_pred - y_pred_err)
    y_resid_rms = np.sqrt(np.sum(y_resid**2) / len(y_resid))
    # y_resid_rms_up = np.sqrt(np.sum(y_resid_up**2) / len(y_resid_up))
    # y_resid_rms_low = np.sqrt(np.sum(y_resid_low**2) / len(y_resid_low))
    y_resid_good_mask = abs(y_resid) < sigma*y_resid_rms
    # y_resid_up_good_mask = abs(y_resid_up) < sigma*y_resid_rms_up
    # y_resid_low_good_mask = abs(y_resid_low) < sigma*y_resid_rms_low
    x_good, y_good, y_resid_good = x[y_resid_good_mask], y[y_resid_good_mask], y_resid[y_resid_good_mask]
    
    
    # Calculate top/bottom n% of the data and final M value, Panel D
    pct_5, pct_95 = np.nanpercentile(y_good, (10, 90))
    y_5, y_95 = y_good[y_good < pct_5], y_good[y_good > pct_95]
    x_5, x_95 = x_good[y_good < pct_5], x_good[y_good > pct_95]
    d_5top, d_5bot = np.nanmean(y_5), np.nanmean(y_95)
    d_5 = (d_5top + d_5bot) / 2
    d_med = np.nanmedian(y_good)
    sigma_d = np.nanstd(y_good)
    m = - (d_5 - d_med) / sigma_d
    
    
    # Do some plotting
    
    # Light curves, Panel A
    # ax_lc.scatter(x, y, c = 'black', s = 5, alpha = 0.75)
    ax_lc.errorbar(x, y, yerr = ye, color = 'black', markersize = 3, fmt = '.')
    ax_lc.tick_params(direction = 'in', which = 'both', right = True, top = True)
    ax_lc.set_title('Light Curve')
    
    # Light curve with fitted Gaussian Process, Panel B
    ax_gp.scatter(x, y, c = 'black', s = 5, alpha = 0.75)
    ax_gp.errorbar(x, y, yerr = ye, color = 'black', markersize = 3, alpha = 0.75, fmt = '.')
    ax_gp.plot(x_fine, y_pred_fine, c = 'tab:red', alpha = 1, linewidth = 1)
    # ax_gp.fill_between(x_fine, y_pred_fine - y_pred_err_fine, y_pred_fine + y_pred_err_fine, color = 'tab:red', alpha = 0.15, linewidth = 0)
    ax_gp.tick_params(direction = 'in', which = 'both', right = True, top = True)        
    ax_gp.set_title('Light Curve & GP Fit')
    
    # Residuals, Panel C
    ax_resid.scatter(x_good, y_resid[y_resid_good_mask], c = 'black', alpha = 0.75, s = 5)
    ax_resid.scatter(x[~y_resid_good_mask], y_resid[~y_resid_good_mask], c = 'green', alpha = 0.75, s = 5)
    ax_resid.axhline(sigma*y_resid_rms, color = 'tab:green', linewidth = 3, alpha = 0.5)
    ax_resid.axhline(-sigma*y_resid_rms, color = 'tab:green', linewidth = 3, alpha = 0.5)
    ax_resid.tick_params(direction = 'in', which = 'both', right = True, top = True)
    ax_resid.set_title('Residuals')
    
    # Top/Bottom 5%, Panel D
    ax_pct.scatter(x_good, y_good, c = 'black', s = 5, alpha = 0.75)
    ax_pct.scatter(x_5, y_5, c = 'tab:purple', alpha = 0.75, s = 5)
    ax_pct.scatter(x_95, y_95, c = 'tab:purple', alpha = 0.75, s = 5)
    ax_pct.axhline(pct_5, color = 'purple', alpha = 0.5, linewidth = 3, label = f'5%={pct_5:.2f}')
    ax_pct.axhline(pct_95, color = 'purple', alpha = 0.5, linewidth = 3, label = f'95%={pct_95:.2f}')
    ax_pct.axhline(d_5, color = 'tab:blue', alpha = 0.5, linewidth = 3, label = f'd$_5$={d_5:.2f}')
    ax_pct.axhline(d_med, color = 'tab:red', alpha = 0.5, linewidth = 3, label = f'd$_{{med}}$={d_med:.2f}')
    ax_pct.tick_params(direction = 'in', which = 'both', right = True, top = True)
    ax_pct.legend(ncols = 2, markerscale = 1, fontsize = 'small')
    ax_pct.set_title(f'{label} - M={m:.4f}')
    
    # Full Figure
    fig.supxlabel('Date [MJD]')
    fig.supylabel('Flux [Arb.]')
    fig.tight_layout()
    
    plt.show()
    
    return m

###

def Q(data = None, file = None, x = None, y = None, ye = None,
      x_col = 'mjd', y_col = 'flux', ye_col = 'flux_err', sep = ',',
      P = 1, 
      polyfit_degree = 1, skip = 1, fine = 1, label = ''):
    
    fig, ((ax_lc, ax_lcgp), (ax_gp, ax_resid)) = plt.subplots(2, 2, figsize = (10, 6), sharex = False)
    
    # Get the x, y, and y_error data
    x, y, ye = parse_data(file = file, data = data, x = x, y = y, ye = ye, x_col = x_col, y_col = y_col, ye_col = ye_col, sep = sep)
    
    # Skip
    if skip > 1:
        x, y, ye = x[::skip], y[::skip], ye[::skip]
    phase = (x % P) / P
    
    # Normalize
    y_median = np.nanmedian(y)
    ye /= y_median
    y /= y_median
                        
    # Divide off the long-term trend from the full light curve
    if polyfit_degree > 0:
        ax_lc.scatter(x, y, color = 'grey', alpha = 0.25, s = 1, zorder = 0) # Light curve before global trend division
        y, ye = polyfit_div(x = x, y = y, ye = ye, degree = polyfit_degree)
    ax_lc.errorbar(x, y, yerr = ye, fmt = '.', markersize = 3, color = 'black', elinewidth = 0.5)
    
    # Sort by phase
    phase, y, ye = parse_data(file = file, data = data, x = phase, y = y, ye = ye, x_col = x_col, y_col = y_col, ye_col = ye_col, sep = sep)
    
    # Fine (for sparsely sampled data)
    if fine > 1:
        phase_fine = np.linspace(0, 1, fine*len(x))
    else:
        phase_fine = copy.deepcopy(phase)
    
    # Make the 3-copy phase-folded light curve
    phase3, phase3_fine = np.append(phase, [phase+1, phase+2]), np.append(phase_fine, [phase_fine+1, phase_fine+2])
    y3, ye3 = np.append(y, [y, y]), np.append(ye, [ye, ye])
    
    
    # Calculations
    
    # , Panel A
    # ax_lc.scatter(x, y, color = 'black', alpha = 0.75, s = 5)
    
    # Get the center phase, Panel B
    y3_pred, y3_pred_err = GP_Q(phase = phase3, y = y3, ye = ye3)
    y3_pred_fine, y3_pred_err_fine = GP_Q(phase = phase3_fine, y = y3, ye = ye3)
    
    # Extend GP to full light curve, Panel C
    phase_mid_mask = np.logical_and(phase3 >= 1, phase3 < 2)
    y_pred, y_pred_err = y3_pred[phase_mid_mask], y3_pred_err[phase_mid_mask]
    y_pred_ext, y_pred_err_ext = list(y_pred) * math.ceil((max(x) - min(x)) / P), list(y_pred_err) * math.ceil((max(x) - min(x)) / P)
    y_pred_tot, y_pred_err_tot = np.array([pred for _x, pred in zip(x, y_pred_ext)]), np.array([pred for _x, pred in zip(x, y_pred_err_ext)])
    i_sorted = np.argsort(x)
    x, y, y_pred_tot, y_pred_err_tot = x[i_sorted], y[i_sorted], y_pred_tot[i_sorted], y_pred_err_tot[i_sorted]
    
    # Calculate residuals and final Q, Panel D
    resid = y - y_pred_tot
    rms_resid = np.sqrt(np.sum(resid**2) / len(resid))
    rms_raw = np.sqrt(np.sum((y-np.nanmean(y))**2) / len(y))
    sigma = np.nanmedian(ye) / np.nanmean(y)
    # sigma = 0.01
    # print(resid, rms_resid, rms_raw)
    q = (rms_resid**2 - sigma**2) / (rms_raw**2 - sigma**2)


    # All the plotting
    
    # Light Curve, Panel A
    ax_lc.tick_params(direction = 'in', which = 'both', right = True, top = True)
    ax_lc.set_title('Light Curve')
    
    # Folded Light curve w/ GP fit, Panel B
    ax_gp.scatter(phase3, y3, color = 'black', alpha = 0.75, s = 5)
    ax_gp.plot(phase3_fine, y3_pred_fine, color = 'tab:red', alpha = 1, linewidth = 2)
    ax_gp.fill_between(phase3_fine, y3_pred_fine + y3_pred_err_fine, y3_pred_fine - y3_pred_err_fine, 
                       color = 'tab:red', alpha = 0.15, linewidth = 0)
    ax_gp.tick_params(direction = 'in', which = 'both', right = True, top = True)
    ax_gp.set_title('3x Folded Light Curve & GP Fit')
    ax_gp.axvline(1, linestyle = ':', color = 'grey', alpha = 0.5)
    ax_gp.axvline(2, linestyle = ':', color = 'grey', alpha = 0.5)
    
    # Light curve w/ GP Fit, Panel C
    ax_lcgp.scatter(x, y, color = 'black', s = 5, alpha = 0.75)
    ax_lcgp.plot(x, y_pred_tot, color = 'tab:red', linewidth = 2, alpha = 1)
    ax_lcgp.fill_between(x, y_pred_tot + y_pred_err_tot, y_pred_tot - y_pred_err_tot, color = 'tab:red', alpha = 0.15)
    
    # Residuals, Panel D
    ax_resid.scatter(x, resid, color = 'black', alpha = 0.75, s = 5)
    ax_resid.axhline(rms_resid, color = 'tab:blue', alpha = 0.5, linewidth = 3, label = f'RMS$_{{residual}}$={rms_resid:.2f}')
    ax_resid.axhline(rms_raw, color = 'tab:green', alpha = 0.5, linewidth = 3, label = f'RMS$_{{raw}}$={rms_raw:.2f}')
    ax_resid.axhline(sigma, color = 'tab:purple', alpha = 0.5, linewidth = 3, label = f'$\sigma$={sigma:.2f}')
    ax_resid.set_title(f'{label} - P={P:.2f} - Q={q:.4f}')
    ax_resid.legend(ncols = 3, markerscale = 1, fontsize = 'small')
    
    # Full Figure
    fig.supxlabel('Date [MJD]')
    fig.supylabel('Flux [Normalized]')
    fig.tight_layout()
    plt.show()
    
    return q

