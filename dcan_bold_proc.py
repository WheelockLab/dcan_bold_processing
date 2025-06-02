'''
This script takes the output from ABCD-HCP-pipeline, load the minimally preprocessed CIFTI file and apply confound removal
Adapted from the MATLAB script dcan_bold_processing.m (https://github.com/DCAN-Labs/dcan_bold_processing/blob/main/matlab_code/dcan_signal_processing.m)
All the dependencies are included in this file (no other script is required)

Potential issue: the GSR version is slightly different from the saved file in the original folder (ABCD from DCAN) potentially because of the differences in interpolation and padding?

Author:
    Jiaxin Cindy Tu (tu.j@wustl.edu), Jim Pollaro (jimp@wustl.edu)
'''
import json, argparse, os, scipy, logging, re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from pathlib import Path

def main(input_path, subject_id, task_id, output_path, session_id=None, settings_file=None, skip_save_figures=False, GSR=False, FD_type=1, brain_radius=50, bandpass_filter_order=2, highpass_cutoff=0.08, lowpass_cutoff=0.009, skip_seconds=5, fd_threshold=0.3):
    # subject_id = args.subject_id
    # task_id = args.task_id
    # input_path = args.input_path
    # output_path = args.output_path
    # FD_type = args.FD_type
    # brain_radius = args.brain_radius # This was in Cindy's code, but No reference. leaving for now. JP 5/25
    # GSR = args.GSR if args.GSR else False

    
    # Part I: parse argument
    # Load json
    has_sessions = True if session_id is not None else False
    run_num = None
    if settings_file:
        settings_file = settings_file
    else:
        # This part isn't working correctly
        subject_dir = f'{input_path}/sub-{subject_id}'
        settings_path = subject_dir
        subject_contents = os.listdir(subject_dir)
        for content in subject_contents:
            if content[:4] == 'ses-':
                settings_path = f'{input_path}/sub-{subject_id}/ses-{session_id}/'
                has_sessions = True
        settings_file = f'{settings_path}files/MNINonlinear/Results/DCANBOLDProc_v4.0.0_mat_config.json'
    with open(settings_file) as file:
        input_json = json.load(file)

    run_match = re.match(r'.*\/Results\/.*_run-([0-9]{8,12})\/DCANBOLDProc_v4\.0\.0\/DCANBOLDProc_v4\.0\.0_mat_config\.json', settings_file)
    if run_match:
        run_num = run_match.group(1)

    logging.basicConfig(
        filename=os.path.join(output_path, "output.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if has_sessions:
        if run_num is None:
            logging.info(f'Now processing subject sub-{subject_id} session-{session_id} task-{task_id}')
        else:
            logging.info(f'Now processing subject sub-{subject_id} session-{session_id} task-{task_id} run {run_num}')
    else:
        logging.info(f'Now processing subject sub-{subject_id} task-{task_id}')

    # Set default filenames and folders
    RESULTS_SUFFIX = 'DCANBOLDProc_v4.0.0'
    base_output_path = f'{output_path}/sub-{subject_id}'
    if has_sessions:
        base_output_path += f'/ses-{session_id}'
        if run_num is not None:
            base_output_path += f'/run-{run_num}'
    base_output_path += f'/task-{task_id}'
    fmri_name_split = input_json['fMRIName'].split('_')
    if len(fmri_name_split) > 2:
        for name in fmri_name_split:
            if name[:4] == '/run-':
                base_output_path += f' {name}'
                break
    base_output_path += f'/{RESULTS_SUFFIX}'
    base_output_path = Path(base_output_path)
    
    if GSR:
        output_path = f'{base_output_path}/gsr'
    else:
        output_path = f'{base_output_path}/no_gsr'

    input_json['GSR'] = GSR
    input_json['brain_radius_mm'] = brain_radius
    input_json['FD_type'] = FD_type
    input_json['path_cii'] = str.replace(input_json['path_cii'], '/output/', input_path)
    input_json['file_mov_reg'] = str.replace(input_json['file_mov_reg'], '/output/', input_path)
    input_json['file_vent'] = str.replace(input_json['file_vent'], '/output/', input_path)    
    input_json['file_wm'] = str.replace(input_json['file_wm'], '/output/', input_path)
    input_json['config'] = f'{output_path}/DCANBOLDProc_v4.0.0_mat_config.json'

    input_json.pop('path_ex_sum')
    input_json.pop('path_wb_c')
    
    # Save parameters in results folder
    directory_path = Path(output_path)
    directory_path.mkdir(parents=True, exist_ok=True)
    with open(input_json['config'], 'w') as file:
        json.dump(input_json, file, sort_keys=True, indent=4)

    # Part II: Load and prepare preprocessed data
    image = nib.load(input_json['path_cii'])
    fdata = image.get_fdata()
    DVAR_pre_reg = calculate_dvars_from_cifti(fdata)

    # Global signals (lines 52, 58, 83 in dcan_bold_processing.m)
    if GSR:
        wm = np.loadtxt(input_json['file_wm'])
        vent = np.loadtxt(input_json['file_vent'])
        WB = np.mean(fdata, axis=1) # Global signal in gray matter
        Glob = np.concatenate(
            (wm.reshape(-1, 1), vent.reshape(-1, 1), WB.reshape(-1, 1)), axis=1
        )
        dGlob = np.vstack(([0, 0, 0], np.diff(Glob, axis=0)))

        if not skip_save_figures:
            plt.figure(figsize=(8, 4))
            plt.plot(Glob, linewidth=1, label=('white matter', 'ventricle', 'grayordinate'))
            plt.title('Global Signal')
            plt.legend()
            plt.savefig(
                os.path.join(output_path, 'globalsignal_trace.png'),
                format='png',
                dpi=300
            )
            plt.close()

    # Get TR, in seconds (line 34 in dcan_signal_processing.m)
    TR = nib.load(input_json['path_cii']).header.get_axis(0).step 
    if TR > 20:
        # if TR in ms, convert to s
        TR /= 1000
    logging.info("TR = " + str(TR))

    # Load movement
    assert os.path.isfile(input_json['file_mov_reg']), 'Movement Regressor File not Found'
    filtered_movement_regressors = np.loadtxt(input_json['file_mov_reg'])
    FR = make_friston_regressors(
        filtered_movement_regressors, input_json['brain_radius_mm']
    ) # 24-param Friston Regressors

    # Calculate FD (line 73 in dcan_bold_processing.m)
    FD, meanFD = calc_FD(FR[:,:6], FD_type=FD_type)
    FD_file_name = os.path.join(
        output_path, str.replace(Path(input_json['file_mov_reg']).name, '.txt', '_FD.txt')
    )
    np.savetxt(FD_file_name, FD)
    logging.info("Mean FD = " + str(meanFD))

    # Calculate kept frames (line 78 in dcan_bold_processing.m)
    if fd_threshold and fd_threshold > 0:
        keepframe = FD <= fd_threshold
    else:
        keepframe = np.full(len(FD), True)
    skip_frames = np.int8(np.floor(skip_seconds / TR))
    keepframe[:skip_frames] = False

    if not skip_save_figures:
        plt.figure(figsize=(8,4))
        for j in np.where(keepframe==0)[0]:
            plt.axvline(x=j, color=[.5, .5, .5], alpha=0.5)
        plt.plot(FD, linewidth=1)
        plt.title('FD')
        plt.savefig(os.path.join(output_path, 'FD_trace.png'), format='png', dpi=300)
        plt.close()

    # Concatenate regressors
    if GSR:
        R = np.concatenate(
            (Glob, dGlob, FR), axis = 1
        ) # with GSR # N.B. this uses 30 parameter, for 36 parameter it would also involve the square of Glob and dGlob
    else:
        R = FR

    # Part III: Regression and bandpass filter to give cleaned BOLD data
    # Demean and detrend regressors
    R = R - np.mean(R[keepframe, :], axis=0)
    R = detrend_manual(R, keepframe)
    data_dd = fdata - np.mean(fdata[keepframe, :], axis=0)
    data_dd = detrend_manual(data_dd, keepframe)

    if not skip_save_figures:
        # grayplots
        crange = [-200, 200] # using +/-2% as Jonathan Power's papers, but DCAN used +/-6%
        plt.figure(figsize=(8,4))
        im = plt.imshow(data_dd.transpose(), aspect='auto', cmap='gray')
        plt.colorbar(location='right')
        im.set_clim(crange)
        plt.plot(np.where(keepframe==0)[0], np.repeat(0, sum(keepframe==0)), 'r|')
        plt.yticks([])
        plt.xlabel('TR (in seconds)')
        plt.savefig(
            os.path.join(output_path, 'grayplots_all.png'),
            format='png',
            dpi=300
        )
        plt.close()

        plt.figure(figsize=(8,4))
        X = data_dd.copy()
        X[np.where(keepframe==1)[0]] = np.NaN
        im = plt.imshow(X.transpose(), aspect='auto', cmap='gray')
        plt.colorbar(location='right')
        im.set_clim(crange)
        plt.yticks([])
        plt.xlabel('TR (in seconds)')
        plt.savefig(
            os.path.join(output_path, 'grayplots_removed.png'),
            format='png',
            dpi=300
        )
        plt.close()

        plt.figure(figsize=(8,4))
        X = data_dd.copy()
        X[np.where(keepframe==0)[0]] = np.NaN
        im = plt.imshow(X.transpose(), aspect='auto', cmap='gray')
        plt.colorbar(location='right')
        im.set_clim(crange)
        plt.yticks([])
        plt.xlabel('TR (in seconds)')
        plt.savefig(
            os.path.join(output_path, 'grayplots_retained.png'),
            format='png',
            dpi=300
        )
        plt.close()

    # Nuisance regression (line 121 in dcan_signal_processing.m)
    b, _, _, _ = np.linalg.lstsq(R[keepframe, :], data_dd[keepframe, :], rcond=None)
    data_postreg = data_dd - R @ b
    DVAR_post_reg = calculate_dvars_from_cifti(data_postreg)

    # Linear interpolation before bandpass (line 144 in dcan_signal_processing.m)
    x = np.where(keepframe)[0]
    x_removed = np.where(keepframe == 0)[0]
    data_interpolated = data_postreg.copy()
    
    x_outsidebound = (x_removed < x[0]) | (x_removed > x[-1])
    y_removed = np.apply_along_axis(
        lambda col: np.interp(x_removed, x, col[keepframe]), axis=0, arr=data_postreg
    )
    # Replace extrapolated points with mean of retained (6 params)
    y_mean = np.mean(data_postreg[keepframe, :], axis=0)
    y_removed[x_outsidebound, :] = y_mean

    
    data_interpolated[keepframe==0, :] = y_removed

    # Bandpass filter with manual zero padding
    fs = 1 / TR  # Sampling frequency
    fNy = fs / 2  # Nyquist frequency
    b_filt, a_filt = scipy.signal.butter(
        bandpass_filter_order / 2,
        np.array([lowpass_cutoff, highpass_cutoff]) / fNy,
        "bandpass",
    )

    # Zero-pad the data for filtering by concatenating rows of zeros on either side of the data
    padding = np.zeros_like(
        data_interpolated
    )  # Create a padding array of the same shape as Rr_int
    pad_amt = padding.shape[0]  # Number of rows to pad

    # Concatenate padding rows on top and bottom of Rr_int
    temp = np.vstack((padding, data_interpolated, padding))

    # Apply the filtfilt function (zero-phase filtering)
    data_filtered = scipy.signal.filtfilt(
        b_filt, a_filt, temp, axis=0, padtype=None
    )  
    
    # Apply filtering along the rows (axis=0)
    data_filtered = data_filtered[pad_amt:-pad_amt]
    DVAR_post_filter = calculate_dvars_from_cifti(data_filtered)

    if not skip_save_figures:
        plt.figure(figsize=(8, 4))
        plt.plot(DVAR_pre_reg, linewidth=1, label="DVARS pre regression", color="b")
        plt.plot(DVAR_post_reg, linewidth=1, label="DVARS post regression", color="r")
        plt.plot(DVAR_post_filter, linewidth=1, label="DVARS post filtered", color="g")
        plt.xlabel("TR")
        plt.legend()
        plt.savefig(
            os.path.join(output_path, "DVARS_trace.png"), format="png", dpi=300
        )
        plt.close()

    if GSR:
        logging.info("Saving cleaned data (w/GSR)")
    else:
        logging.info("Saving cleaned data (wo/GSR)")

    # Save filtered data
    ax1 = image.header.get_axis(0)
    ax2 = image.header.get_axis(1)
    header = (ax1, ax2)
    output_image = nib.cifti2.cifti2.Cifti2Image(np.single(data_filtered), header)
    output_image.to_filename(
        os.path.join(
            output_path,
            input_json["FNL_preproc_CIFTI_basename"] + ".dtseries.nii",
        )
    )
    return
    
# All dependency functions
def parse_arguments():
    parser = argparse.ArgumentParser(description='confound removal for BOLD fMRI data')
    parser.add_argument(
        'input_path',
        help='Path to input data, i.e. data preprocessed and mapped to standard space',
        type=str
    )
    parser.add_argument(
        'subject_id',
        type=str,
        help="The subject number without 'sub-'"
    )
    parser.add_argument(
        'task_id',
        type=str,
        help="The folder name for the scan without 'task-', i.e. 'rest01' for 'task-rest01'"
    )
    parser.add_argument(
        'output_path',
        help='Path to store results',
        type=str
    )
    parser.add_argument(
        '--session_id',
        type=int,
        help="Session id for the subject without 'ses-' (defaults to 0)",
        required=False,
        default=0
    )
    parser.add_argument(
        '--settings_file',
        type=str,
        help='path to json file for settings (defaults to DCANBOLDProcessing location assuming that pipeline was run)',
        required=False
    ),
    parser.add_argument(
        '--skip_save_figures',
        help='Skip saving output figures',
        action='store_true',
        required=False
    )
    parser.add_argument(
        '--GSR',
        help='Include Global Signal Regression (CSF, WM, GM, and derivatives)',
        action='store_true',
        required=False
    ),
    parser.add_argument(
        '--FD_type',
        type=int,
        help='L1 [1] (default) or L2 [2]',
        default=1
    )
    parser.add_argument(
        '--brain_radius',
        type=int,
        default=50,
        help='estimage of brain radius in mm. Used for FD and movement regressors (default: 50)'
    )
    parser.add_argument(
        '--bandpass_filter_order',
        type=int,
        default=2,
        help='Order for the bandpass filter used for respiration filter (default: 2)',
    )
    parser.add_argument(
        '--highpass_cutoff',
        type=float,
        default=0.08,
        help='Highpass filter cutoff frequence in Hertz (default: 0.08)'
    )
    parser.add_argument(
        '--lowpass_cutoff',
        type=float,
        default=0.009,
        help='Lowpass filter cutoff frequence in Hertz (default: 0.009)'
    )
    parser.add_argument(
        '--skip_seconds',
        type=int,
        default=5,
        help='Seconds to skip at beginning (default: 5)'
    )
    parser.add_argument(
        '--fd_threshold',
        type=float,
        default=0.3,
        help='Framewise displacement threshold (default: 0.3)'
    )
    return parser.parse_args()

def calculate_dvars_from_cifti(data):
    """
    This function calculates DVARS (Derivative of Variance) based on grayordinates (WM and non-brain excluded).

    Parameters:
    data (ndarray): 2D numpy array with shape (tr, g), where g represents the number of grayordinates and tr is the number of time points.

    Returns:
    dvars (float): The calculated DVARS value.
    """
    # Check size and transpose if needed
    num_timepoints, num_grayordinates = data.shape
    if num_grayordinates < num_timepoints:
        data = data.T
        print("data transposed due to timepoints > grayordinates, double check input")

    # Calculate differences across timepoints
    data_diff = np.diff(data, axis=0)

    # Calculate DVARS as the root mean square of the differences
    dvars = np.hstack((np.nan, np.sqrt(np.mean(data_diff**2, axis=1))))

    return dvars

def make_friston_regressors(R, hd_mm):
    """
    This function takes a matrix `MR` of 6 degrees of freedom (DOF) movement correction
    parameters and calculates the corresponding 24 Friston regressors.

    Parameters:
    -----------
    MR : numpy array of shape (r, c)
        A matrix where r is the number of time points and c are the 6 DOF movement regressors.
        If the number of columns is more than 6, only the first 6 columns are considered.

    hd_mm : float, optional
        The head radius in mm. Default is 50 mm.

    Returns:
    --------
    FR : numpy array of shape (r, 24)
        A matrix containing 24 Friston regressors.
    """
    MR = R[:, :6]
    MR[:, 3:] = MR[:, 3:] * np.pi * hd_mm / 180
    # Calculate the first part of the Friston regressors (MR and MR^2)
    FR = np.hstack([MR, MR**2])

    # Create a dummy array for the temporal derivatives (lagged version of FR)
    dummy = np.zeros_like(FR)
    dummy[1:, :] = FR[:-1, :]  # shift FR by one time step
    dummy[0, :] = 0  # set the first row to 0

    # Concatenate the original FR and the lagged version
    FR = np.hstack([FR, dummy])
    return FR

def calc_FD(R, FD_type=1):
    '''
    This function calculates framewise displacement (Power et al. 2012 Neuroimage)
    The columns 3-6 (angular displacement) is assumed to be already converted to mm before passed in this function
    '''

    dR = np.diff(R, axis=0)  # First-order derivative
    ddR = np.diff(dR, axis=0)  # Second-order derivative
    if FD_type == 1:
        # L1-norm - sum of absolute values of first-order derivatives
        FD = np.sum(np.absolute(dR), axis=1)
        meanFD = np.mean(FD)
        FD = np.hstack(
            (
                np.zeros(
                    1,
                ),
                FD,
            )
        )  # Pad zeros to make it the same length as the original data
    elif FD_type == 2:
        # L2-norm - sum of absolute values of second-order derivatives
        FD = np.sum(np.absolute(ddR), axis=1)
        meanFD = np.mean(FD)
        FD = np.hstack(
            (
                np.zeros(
                    2,
                ),
                FD,
            )
        )  # Pad zeros to make it the same length as the original data
    return FD, meanFD

def detrend_manual(data, keepframe):
    '''
    Remove linear trends
    '''
    detrended_data = data.copy()
    time_points = np.where(keepframe)[0]
    time_points_all = np.array(range(keepframe.shape[0]))

    # Create the design matrix for linear regression (constant + linear term)
    X = np.vstack(
        [time_points, np.ones(len(time_points))]
    ).T  # Shape (len(keepframe), 2)
    Xall = np.vstack([time_points_all, np.ones(len(time_points_all))]).T

    # Perform the linear regression for all columns at once using least squares
    # Y is the data[keepframe, :] with shape (len(keepframe), n)
    Y = data[keepframe, :]

    # Compute the least squares solution to find the slope and intercept for each column
    beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)  # beta shape is (2, n)

    # Calculate the trend for each column using the coefficients
    trend = Xall @ beta  # Shape (len(keepframe), n)

    # Subtract the trend from the data at the all indices
    detrended_data -= trend

    return detrended_data

if __name__ == '__main__':
    main(*parse_arguments())