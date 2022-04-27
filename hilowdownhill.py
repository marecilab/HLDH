import numpy as np
from dipy.core.gradients import gradient_table
from dipy.io.image import load_nifti
from dipy.io.image import save_nifti
from dipy.reconst import dti
from dipy.core.ndindex import ndindex
from dipy.reconst.dti import lower_triangular
from dipy.reconst.dti import design_matrix
import dipy.reconst.fwdti as fwdti
from dipy.core.geometry import nearest_pos_semi_def
from dipy.segment.mask import median_otsu

def HLDL(indir, outdir, bthresh):
    #load the full dataset, here it has the basename "all", also generates an affine matrix which allows for nifti images to be saved later
    data,affine = load_nifti(indir + 'all.nii')
    #data, bval and bvec file should have the same basename; here the full dataset has the basename "all" 
    gt = gradient_table(indir +'all.bval',indir +'all.bvec')
    #generate B-matrix for full dataset
    all_B = design_matrix(gt)
    #remove dummy variable from the B-matrix, it is automatically included by DIPY but will break calculations later on
    all_B = all_B[:,:6]
    #generate low b-value gradient table, using b-values <= bthresh
    low_gt = gradient_table(gt.bvals[np.where(gt.bvals<=bthresh)],gt.bvecs[np.where(gt.bvals<=bthresh)])
    #generate B-matrix for the low b-value data
    low_B = design_matrix(low_gt)
    #remove dummy variable from the B-matrix, it is automatically included by DIPY but will break calculations later on
    low_B = low_B[:,:6]
    #generate high b-value gradient table, using b-values >= bthresh excluding b=0 volumes
    high_gt = gradient_table(gt.bvals[np.where(gt.bvals>=bthresh)], gt.bvecs[np.where(gt.bvals>=bthresh)])
    #create a mask, and mask the input data
    data, mask = median_otsu(data, vol_idx=[0,1], median_radius=4, numpass=4, dilate=1)
    #split data into high/low datasets corresponding to volumes above/below bthresh
    high_data = data[:,:,:,np.ndarray.flatten(np.array(np.where(gt.bvals>=bthresh)))]
    low_data = data[:,:,:,np.ndarray.flatten(np.array(np.where(gt.bvals<=bthresh)))]
    #fit the high data using single-compartment DTI model, assuming FW-compartment contribution is negligible
    dtimodel = dti.TensorModel(high_gt)
    dtifit = dtimodel.fit(high_data,mask)
    #generate lower triangular of the 3x3 symmetric diffusion tensor
    dt = lower_triangular(dtifit.quadratic_form)
    #generate non-diffusion weighted volume, taking mean of all b0 images
    S0 = np.mean(data[:,:,:,gt.b0s_mask], axis=-1)
    #generate lower triangular of an isotropic tensor which sets the ADC of water = 3e-3
    diso = dt.copy()
    diso[..., 0] = diso[..., 2] = diso[..., 5] = 3e-3
    diso[..., 1] = diso[..., 3] = diso[..., 4] = 0
    #fit the low b-value data using the DTI model, this is not part of the HLDL method, but is part of a processing step I borrowed from dipy to prevent bad calculations
    dtimodel2 = dti.TensorModel(low_gt)
    dtifit2 = dtimodel2.fit(low_data,mask)
    #initialize the p, q, and f arrays
    p = np.zeros(mask.shape + (low_gt.bvals.shape[0],))
    q = np.zeros(mask.shape + (low_gt.bvals.shape[0],))
    #f is the free-water volume fraction array
    f = np.zeros(mask.shape)
    #generate index, which iterates over every voxel of the given array shape (i.e. [0,0,0], [0,0,1], [0,0,2] ... [128,128,72])
    index = ndindex(mask.shape)
    for v in index:
        if dtifit2.md[v] < 2.9e-3 and np.mean(high_data[v], axis=-1) > 1e-6 and S0[v] > 1e-6:
            if mask[v]==True:
                #these equations are straight from the HLDL paper (eq. 8)
                p[v] = np.exp(np.dot(diso[v], low_B.T)) - (np.exp(np.dot(dt[v], low_B.T)))
                q[v] = (low_data[v] / S0[v]) - (np.exp(np.dot(dt[v], low_B.T)))
                f[v] = np.dot(p[v],q[v]) / np.dot(p[v],p[v])
        #if the mean diffusivity in the voxel approaches that of free-water, set free-water = 1
        else:
            if dtifit2.md[v] > 2.9e-3:
                f[v] = 1
    #at this point the high/low part of the HLDL method is complete, a diffusion tensor (dt) has been generated from the tissue using the high data, which was then used to generate the free-water volume fraction (f) from the low data
    #now the downhill part of the HLDL method minimzes the residual of the fit
    #initialize the arrays
    pred_sig = np.zeros(mask.shape + (gt.bvals.shape[0],))
    #these arrays are named 2 to distinguish them from the previous arrays; we have to have two arrays for each variable in order to compare their residuals and choose the best one, the one which minimizes the residual
    p_2 = np.zeros(mask.shape + (gt.bvals.shape[0],))
    q_2 = np.zeros(mask.shape + (gt.bvals.shape[0],))
    f_2 = np.zeros(mask.shape)
    dt_2 = np.zeros(dt.shape)
    #norm1/norm2 are the square root of the sum of squared errors of the data, the residuals of the fit
    norm1 = np.zeros(mask.shape)
    norm2 = np.zeros(mask.shape)
    #set the desired number of iterations for the minimization loop, it does not need more than 10
    iter = 10
    for i in range(0, iter):
        index = ndindex(mask.shape)
        for v in index:
            if dtifit2.md[v] < 2.9e-3 and np.mean(data[v], axis=-1) > 1e-6 and S0[v] > 1e-6:
                if mask[v] == True:
                    #Calculate the predicted signal of the tissue compartment using the values of f, this is eq. 7 of the HLDL paper
                    pred_sig[v] = ((data[v] - (S0[v] * f[v] * np.exp(np.dot(diso[v], all_B.T))))/(1 - f[v]))
            #if the mean-diffusivity of the voxel is close to that of free-water, the signal calculation is done using the equation below, notice if FW = 1 in the equation above, the equation is undefined
            else:
                if dtifit2.md[v] > 2.9e-3:
                    pred_sig[v] = S0[v] * np.exp(np.dot(diso[v], all_B.T))
        #fit the predicted signal of the tissue compartment using the single-compartment DTI model, assuming no contribution from free-water
        dtimodel3 = dti.TensorModel(gt)
        dtifit3 = dtimodel3.fit(pred_sig,mask)
        #generate the quadratic form of the diffusion tensor from the DTI fit (the 3x3 symmetric diffusion tensor)
        quadform = dtifit3.quadratic_form
        index = ndindex(mask.shape)
        for v in index:
            if dtifit3.md[v] < 2.9e-3 and np.mean(data[v], axis=-1) > 1e-6 and S0[v] > 1e-6:
                if mask[v] == True:
                    #generate the lower triangular of the diffusion tensor, ensuring positive semi-definite tensors
                    dt_2[v] = lower_triangular(nearest_pos_semi_def(quadform[v]))
                    #p,q,f are calculated the same way as above (eq. 8), different from before because here we are using the new dt_2 and the whole dataset
                    p_2[v] = np.exp(np.dot(diso[v], all_B.T)) - (np.exp(np.dot(dt_2[v], all_B.T)))
                    q_2[v] = (data[v] / S0[v]) - (np.exp(np.dot(dt_2[v], all_B.T)))
                    f_2[v] = np.dot(p_2[v],q_2[v]) / np.dot(p_2[v],p_2[v])
                    #calculate the residuals, comparing the previous (dt, f) pair with the new (dt_2, f_2) pair, these functions correspond to (eq. 5) of the HLDL paper
                    norm1[v] = np.sqrt(np.sum((np.square(data[v] - (S0[v] * ((1 - f[v]) * np.exp(np.dot(dt[v], all_B.T)) + f[v] * np.exp(np.dot(diso[v], all_B.T)))))), axis=-1))
                    norm2[v] = np.sqrt(np.sum((np.square(data[v] - (S0[v] * ((1 - f_2[v]) * np.exp(np.dot(dt_2[v], all_B.T)) + f_2[v] * np.exp(np.dot(diso[v], all_B.T)))))), axis=-1))
            else:
                if dtifit3.md[v] > 2.9e-3:
                    f[v] = 1
            #if at a voxel the residual1 is greater than the residual2, assign the values of (dt_2, f_2) to the values of (f, dt) at that voxel; dt_2 and f_2 continue to be regenerated as the optimization loop runs through iterations, so we want to save the most recent values which minimized the residuals in the dt, f arrays
            if norm1[v] >= norm2[v]:
                f[v] = f_2[v]
                dt[v] = dt_2[v]
        #in theory, if the residual no longer reduces, the loop should break; this command breaks the loop if the residual value at every voxel in both arrays are within a specified threshold of each other; this step can probably be improved somehow because I do not think it is ever the case that the residuals in all of the voxels are within this threshold
        if np.allclose(norm1, norm2, 1, 1):
            break
    #initialize the predicted signal array
    pred_sig = np.zeros(mask.shape + (gt.bvals.shape[0],))
    index = ndindex(mask.shape)
    for v in index:
        if dtifit3.md[v] < 2.9e-3 and np.mean(data[v], axis=-1) > 1e-6 and S0[v] > 1e-6:
            if mask[v] == True:
                #calculate the predicted signal using the final calculated values of dt and f and the two-compartment FW-DTI model, (eq. 4) of the HLDL paper)
                pred_sig[v] = S0[v] * ((1 - f[v]) * np.exp(np.dot(dt[v], all_B.T)) + f[v] * np.exp(np.dot(diso[v], all_B.T)))
        else:
            if dtifit3.md[v] > 2.9e-3:
                pred_sig[v] = S0[v] * np.exp(np.dot(diso[v], all_B.T))
    #fit the predicted signal using non-linear least squares minimization with DIPY's fwdti model
    fwdtimodel = fwdti.FreeWaterTensorModel(gt)
    fwdtifit = fwdtimodel.fit(pred_sig,mask)
    #save the free-water image as a nifti file
    f = fwdtifit.f
    save_nifti(outdir + "freewater.nii", f, affine)
