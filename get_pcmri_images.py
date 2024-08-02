# ------------------------------------------------------------------------------
# NFFDy SP 2024 - Turbulence and flow-MRI
#
# - extract magnitude and phase images from k-space scans
# - unwrap phase difference
# - compute velocity image
# - ensure axisymmetry
# - write machine precision axisymm. images to be used by revolve.py (3D proj)
# ------------------------------------------------------------------------------
# Authors: Alexandros Kontogiannis, Cambridge 2024
#          Emily Manchester,        Leeds     2024
# ------------------------------------------------------------------------------

import os
import numpy as np
import scipy.io as scio
import scipy.fft as sfft
from skimage.segmentation import chan_vese
import matplotlib.pyplot as plt

cmap = 'gray'

# USER-INPUT 
# ------------------------------------------------------------------------------
VENCs = {'34': 4.0, '35': 2.5, '36': 0.05} # m/s
dataset_folder     = '../run3_2024-07-25/'
flow_scan_id       = 30
no_flow_scan_id    = 34
scan_repetitions   = 64
VENC               = VENCs[str(no_flow_scan_id)]
flow_range         = {'flow': VENC, 'no-flow': VENC} # m/s
fov                = [80e-3,40e-3]                   # m
voxel_spacing      = [2e-4, 2e-4, 1e-3]              # m, [0.2, 0.2, 1] mm
plot_imtrx_to_imgs = False
data_type          = 'cimage' # choose between cimage (complex image) or kspace
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def compute_velocity(phases,vel_range,comp):
    # Hadamard sequence
    comp_polarities = {'u_r': [-1, 1, 1,-1],\
                       'u_p': [-1, 1,-1, 1],\
                       'u_s': [-1,-1, 1, 1]}
    polarities = comp_polarities[comp]
    vel = np.sum([pol*phi for pol,phi in zip(polarities,phases)],axis=0)
    # unwrap phase difference
    vel = vel - 2*np.pi*( (vel//np.pi - 1)//2 + 1)
    # velocity map constant (VENC)
    vel *= vel_range/np.pi
    return vel

def compute_mask(avg_magn,max_iter=500):
    # Chan-Vese segmentation: segm[0]: mask, segm[1]: level-set
    segm = chan_vese(avg_magn, mu              = 0.1     , lambda1 = 1,\
                               lambda2         = 1        ,tol     = 1e-5,\
                               dt      = 0.5, max_iter = max_iter,\
                               init_level_set  = "checkerboard",\
                               extended_output = True)
    return segm[0],segm[1]

def  get_flow_and_magnitude_images(flow_scan_id,no_flow_scan_id,\
                                   scan_rep,flow_range,vel_component,with_plot):
    print('> ** Component '+vel_component+':')

    all_magns = []; avg_magns = []; masks = []; vels = []
    for scan_set_key in scan_set_type:

        scan_set = scan_set_type[scan_set_key]
        scans = all_scans[scan_set_key]
        
        # load kspace images
        print('> Using scan '+scan_set+' with shape: ',scans.shape)

        if scans.ndim == 5: 
            # scan sets with multiple repetitions (increase SNR of steady flows)
            scans = np.transpose(scans,(0,1,2,3,4))[:,:,0,:,scan_rep]
            print('- scan_rep: ',scan_rep)
        if scans.ndim == 4: 
            # scan sets with no repetitions
            scans = np.transpose(scans,(0,3,2,1))[:,:,0,:]

        # compute complex image (magnitude and phase) from kspace images
        magns  = []; phases = []
        for scan_no in range(scans.shape[-1]):
            if   data_type == 'kspace':
                # get complex image from k-space data
                sig = sfft.fftshift(sfft.fft2(sfft.ifftshift(scans[:,:,scan_no])))
            elif data_type == 'cimage':
                sig = scans[:,:,scan_no]
            else:
                print('> Unknown data_type: use either cimage or kspace')
            # get magnitude and phase images 
            magns.append(np.abs(sig))
            phases.append(np.angle(sig))

        # SNR estimation
        SNR = (np.average(magns[0])/np.std(magns[0][:, 0])\
              +np.average(magns[0])/np.std(magns[0][:, 1])\
              +np.average(magns[0])/np.std(magns[0][:,-1])\
              +np.average(magns[0])/np.std(magns[0][:,-2]))/4
        print('> SNR estimation:',SNR)

        var_phi = 1/SNR**2; phase_img_no = 4; avg_img_no = 5
        var_u   = (flow_range['flow']/np.pi)**2*phase_img_no*var_phi
        var_u  /= avg_img_no
        print('> Noise s.d. in velocity image: {}'.format(var_u**0.5))
        #print(np.sqrt((flow_range['flow']/np.pi)**2*phase_img_no*var_phi/avg_img_no))

        # compute mask using Chan-Vese segmentation of magnitude image
        avg_magn  = np.sum(magns,axis=0)
        avg_magn /= np.max(avg_magn)
        mask,_    = compute_mask(avg_magn,max_iter=10)
        #mask,_    = compute_mask(avg_magn)

        # compute velocity from phase differences
        vel = compute_velocity(phases,flow_range[scan_set_key],vel_component)

        # save flow and no flow images
        vels.append(vel)
        all_magns.append(np.sum(magns,axis=0))
        avg_magns.append(avg_magn)
        masks.append(mask)

    if with_plot:
        plt.figure(figsize=(12, 16), dpi=200)
        vel_image = vels[1]-vels[0]
        ax1 = plt.subplot(511)
        ax1.imshow(avg_magns[1].T,cmap=cmap)
        ax1.set_title('magnitude image (sum of 4 flow scans - normalised)')
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        ax2 = plt.subplot(512)
        ax2.imshow(masks[1].T,cmap=cmap)
        ax2.set_title('mask (generated using Chan-Vese segmentation)')
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

        ax3 = plt.subplot(513)
        ax3.imshow(vels[1].T,cmap=cmap)
        ax3.set_title('masked flow (uncorrected) velocity image')
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)

        ax4 = plt.subplot(514)
        ax4.imshow(vels[0].T,cmap=cmap)
        ax4.set_title('masked no-flow velocity image')
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)

        ax5 = plt.subplot(515)
        ax5.imshow(vel_image.T,cmap=cmap)
        ax5.set_title('masked (zero-flow corrected) velocity image')
        ax5.get_xaxis().set_visible(False)
        ax5.get_yaxis().set_visible(False)

        # plt.show()
        directory = 'IMtrx_to_images'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory+'/'+vel_component+'_'+str(scan_rep)+\
               '_flow'+str(flow_scan_id)+'no_flow'+str(no_flow_scan_id)+'.png')

    return vels,avg_magns,masks,all_magns,phases

def export_to_vtk(data_scalar1, data_scalar2, data_vector, filename, voxel_spacing):
    """
    Exports a 3D array to a VTK file.
    
    :param data_scalar1: 3D numpy array representing the first scalar field.
    :param data_scalar2: 3D numpy array representing the second scalar field.
    :param data_vector: A list of 3D numpy arrays representing the vector field.
    :param filename: Name of the VTK file to create.
    :param voxel_spacing: List or tuple containing the voxel spacing in x, y, and z directions.
    """
    
    # Get the size of the 2D data
    nx, ny = data_vector[0].shape[:2]
    nz = 1

    # Open the file
    try:
        with open(filename, 'w') as f:
            # Write VTK header and data format
            f.write('# vtk DataFile Version 3.0\n')
            f.write('VTK from Python\n')
            f.write('ASCII\n')

            # Write dataset structure
            f.write('DATASET STRUCTURED_POINTS\n')
            f.write(f'DIMENSIONS {nx} {ny} {nz}\n')
            f.write('ORIGIN 0 0 0\n')
            f.write(f'SPACING {voxel_spacing[0]} {voxel_spacing[1]} {voxel_spacing[2]}\n')
            f.write(f'POINT_DATA {nx * ny * nz}\n')

            # Write data header for the first scalar field
            f.write('SCALARS MAG_FC float\n')
            f.write('LOOKUP_TABLE default\n')
            # Write the data
            #for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f'{data_scalar1[i, j]} ')
                    #f.write(f'{data_scalar1[i, j, k]} ')
                f.write('\n')

            # Write data header for the second scalar field
            f.write('SCALARS MAG_FS float\n')
            f.write('LOOKUP_TABLE default\n')
            # Write the data
            for j in range(ny):
                for i in range(nx):
                    f.write(f'{data_scalar2[i, j]} ')
                f.write('\n')

            # Write vector data
            f.write('VECTORS U float\n')
            for j in range(ny):
                for i in range(nx):
                    # Output each component of the vector
                    f.write(f'{data_vector[0][i, j]} {data_vector[1][i, j]} {data_vector[2][i, j]} ')
                f.write('\n')

    except IOError:
        print('Cannot open the file for writing.')
# ------------------------------------------------------------------------------

# Load scans
scan_set_type = {'no-flow': dataset_folder+'kspace_'+str(no_flow_scan_id)+'.mat',\
                    'flow': dataset_folder+'kspace_'+str(   flow_scan_id)+'.mat'}
all_scans     = {'no-flow': scio.loadmat(scan_set_type['no-flow'])['IMtrx'],\
                    'flow': scio.loadmat(scan_set_type[   'flow'])['IMtrx']}

# POST-PROC STEP 1: k-space to physical space
# ------------------------------------------------------------------------------
vel_images     = {'u_r':[],'u_p':[],'u_s':[]}
mag_images     = {'u_r':[],'u_p':[],'u_s':[]}
avg_mag_images = {'u_r':[],'u_p':[],'u_s':[]}

for vel_component in ['u_r','u_p','u_s']:
    for scan_rep in range(scan_repetitions):
        vels,avg_magns,masks,magns,phases = get_flow_and_magnitude_images(\
                                          flow_scan_id    = flow_scan_id,\
                                          no_flow_scan_id = no_flow_scan_id,\
                                          scan_rep        = scan_rep,\
                                          flow_range      = flow_range,\
                                          vel_component   = vel_component,\
                                          with_plot       = plot_imtrx_to_imgs)
        vel_images[vel_component].append(vels[1]-vels[0]) # flow - no_flow
        mag_images[vel_component].append(magns)  # [no-flow,flow] for each comp
        avg_mag_images[vel_component].append(avg_magns) 

# get overall average magnitude (using all components)
magn_avg_all_comp  = np.sum(np.sum([np.sum(avg_mag_images[comp],axis=0)\
                     for comp in avg_mag_images],axis=0),axis=0)
magn_avg_all_comp /= np.max(magn_avg_all_comp)

# get averaged velocity components (increased SNR)
vel_avg = [np.sum(vel_images[comp],axis=0)/scan_repetitions\
           for comp in vel_images]

# get averaged (no-flow) mag images (increased SNR)
mag_noflow_all = {'u_r': [img[0] for img in mag_images['u_r']],\
                  'u_p': [img[0] for img in mag_images['u_p']],\
                  'u_s': [img[0] for img in mag_images['u_s']]}
mag_noflow_avg = [np.sum(mag_noflow_all[comp],axis=0)/scan_repetitions\
                  for comp in vel_images]

# get averaged (flow) mag images (increased SNR)
mag_flow_all = {'u_r': [img[1] for img in mag_images['u_r']],\
                'u_p': [img[1] for img in mag_images['u_p']],\
                'u_s': [img[1] for img in mag_images['u_s']]}
mag_flow_avg   = [np.sum(mag_flow_all[comp],axis=0)/scan_repetitions\
                  for comp in vel_images]

# compute TKE
vel_sigmas = {'u_r':[],'u_p':[],'u_s':[]}
k_flow   = np.pi/flow_range['flow']
k_noflow = np.pi/flow_range['no-flow']
for k,comp in enumerate(vel_images):
    m_noflow = mag_noflow_avg[k]
    m_flow   = mag_flow_avg[k]
    
    # Normalise magnitude images, scaled by maximum respective velocities
    # m_noflow_norm = mag_noflow_avg[k]/np.max(mag_noflow_avg[k])
    # m_flow_norm = mag_flow_avg[k]/np.max(mag_flow_avg[k])
    
    # Normalise magnitude images, both scaled by expected background magnitude
    # m_noflow_norm = mag_noflow_avg[k]/5.31
    # m_flow_norm = mag_flow_avg[k]/25.12
    
    # Normalise magnitude images, flow scaled by expected background magnitudes
    # m_noflow_norm = mag_noflow_avg[k]
    # m_flow_norm = mag_flow_avg[k]/4.73
    
    # Shift no flow magnitude image by addition: ensure same mean value in laminar region
    # m_noflow_norm = mag_noflow_avg[k]+19.81
    # m_flow_norm = mag_flow_avg[k]
    
    # Unchanged
    m_noflow_norm = mag_noflow_avg[k]
    m_flow_norm = mag_flow_avg[k]
    
    ratio = m_noflow_norm/m_flow_norm
    ratio[ratio < 1] = 1
    
    # Sigma calculation using normalised data
    sig = np.sqrt( 2*np.log(ratio) / (k_flow**2) )
    #sig = np.sqrt( 2*np.log((m_noflow_norm)/(m_flow_norm)) / (k_flow**2) )
    #sig = np.sqrt( 2*np.log((m_noflow_norm)/(m_flow_norm)) / ((k_flow**2)-(k_noflow**2)) )

    sig = np.nan_to_num(sig)
    vel_sigmas[comp] = sig

#tke = (vel_sigmas['u_r']**2+vel_sigmas['u_p']**2+vel_sigmas['u_s']**2)**0.5
tke = 0.5*( (vel_sigmas['u_r']**2) + (vel_sigmas['u_p']**2) + (vel_sigmas['u_s']**2) )

plt.figure(figsize=(12, 16), dpi=200)
ax1 = plt.subplot(411)
ax1.imshow(magn_avg_all_comp.T,cmap=cmap)
ax1.set_title('overall averaged magnitude (normalised)')
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2 = plt.subplot(412)
ax2.imshow(vel_avg[0].T,cmap=cmap)
ax2.set_title('u_r averaged')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax3 = plt.subplot(413)
ax3.imshow(vel_avg[1].T,cmap=cmap)
ax3.set_title('u_p averaged')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax4 = plt.subplot(414)
ax4.imshow(vel_avg[2].T,cmap=cmap)
ax4.set_title('u_s averaged')
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)

# Plot magnitude images
plt.figure(figsize=(12, 16), dpi=200)
ax1 = plt.subplot(611)
ax1.imshow(mag_noflow_avg[0].T,cmap=cmap)
ax1.set_title('Mag noflow: component 0')
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2 = plt.subplot(612)
ax2.imshow(mag_flow_avg[0].T,cmap=cmap)
ax2.set_title('Mag flow: component 0')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax3 = plt.subplot(613)
ax3.imshow(mag_noflow_avg[1].T,cmap=cmap)
ax3.set_title('Mag noflow: component 1')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ax4 = plt.subplot(614)
ax4.imshow(mag_flow_avg[1].T,cmap=cmap)
ax4.set_title('Mag flow: component 1')
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
ax5 = plt.subplot(615)
ax5.imshow(mag_noflow_avg[2].T,cmap=cmap)
ax5.set_title('Mag noflow: component 2')
ax5.get_xaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)
ax6 = plt.subplot(616)
ax6.imshow(mag_flow_avg[2].T,cmap=cmap)
ax6.set_title('Mag flow: component 2')
ax6.get_xaxis().set_visible(False)
ax6.get_yaxis().set_visible(False)

# Plot TKE
plt.figure(figsize=(3, 4), dpi=200)
plt.title('TKE')
im = plt.imshow(tke.T, cmap='magma')   #plt.imshow(tke.T,cmap='magma')
plt.colorbar(im, orientation='vertical')
plt.imsave('tke.png',arr=tke.T,cmap='magma')


# Debug TKE
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
im0 = axs[0].imshow(m_noflow_norm.T, cmap='gray')#, vmin=0, vmax=1.3)
axs[0].set_title('mag-fc')
fig.colorbar(im0, ax=axs[0], orientation='vertical')

im1 = axs[1].imshow(m_flow_norm.T, cmap='gray')#, vmin=0, vmax=1.3)
#im1 = axs[1].imshow(temp2.T, cmap='seismic', vmin=-0.5, vmax=0.5)
axs[1].set_title('mag-fs')
fig.colorbar(im1, ax=axs[1], orientation='vertical')

im2 = axs[2].imshow(ratio.T, cmap='seismic', vmin=0.5, vmax=1.5)
axs[2].set_title('mag-fc/mag-fs')
fig.colorbar(im2, ax=axs[2], orientation='vertical')

im3 = axs[3].imshow(tke.T, cmap='magma')
axs[3].set_title('TKE')
fig.colorbar(im3, ax=axs[3], orientation='vertical')
plt.show()

# Plot Mag-fs
plt.figure(figsize=(3, 4), dpi=200)
plt.title('Mag-fs VENC {}'.format(VENC))
im = plt.imshow(m_flow_norm.T, cmap='gray', vmin=1, vmax=7)
plt.colorbar(im, orientation='vertical')

###########################################################

directory = 'IMtrx_to_images'
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory+'/images_averaged.png')
np.savez(directory+'/images_averaged',\
     magn=magn_avg_all_comp.T,\
     u_r=vel_avg[0].T,u_p=vel_avg[1].T,u_s=vel_avg[2].T)
plt.show()

##############################################################
# EXPORT DATA TO VTK

# filename_out = 'MRV_data.vtk'

# export_to_vtk(data_scalar1 = m_noflow,\
#               data_scalar2 = m_flow,\
#               data_vector  = vel_avg,\
#               filename     = filename_out,\
#               voxel_spacing= voxel_spacing)

# get_flow_and_magnitude_images(\
#                                   flow_scan_id    = flow_scan_id,\
#                                   no_flow_scan_id = no_flow_scan_id,\
#                                   scan_rep        = scan_rep,\
#                                   flow_range      = flow_range,\
#                                   vel_component   = vel_component,\
#                                   with_plot       = plot_imtrx_to_imgs)

# POST-PROC STEP 2: make images AXISYMMETRIC to machine precision 
# ------------------------------------------------------------------------------
directory = 'myoko_make_axisymm'
scio.savemat(directory+'/magn_avg_all_comp.mat',\
                      {'magn_avg_all_comp':magn_avg_all_comp})
scio.savemat(directory+'/u_r_avg.mat',{'u_r_avg':vel_avg[0]})
scio.savemat(directory+'/u_p_avg.mat',{'u_p_avg':vel_avg[1]})
scio.savemat(directory+'/u_s_avg.mat',{'u_s_avg':vel_avg[2]})
scio.savemat(directory+'/mag_noflow.mat',{'mag_fc':m_noflow})
scio.savemat(directory+'/mag_flow.mat',{'mag_fs':m_flow})
scio.savemat(directory+'/ratio.mat',{'ratio':ratio})


# cwd = os.getcwd()
# os.chdir(cwd+'/'+directory)
# print('\n> Running MYokos MATLAB script..')
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.Driver(nargout=0)
# os.chdir(cwd)

# # load MYoko's corrected (affine transform) magnitude image
# magn_avg_all_comp_corr = \
#               scio.loadmat(directory+'/magn_avg_all_comp_corr.mat')['I_axi']
# # slightly enhance contrast for segmentation
# magn_avg_all_comp_corr = magn_avg_all_comp_corr**0.5

# # load MYoko's corrected (affine transform) velocity images
# u_r_avg_corr = scio.loadmat(directory+'/u_r_avg_corr.mat')['I_axi']
# u_p_avg_corr = scio.loadmat(directory+'/u_p_avg_corr.mat')['I_axi']
# u_s_avg_corr = scio.loadmat(directory+'/u_s_avg_corr.mat')['I_axi']

# # mirror-average (machine precision axisymmetry)
# magn_axi    = (magn_avg_all_comp_corr+magn_avg_all_comp_corr[:,::-1])/2
# mask,ls_axi = compute_mask(magn_axi)
# mask_axi    = mask+mask[:,::-1]
# mask_axi    = np.where(mask_axi != 0.,mask_axi/np.abs(mask_axi),mask_axi) 

# # mirror-average (machine precision axisymmetry)
# u_r_axi = (u_r_avg_corr+u_r_avg_corr[:,::-1])/2
# u_p_axi = (u_p_avg_corr-u_p_avg_corr[:,::-1])/2
# u_s_axi = (u_s_avg_corr-u_s_avg_corr[:,::-1])/2

# # plot mirror-averaged images
# ax1 = plt.subplot(411)
# ax1.imshow(magn_axi.T,cmap=cmap)
# ax2 = plt.subplot(412)
# ax2.imshow(mask_axi.T,cmap=cmap)
# ax3 = plt.subplot(413)
# ax3.imshow((mask_axi*u_r_axi).T,cmap=cmap)
# ax4 = plt.subplot(414)
# ax4.imshow((mask_axi*u_p_axi).T,cmap=cmap)
# plt.show()

# plt.plot((mask_axi*u_r_axi)[350,:])
# plt.plot((mask_axi*u_p_axi)[350,:])
# plt.show()

# #for i in range(50):
# import numpy.ma as ma
# mask_ = np.array(1-mask_axi,dtype=bool)
# u_p_axi_m = mx = ma.masked_array(u_p_axi, mask=mask_)
# print('VEL NOISE mean/std:', np.mean(u_p_axi_m[350:400,:]),\
#                              np.std(u_p_axi_m[350:400,:]))

# print('VEL NOISE mean/std throat:', np.mean(u_p_axi_m[85:115,:]),\
#                                     np.std(u_p_axi_m[85:115,:]))

# np.savez(directory+'/images_averaged',\
#      magn=magn_axi.T,mask=mask_axi.T,\
#      u_r=(mask_axi*u_r_axi).T,u_p=(mask_axi*u_p_axi).T,\
#      u_s=mask_axi.T)

# # create .csv file for Richard
# # x,y = np.indices(u_r_axi.shape)
# # data = np.array([x.ravel(),y.ravel(),mask_axi.ravel(),\
# #                  u_r_axi.ravel(),u_p_axi.ravel()])
# # np.savetxt("out.csv",data.T,delimiter=",",\
# #             header="x,y,mask,vel_axi,vel_rad")

# # POST-PROC STEP 3: mass flow check and ground-truth geometry check
# # ------------------------------------------------------------------------------
# # sanity check for mass flow
# # -----------------------------------------------------------
# radial_dist   = np.zeros_like(mask_axi)
# radial_pixels = mask_axi.shape[1]//2 # assumes even number

# for k in range(mask_axi.shape[0]):
#     pixel_dr         = fov[1]/(2*radial_pixels)
#     dist             = pixel_dr*np.arange(-radial_pixels,radial_pixels)
#     radial_dist[k,:] = np.abs(dist)

# # sanity check
# mask_axi_onesided = mask_axi[radial_pixels:]
# radius = np.sum(mask_axi_onesided,axis=1)/2
# mass_flow_integral_approx = mask_axi_onesided\
#                            *u_r_axi[radial_pixels:]\
#                            *radial_dist[radial_pixels:] 
# from scipy.ndimage import gaussian_filter1d
# plt.plot(np.sum(mass_flow_integral_approx,axis=1))
# plt.plot(gaussian_filter1d(np.sum(mass_flow_integral_approx,axis=1),5))
# plt.show()
# # -----------------------------------------------------------

# ## plot magn scan and 0th level set together
# plt.imshow(magn_axi.T,cmap=cmap,extent=[0, fov[0], 0, fov[1]])
# nz,ny = magn_axi.shape
# yy,zz = np.meshgrid(np.linspace(0,fov[1],ny),np.linspace(0,fov[0],nz))
# plt.contour(zz.T,yy.T,ls_axi.T,levels=[0.])
# plt.show()

# # gt mask
# # -----------------------------------------------------------

# # ground-truth mask for [21July_test3/, flow scan 11]
# mask_gt = np.zeros_like(mask_axi)
# mask_gt[:126,50:90] = 1 
# mask_gt[126:,10:130]= 1

# plt.imshow(mask_axi.T,cmap='Greys',alpha=0.5)
# plt.imshow(mask_gt.T,cmap='Greys',alpha=0.5)
# plt.show()

# # ground-truth level-set
# mask,ls = compute_mask(mask_gt)
# level_set_gt = -ls
# # -----------------------------------------------------------

# # POST-PROC STEP 4: save axisymmetric images for 3D projection 
# # ------------------------------------------------------------------------------
# directory = 'ready_to_revolve_'+dataset_folder+\
#             'flow_'+str(flow_scan_id)+'_no_flow_'+str(no_flow_scan_id)
# if not os.path.exists(directory):
#     os.makedirs(directory)
# np.save(directory+'/magn_axi.npy',magn_axi)
# np.save(directory+'/vel_axial_axi.npy',u_r_axi)
# np.save(directory+'/vel_radial_axi.npy',u_p_axi)
# np.save(directory+'/mask_axi.npy',mask_axi)
# np.save(directory+'/mask_gt.npy',mask_gt)
# np.save(directory+'/level_set_gt.npy',level_set_gt)

