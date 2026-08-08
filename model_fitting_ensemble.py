import matplotlib.pyplot as plt
import numpy as np
import os
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import galsim
import pickle
from matplotlib import patches
import time

from kl_pipe.parameters import ImagePars
from kl_pipe.source import SourceModel
from kl_pipe.velocity import CenteredVelocityModel
from kl_pipe.intensity import InclinedExponentialModel
from kl_pipe.lines import EmissionLine, LINE_LAMBDAS
from kl_pipe.observation import build_image_obs, build_fiber_obs
from kl_pipe.observation import group_fiber_obs
from kl_pipe.noise import add_intensity_noise, add_fiberspec_noise
from kl_pipe.render import RenderConfig
from kl_pipe.fiber import build_fiber_pars_for_line
from kl_pipe.priors import PriorDict, Uniform, Gaussian, TruncatedNormal
from kl_pipe.sampling import InferenceTask, NumpyroSamplerConfig, build_sampler
from kl_pipe.sampling.diagnostics import plot_corner, plot_trace

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--seed",type=int, default=0, help="Random seed used to generate observational noise")


folder_name = "/Users/laurenhwillett/Desktop/DESI_KL/fiber_likelihood_things/mcmc_results/numpyro_sm_17_param_realism_3fiber_1600_800_4_speedup"
#check whether a folder already exists!!
if os.path.isdir(folder_name) is False:
    os.mkdir(folder_name)
else:
   raise TypeError('Folder already exists!!')

g_fwhm=1.2
g_SNR = 300
g_noise_seed = 11

spec_fwhm = 1.2
spec_SNRs=[50.0, 50.0, 70.0]
spec_noise_seeds = [1, 2, 3]

cosi = 0.4

# Joint source: velocity + one broadband band.
g_model = InclinedExponentialModel()
halpha_model = InclinedExponentialModel()
halpha_cont = InclinedExponentialModel()
joint_source = SourceModel(
    velocity_model=CenteredVelocityModel(),
    broadband_models={'g': g_model},#, 'r': r_model, 'z': z_model},
    emission_lines={'Halpha': EmissionLine(intensity=halpha_model, continuum=halpha_cont)},
)

z=0.058
true_pars = {
    'cosi': cosi,
    'theta_int': 0.1*np.pi,
    'g1': 0.02,
    'g2': -0.01,
    'vel.v0': 0.0,
    'vel.vcirc': 200.0, #150.0,
    'vel.rscale': 3.12,
    'flux': 150.0,
    'rscale': 3.12, #3.0,
    'h_over_r': 0.1,
    'x0': 0.0,
    'y0': 0.0,
    'Halpha.flux': 120.0, #100.0
    'Halpha.rscale': 3.12, #3.12,
    'Halpha.h_over_r': 0.1,
    'Halpha.x0': 0.0,
    'Halpha.y0': 0.0,
    'Halpha.dispersion': 50.0,
    'Halpha.cont.flux_per_nm': 80.0, #8.0, #25.0, #flux per nm actually
    'Halpha.cont.rscale': 3.12, #3.12,
    'Halpha.cont.h_over_r': 0.1,
    'Halpha.cont.x0': 0.0,
    'Halpha.cont.y0': 0.0,
    'z': z,
}

pixel_scale=0.262 #for DESI legacy survey
imagepars = ImagePars(shape=(251, 251), pixel_scale=pixel_scale, indexing='ij')

phot_psf = galsim.Gaussian(g_fwhm)
rc = RenderConfig(oversample=1, spectral_oversample=1)

#broadband data (PSF-convolved)
g_obs_clean = build_image_obs(imagepars, psf=phot_psf, render_config=rc,
                             int_model=g_model, broadband_key='g')
g_true = np.asarray(joint_source.render_broadband(true_pars, g_obs_clean, 'g'))

g_noisy, g_var = add_intensity_noise(g_true, target_snr=g_SNR, seed=g_noise_seed, include_poisson=True)
g_obs = build_image_obs(imagepars, psf=phot_psf, render_config=rc, int_model=g_model,
                       broadband_key='g', data=jnp.asarray(g_noisy), variance=g_var)
plt.imshow(g_noisy)#, vmin=0, vmax=0.3)
plt.title('SNR = '+ str(g_SNR))
plt.colorbar()
plt.savefig(folder_name+'/g_image.png')

fiber_blur = 3.4 # pixels
spec_psf = galsim.Gaussian(spec_fwhm)

fiber_rad = 0.75 # arcsec
fiber_offset = 15.75105607509613*0.4 #arcsec (distance between fibers and center)
SCR_DIR = '/Users/laurenhwillett/Desktop/DESI_KL/kl_roman_pipe_2/kl_roman_pipe/' #to get data folder
chn = 'r' #this is the channel for the spectrum. needs to encompass redshifted H-alpha line
_bp = SCR_DIR+"data/Bandpass/DESI/%s.dat"%(chn)

offsets = [(fiber_offset*np.cos(0),         fiber_offset*np.sin(0)),
            (fiber_offset*np.cos(np.pi),   fiber_offset*np.sin(np.pi)),
            (0,0)]
theta_int_angle = true_pars['theta_int']
R = np.array([[np.cos(theta_int_angle), -np.sin(theta_int_angle)],
              [np.sin(theta_int_angle), np.cos(theta_int_angle)]])
rotated_offsets = []
for offset_og in offsets:
    rotated_offsets.append(tuple(np.matmul(R,offset_og)))

ax = plt.figure().add_subplot()
ax.imshow(g_true)
pixscale = pixel_scale

for j in range(len(rotated_offsets)):
    x, y = rotated_offsets[j]
    x = x/pixscale + imagepars.shape[0]/2
    y = y/pixscale + imagepars.shape[0]/2
    offsetpos = (x,y)
    fiber = patches.Circle(offsetpos, 0.75/pixscale, color='red', fill=False)
    ax.add_patch(fiber)
    ax.text(x-6, y-6, str(j+1), fontsize=16, color='red')
plt.title('Fiber Config')
plt.savefig(folder_name+'/fiber_config.png')

z=true_pars['z']
_IMAGE_PARS_spec = ImagePars(shape=(100, 100), pixel_scale=0.22, indexing='ij') # pixel_scale= arcsec/pixel
fiberpars_instances = []
fiberobs_instances = []
rendered_spectra = []
for i in range(0, len(rotated_offsets)):
    fiberpars_instance = build_fiber_pars_for_line(
        lambda_rest = LINE_LAMBDAS['Halpha'], redshift=z, delta_lambda=0.1, image_pars=_IMAGE_PARS_spec, fiber_radius=fiber_rad, fiber_blur = fiber_blur, fiber_dx=rotated_offsets[i][0], fiber_dy=rotated_offsets[i][1], bandpass_path = _bp)
    fiberobs_instance = build_fiber_obs(fiberpars_instance, z, psf=spec_psf, render_config=rc)
    rendered_spec = joint_source.render_fiber(pars = true_pars, obs=fiberobs_instance)
    fiberpars_instances.append(fiberpars_instance)
    fiberobs_instances.append(fiberobs_instance)
    rendered_spectra.append(rendered_spec)
cubepars_instance = fiberobs_instance.cube_pars
for fiberobs_instance in fiberobs_instances:
    plt.imshow(fiberobs_instance.ATMPSF_conv_fiber_mask)
    plt.show()

fiberspectra_noisy = []
fiberspectra_var = []
for i in range(0, len(rendered_spectra)):
    rendered_spec = rendered_spectra[i]
    spec_SNR = spec_SNRs[i]
    spec_noise_seed = spec_noise_seeds[i]
    fiberspec_noisy, fiberspec_var = add_fiberspec_noise(rendered_spec, spec_SNR, include_poisson=True, seed=spec_noise_seed)
    fiberspectra_noisy.append(fiberspec_noisy)
    fiberspectra_var.append(fiberspec_var)
    plt.errorbar(cubepars_instance.lambda_grid, fiberspec_noisy, yerr = np.sqrt(fiberspec_var), label='fiber '+ str(i+1)+ '; SNR =' + str(SNR))
    plt.xlabel('wavelength (nm)')
    plt.ylabel('flux')
    plt.xlim(685, 702.5)
    plt.legend()
plt.savefig(folder_name+'/fiber_spectra.png')
fiberobs_spectra = []
for i in range(0, len(fiberspectra_noisy)):
    data = fiberspectra_noisy[i]
    var = fiberspectra_var[i]
    fiberpars = fiberpars_instances[i]
    obs = build_fiber_obs(fiberpars, z, data=data, variance=var, psf=spec_psf, render_config=rc)
    fiberobs_spectra.append(obs)
#group them up
fiberobs_spectra_groups = group_fiber_obs({'fiber0': fiberobs_spectra[0], 'fiber1': fiberobs_spectra[1], 'fiber2': fiberobs_spectra[2]})

#build InferenceTask from obs
priors = PriorDict({
    'cosi': TruncatedNormal(cosi, 0.15, 0.05, 0.99),
    'theta_int': TruncatedNormal(0.1*np.pi, 0.3, 0.01, np.pi / 2),
    'g1': Uniform(-0.25, 0.25),
    'g2': Uniform(-0.25, 0.25),
    'vel.v0': Gaussian(0.0, 10.0),
    'vel.vcirc': TruncatedNormal(200.0, 80.0, 50.0, 550.0),   # TF prior
    'vel.rscale': Uniform(0.1, 5.0),
    'flux': TruncatedNormal(150.0, 20.0, 30.0, 250.0),
    'rscale': Uniform(0.1, 5.0),
    'h_over_r': true_pars['h_over_r'],                                     # fixed (low observable sensitivity)
    'x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.flux': TruncatedNormal(120.0, 20.0, 30.0, 250.0),
    'Halpha.rscale': Uniform(0.1, 5.0), #2.0,
    'Halpha.h_over_r': true_pars['Halpha.h_over_r'],                                
    'Halpha.x0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.y0': TruncatedNormal(0.0, 0.1, -0.5, 0.5),
    'Halpha.dispersion': TruncatedNormal(50.0, 20.0, 5.0, 150.0),
    'Halpha.cont.flux_per_nm': TruncatedNormal(80.0, 15.0, 0.01, 200.0),
    'Halpha.cont.rscale': true_pars['Halpha.cont.rscale'],
    'Halpha.cont.h_over_r': true_pars['Halpha.cont.h_over_r'],
    'Halpha.cont.x0': true_pars['Halpha.cont.x0'],
    'Halpha.cont.y0': true_pars['Halpha.cont.y0'],
    'z': true_pars['z'],                                                  
})
task = InferenceTask.from_obs(
    joint_source, priors,
    image_obs={'g': g_obs},
    fiber_obs = fiberobs_spectra_groups[0], spectral_oversample=1)
theta_true = []
for name in task.sampled_names:
    theta_true.append(true_pars[name])
theta_true = jnp.array(theta_true)
print(theta_true)

start = time.time()
config_nuts = NumpyroSamplerConfig(
    n_samples=1600,
    n_warmup=800,
    n_chains=4,
    dense_mass=True,
    reparam_strategy='prior',
    max_tree_depth = 8,
    target_accept_prob=0.8,
    chain_method='parallel',
    seed=42)
sampler = build_sampler('numpyro', task, config_nuts)
result = sampler.run()
end = time.time()
print('runtime', end-start)

with open(folder_name+"/results.pkl", "wb") as file:
    pickle.dump(result, file)

true_values = true_pars

output_string = ""
summary = result.get_summary()

output_string += ((str("=" * 100) + '\n'))
output_string += ("SAMPLING SUMMARY" + '\n')
output_string += ((str("=" * 100) + '\n'))
output_string += (str(f"Backend: {result.metadata.get('backend', 'unknown')}") + '\n')
output_string += (str(f"N samples: {result.n_samples}") + '\n')
output_string += (str(f"N params: {result.n_params}")+ '\n')
output_string += (str(f"Runtime (s): {end-start}")+ '\n')

if result.acceptance_fraction is not None:
    output_string += (str(f"Acceptance: {result.acceptance_fraction:.1%}") + '\n')

if result.evidence is not None:
    output_string += (str(f"Log evidence: {result.evidence:.2f}")+ '\n')
    if result.evidence_error is not None:
        output_string += (str(f"Evidence error: {result.evidence_error:.2f}")+ '\n')

output_string += (str("=" * 100) + '\n')
output_string += (str(f"{'Parameter':<15} {'Median':>12} {'Std':>12} {'[16%, 84%]':>20}"))#+ '\n')
if true_values:
    output_string += (str(f" {'True':>12} {'Error':>10}")+ '\n')
else:
    output_string += ('\n')
output_string += (str("=" * 100) + '\n')

for name in result.param_names:
    s = summary[name]
    median = s['quantiles'][0.5]
    q16 = s['quantiles'][0.16]
    q84 = s['quantiles'][0.84]
    std = s['std']

    output_string += (str(f"{name:<15} {median:>12.4f} {std:>12.4f} [{q16:>8.4f}, {q84:>8.4f}]"))# + '\n')

    if true_values and name in true_values:
        true_val = true_values[name]
        if abs(true_val) > 1e-10:
            rel_error = abs(median - true_val) / abs(true_val)
            output_string += (str(f" {true_val:>12.4f} {rel_error:>9.1%}")+ '\n')
        else:
            output_string += (str(f" {true_val:>12.4f} {'N/A':>10}")+ '\n')
    else:
        output_string += ('\n')

if result.fixed_params:
    output_string += (str("=" * 100) + '\n')
    output_string += (str("Fixed parameters:")+ '\n')
    for name, val in result.fixed_params.items():
        output_string += (str(f"  {name}: {val}")+ '\n')

output_string += ((str("=" * 100) + '\n'))
print(output_string)
with open(folder_name+"/summary.txt", "w") as f:
    f.write(output_string)

# Corner plot
fig = plot_corner(result, sampler_info={'name': 'numpyro'}, true_values=true_pars)
plt.savefig(folder_name+"/corner.png")

# Trace plots (useful for convergence diagnosis)
fig = plot_trace(result)
plt.savefig(folder_name+"/trace.png")

with open(folder_name+"/configs.txt", "w") as f:
    print(joint_source, file=f)
    imagepars_phot_string = str([str(imagepars.shape),str(imagepars.pixel_scale), str(imagepars.indexing)])
    imagepars_spec_string = str([str(_IMAGE_PARS_spec.shape),str(_IMAGE_PARS_spec.pixel_scale), str(_IMAGE_PARS_spec.indexing)])
    print('imagepars_phot:', imagepars_phot_string, file=f)
    print('imagepars_spec:', imagepars_spec_string, file=f)

    fiber_dict = {'fiber_blur': fiber_blur,
                'fiber_rad': fiber_rad,
                'fiber_offset': fiber_offset,
                'fiber_positions': rotated_offsets,
                'channel': chn}
    print('fiber configs:', fiber_dict, file=f)
    print('phot_psf:', phot_psf, 'FWHM:', g_fwhm, file=f)
    print('spec_psf:', spec_psf, 'FWHM:', spec_fwhm, file=f)
    print('render config:', rc, file=f)
    print('true_pars:', true_pars, file=f)
    print('priors:', priors, file=f)

    nuts_config_str = str([config_nuts.n_samples, config_nuts.n_warmup, config_nuts.n_chains, config_nuts.dense_mass, config_nuts.max_tree_depth, config_nuts.target_accept_prob])
    print('nuts config:', nuts_config_str, file=f)