__author__ = "dangilman"

# Import these from lenstronomy, they do not need to be coded in JAX
# since these are just quick setup routines
from lenstronomy.LensModel.Util.decouple_multi_plane_util import (
    coordinates_and_deflections,
    setup_lens_model,
    setup_grids,
    setup_raytracing_lensmodels,
)

# Interpolation functions are imported from jax.scipy instead of scipy
def decoupled_multiplane_class_setup(
    lens_model_free,
    x,
    y,
    alpha_x_foreground,
    alpha_y_foreground,
    alpha_beta_subx,
    alpha_beta_suby,
    z_split,
    coordinate_type="POINT",
    interp_points=None,
    x_image=None,
    y_image=None,
    method="linear",
    bounds_error=False,
    fill_value=None,
):
    """This funciton creates the keyword arguments for a LensModel instance that is the
    decoupled multi-plane approxiamtion for the specified lens model.

    :param lens_model_free: the lens model with parameters free to vary
    :param x: comoving coordinate at z_split
    :param y: comoving coordinate at z_split
    :param alpha_x_foreground: ray angles at z_split (not including lens_model_free
        contribution)
    :param alpha_y_foreground: ray angles at z_split (not including lens_model_free
        contribution)
    :param alpha_beta_subx: deflection field from halos at redshift > z_split given the
        initial guess for the keyword arguments in lens_model_free
    :param alpha_beta_suby: deflection field from halos at redshift > z_split given the
        initial guess for the keyword arguments in lens_model_free
    :param z_split: redshift at which the lens model is decoupled from the line of sight
    :param coordinate_type: specifies the type of interpolation to use. Options are
        POINT, GRID, or MULTIPLE_IMAGES. POINT specifies a single point at which to
        compute the interpolation GRID specifies the interpolation on a regular grid
        MULTIPLE_IMAGES does interpolation on an array using the NEAREST method.
    :param lens_model_free:
    :param x: transverse comoving distance in x direction of the light rays at the main
        deflector
    :param y: transverse comoving distance in y direction of the light rays at the main
        deflector
    :param alpha_x_foreground: deflection angles from halos at redshift z<=z_split
    :param alpha_y_foreground: deflection angles from halos at redshift z<=z_split
    :param alpha_beta_subx: deflection angles from halos at redshift z > z_lens
    :param alpha_beta_suby: deflection angles from halos at redshift z > z_lens
    :param z_split: the redshift where foreground and background halos are split
    :param coordinate_type: a string specifying the type of coordinate of x. Options are
        GRID, POINT, and MULTIPLE_IMAGES
    :param interp_points: optional keyword argument passed to GRID method that specifies
        the interpolation grid
    :param x_image: optional keyword argument passed to multiple images argument that
        specifies the image coordinates
    :param y_image: optional keyword argument passed to multiple images argument that
        specifies the image coordinates
    :param method: the interpolation method used by RegularGridInterpolator if
        coordinate_type=='GRID'
    :param bounds_error: passed to RegularGridInterpolater, see documentation there
    :param fill_value: passed to RegularGridInterpolator, see documentation there
    :return: keyword arguments that can be passed into a LensModel class to create a
        decoupled-multiplane lens model
    """
    if bounds_error:
        raise ValueError("JAX does not support bounds_error")
    
    if coordinate_type == "GRID":
        from jax.scipy.interpolate import RegularGridInterpolator

        npix = int(len(x) ** 0.5)
        interp_xD = RegularGridInterpolator(
            interp_points,
            x.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_yD = RegularGridInterpolator(
            interp_points,
            y.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_foreground_alpha_x = RegularGridInterpolator(
            interp_points,
            alpha_x_foreground.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_foreground_alpha_y = RegularGridInterpolator(
            interp_points,
            alpha_y_foreground.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_deltabeta_x = RegularGridInterpolator(
            interp_points,
            alpha_beta_subx.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
        interp_deltabeta_y = RegularGridInterpolator(
            interp_points,
            alpha_beta_suby.reshape(npix, npix).T,
            bounds_error=bounds_error,
            fill_value=fill_value,
            method=method,
        )
    elif coordinate_type == "POINT":
        interp_xD = lambda *args: x
        interp_yD = lambda *args: y
        interp_foreground_alpha_x = lambda *args: alpha_x_foreground
        interp_foreground_alpha_y = lambda *args: alpha_y_foreground
        interp_deltabeta_x = lambda *args: alpha_beta_subx
        interp_deltabeta_y = lambda *args: alpha_beta_suby

    elif coordinate_type == "MULTIPLE_IMAGES":
        raise ValueError("MULTIPLE_IMAGES not supported in jaxtronomy")
        # from scipy.interpolate import NearestNDInterpolator

        # interp_points = list(zip(x_image, y_image))
        # interp_xD = NearestNDInterpolator(interp_points, x)
        # interp_yD = NearestNDInterpolator(interp_points, y)
        # interp_foreground_alpha_x = NearestNDInterpolator(
        #     interp_points, alpha_x_foreground
        # )
        # interp_foreground_alpha_y = NearestNDInterpolator(
        #     interp_points, alpha_y_foreground
        # )
        # interp_deltabeta_x = NearestNDInterpolator(interp_points, alpha_beta_subx)
        # interp_deltabeta_y = NearestNDInterpolator(interp_points, alpha_beta_suby)

    else:
        raise Exception(
            "coordinate type must be either GRID, POINT, MULTIPLE_IMAGES, or MULTIPLE_IMAGES_GRID"
        )

    kwargs_decoupled_lens_model = {
        "x0_interp": interp_xD,
        "y0_interp": interp_yD,
        "alpha_x_interp_foreground": interp_foreground_alpha_x,
        "alpha_y_interp_foreground": interp_foreground_alpha_y,
        "alpha_x_interp_background": interp_deltabeta_x,
        "alpha_y_interp_background": interp_deltabeta_y,
        "z_split": z_split,
    }
    kwargs_lens_model = {
        "lens_model_list": lens_model_free.lens_model_list,
        "profile_kwargs_list": lens_model_free.profile_kwargs_list,
        "lens_redshift_list": lens_model_free.redshift_list,
        "cosmo": lens_model_free.cosmo,
        "multi_plane": True,
        "z_source": lens_model_free.z_source,
        "decouple_multi_plane": True,
        "kwargs_multiplane_model": kwargs_decoupled_lens_model,
    }
    return kwargs_lens_model