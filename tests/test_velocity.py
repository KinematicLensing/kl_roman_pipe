import unittest
import numpy as np
from astropy import units
import matplotlib.pyplot as plt

from kl_tools.velocity import VelocityModel, OffsetVelocityModel, VelocityMap
from kl_tools.coordinates import OrientedAngle
from kl_tools.utils import get_test_dir, make_dir, build_map_grid
from kl_tools.kross.rotation_curve_fitter import dist_to_major_axis


class TestVelocityModel(unittest.TestCase):
    def setUp(self) -> None:
        # Set up any necessary objects or variables for the tests
        self.model_name = 'centered'
        self.model_pars = {
            'g1': 0.1,
            'g2': -0.075,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 10.0,
            'vcirc': 200,
            'rscale': 5,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }

        self.model = VelocityModel(self.model_pars)

        return

    def test_model_pars(self) -> None:
        model = self.model
        pars = self.model_pars
        self.assertIsInstance(model, VelocityModel)
        self.assertEqual(model.name, self.model_name)
        self.assertEqual(model.pars['v0'], pars['v0'])
        self.assertEqual(model.pars['vcirc'], pars['vcirc'])
        self.assertEqual(model.pars['rscale'], pars['rscale'])
        self.assertEqual(model.pars['sini'], pars['sini'])

        return


class TestOffsetVelocityModel(unittest.TestCase):
    def setUp(self) -> None:
        # Set up any necessary objects or variables for the tests
        self.model_name = 'offset'
        self.model_pars = {
            'g1': 0.1,
            'g2': -0.075,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 10.0,
            'vcirc': 200,
            'rscale': 5,
            'x0': 10,
            'y0': 10,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }

        self.model = OffsetVelocityModel(self.model_pars)

        return

    def test_model_pars(self) -> None:
        model = self.model
        pars = self.model_pars
        self.assertIsInstance(model, VelocityModel)
        self.assertEqual(model.name, self.model_name)
        self.assertEqual(model.pars['v0'], pars['v0'])
        self.assertEqual(model.pars['vcirc'], pars['vcirc'])
        self.assertEqual(model.pars['rscale'], pars['rscale'])
        self.assertEqual(model.pars['sini'], pars['sini'])
        self.assertEqual(model.pars['x0'], pars['x0'])
        self.assertEqual(model.pars['y0'], pars['y0'])

        return


class TestVelocityMap(unittest.TestCase):

    def setUp(self) -> None:
        self.plot_dir = get_test_dir() / 'plots' / 'velocity'
        make_dir(self.plot_dir)

        # used for plots
        self.rmax = 30
        self.show = False

        # setup models
        self.model_name = 'centered'
        self.model_pars = {
            'g1': 0.1,
            'g2': -0.075,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 10.0,
            'vcirc': 200,
            'rscale': 5,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }

        self.vmap = VelocityMap(self.model_name, self.model_pars)

        offset_model_name = 'offset'
        offset_model_pars = {
            'g1': 0.1,
            'g2': -0.075,
            'theta_int': np.pi / 3,
            'sini': 0.8,
            'v0': 10.0,
            'vcirc': 200,
            'rscale': 5,
            'x0': 10,
            'y0': 10,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }
        self.offset_vmap = VelocityMap(offset_model_name, offset_model_pars)

        return

    def test_plot_vmap(self) -> None:
        outdir = self.plot_dir
        vmap = self.vmap
        rmax = self.rmax
        show = self.show
        center = True

        outfile = str(outdir / 'speedmap-transorms.png')
        vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=True, center=center, rmax=rmax
        )

        outfile = str(outdir / 'vmap-transorms.png')
        vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=False, center=center, rmax=rmax
        )

        # now for offset model
        offset_vmap = self.offset_vmap

        outfile = str(outdir / 'speedmap-transorms-offset.png')
        offset_vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=True, center=center, rmax=rmax
        )

        outfile = str(outdir / 'vmap-transorms-offset.png')
        offset_vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=False, center=center, rmax=rmax
        )

        return

    def test_plot_planes(self) -> None:
        outdir = self.plot_dir
        vmap = self.vmap
        rmax = self.rmax
        show = self.show
        center = True

        for plane in ['disk', 'gal', 'source', 'obs']:
            outfile = str(outdir / f'vmap-{plane}.png')
            vmap.plot(plane, outfile=outfile, show=show, center=center, rmax=rmax)

        # plot combined velocity field planes
        plot_kwargs = {'rmax': rmax}
        outfile = str(outdir / 'vmap-all.png')
        vmap.plot_all_planes(
            plot_kwargs=plot_kwargs,
            show=show,
            outfile=outfile,
            center=center,
        )

        # plot combined velocity field planes normalized by c
        outfile = str(outdir / 'vmap-all-normed.png')
        vmap.plot_all_planes(
            plot_kwargs=plot_kwargs,
            show=show,
            outfile=outfile,
            normalized=True,
            center=center,
        )

        # now for offset model

        offset_vmap = self.offset_vmap
        outfile = str(outdir / 'vmap-all-offset.png')
        offset_vmap.plot_all_planes(
            plot_kwargs=plot_kwargs,
            show=show,
            outfile=outfile,
            center=center,
        )

        return

    def test_plot_rectangle_vmap(self) -> None:
        outdir = self.plot_dir
        vmap = self.vmap
        show = self.show
        center = True

        Nx, Ny = 50, 30
        X, Y = build_map_grid(Nx, Ny)

        outfile = str(outdir / 'speedmap-transorms-rect.png')
        vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=True, center=center, X=X, Y=Y
        )

        outfile = str(outdir / 'vmap-transorms-rect.png')
        vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=False, center=center, X=X, Y=Y
        )

        # now for offset model
        offset_vmap = self.offset_vmap

        outfile = str(outdir / 'speedmap-transorms-offset-rect.png')
        offset_vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=True, center=center, X=X, Y=Y
        )

        outfile = str(outdir / 'vmap-transorms-offset-rect.png')
        offset_vmap.plot_map_transforms(
            outfile=outfile, show=show, speed=False, center=center, X=X, Y=Y
        )

        return

    def test_vmap_angle_orientation(self) -> None:
        # Test that the theta_int angle is correctly oriented (cartesian)

        model_pars = {
            'g1': 0.0,
            'g2': -0.0,
            'theta_int': 0,
            'sini': 0.8,
            'v0': 0.0,
            'vcirc': 200,
            'rscale': 5,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }

        vmap = VelocityMap('centered', model_pars)

        # we want the image to be 100 pixels wide and 50 pixels tall
        Nx, Ny = 100, 50
        X, Y = build_map_grid(Nx, Ny, indexing='xy')

        vmap_obs = vmap('obs', X, Y)

        plt.subplot(131)
        plt.imshow(X, origin='lower')
        plt.colorbar()
        plt.title('X')

        plt.subplot(132)
        plt.imshow(Y, origin='lower')
        plt.colorbar()
        plt.title('Y')

        plt.subplot(133)
        plt.imshow(vmap_obs, origin='lower')
        plt.colorbar()
        theta_int = np.rad2deg(model_pars['theta_int'])
        plt.title(f'Vmap Observed; theta_int = {theta_int:.2f} deg')

        plt.gcf().set_size_inches(16, 4)

        outfile = self.plot_dir / 'vmap-angle-orientation.png'
        plt.savefig(outfile)

        if self.show is True:
            plt.show()
        else:
            plt.close()

        return

    def test_rscale_pixel(self) -> None:
        # test if the rscale parameter is correctly applied, in pixels

        pa = OrientedAngle(30, unit=units.deg, orientation='cartesian')

        model_pars = {
            'g1': 0.0,
            'g2': 0.0,
            'theta_int': pa.cartesian.rad,
            'sini': 0.8,
            'v0': 0.0,
            'vcirc': 200,
            'rscale': 5,
            'r_unit': units.Unit('pixel'),
            'v_unit': units.km / units.s,
        }

        vmap = VelocityMap('centered', model_pars)

        Nx, Ny = 100, 100
        X, Y = build_map_grid(Nx, Ny, indexing='xy')

        vmap_obs = vmap('obs', X, Y)

        # plot the velocity field & PA
        plt.imshow(vmap_obs, origin='lower')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.title('Vmap Observed; rscale = 5 pixels; theta_int = 30 deg')
        plt.tight_layout()
        plt.gcf().set_size_inches(8, 6)

        outfile = self.plot_dir / 'vmap-rscale-pixel.png'
        plt.savefig(outfile)
        plt.close()

        outfile = self.plot_dir / 'vmap-rscale-pixel_rotation_curve.png'
        scale_radius = 10  # arcsec
        vmap.plot_rotation_curve(
            X, Y, out_file=outfile, show=True, scale_radius=scale_radius
        )

        return

    def test_rscale_arcsec(self) -> None:
        # test if the rscale parameter is correctly applied, in arcsec

        pa = OrientedAngle(30, unit=units.deg, orientation='cartesian')

        model_pars = {
            'g1': 0.0,
            'g2': 0.0,
            'theta_int': pa.cartesian.rad,
            'sini': 0.8,
            'v0': 0.0,
            'vcirc': 200,
            'rscale': 5,
            'r_unit': units.arcsec,
            'v_unit': units.km / units.s,
        }

        vmap = VelocityMap('centered', model_pars)

        Nx, Ny = 100, 100
        X, Y = build_map_grid(Nx, Ny, indexing='xy')

        pix_scale = 0.2 * units.arcsec
        vmap_obs = vmap('obs', X, Y, pix_scale=pix_scale)

        # plot the velocity field & PA
        plt.imshow(vmap_obs, origin='lower')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.title('Vmap Observed; rscale = 5 arcsec; theta_int = 30 deg')
        plt.tight_layout()
        plt.gcf().set_size_inches(8, 6)

        outfile = self.plot_dir / 'vmap-rscale-arcsec.png'
        plt.savefig(outfile)
        plt.close()

        outfile = self.plot_dir / 'vmap-rscale-arcsec_rotation_curve.png'
        scale_radius = 7  # arcsec
        vmap.plot_rotation_curve(
            X,
            Y,
            pix_scale=pix_scale,
            out_file=outfile,
            show=True,
            scale_radius=scale_radius,
        )

        return


if __name__ == '__main__':
    unittest.main()
