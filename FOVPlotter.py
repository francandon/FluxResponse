import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits
from astropy.wcs import WCS

class FOVPlotter:
    def __init__(self, fits_path, num_bins=200, cmap='inferno', log_scale=False, fov_size=0.2):
        self.fits_path = fits_path
        self.num_bins = num_bins
        self.cmap = cmap
        self.log_scale = log_scale
        self.fov_size = fov_size
        self.circles = []

        # Verify that the FITS file exists
        if not os.path.exists(self.fits_path):
            raise FileNotFoundError(f"FITS file not found: {self.fits_path}")

        # Load data and create the initial plot
        self.load_fits_data()
        self.create_plot()

    def __str__(self):
        return f"FOVPlotter({self.fits_path})"

    def load_fits_data(self):
        """Load data from the FITS file and store necessary attributes."""
        with fits.open(self.fits_path) as fits_file:
            events_data = fits_file['EVENTS'].data
            header = fits_file['EVENTS'].header

            # Extract WCS parameters for 'X' and 'Y' columns
            crpix1 = header['TCRPX14']
            crval1 = header['TCRVL14']
            cdelt1 = header['TCDLT14']
            ctype1 = header['TCTYP14']

            crpix2 = header['TCRPX15']
            crval2 = header['TCRVL15']
            cdelt2 = header['TCDLT15']
            ctype2 = header['TCTYP15']

            # Manually create WCS object
            wcs_obj = WCS(naxis=2)
            wcs_obj.wcs.crpix = [crpix1, crpix2]
            wcs_obj.wcs.crval = [crval1, crval2]
            wcs_obj.wcs.cdelt = [cdelt1, cdelt2]
            wcs_obj.wcs.ctype = [ctype1, ctype2]
            wcs_obj.wcs.cunit = ['deg', 'deg']

            # Get 'X' and 'Y' pixel coordinates
            x = events_data['X']
            y = events_data['Y']

            # Transform pixel coordinates to RA and Dec using WCS
            sky_coords = wcs_obj.pixel_to_world(x, y)
            ra_wcs = sky_coords.ra.deg
            dec_wcs = sky_coords.dec.deg

        # Store data in instance variables
        self.ra_wcs = ra_wcs
        self.dec_wcs = dec_wcs
        self.header = header
        self.wcs = wcs_obj

    def create_plot(self):
        """Create the initial FOV plot."""
        # Handle zero counts for log scale
        if self.log_scale:
            norm = LogNorm()
        else:
            norm = None

        # Extract pointing coordinates
        ra_center_fov = self.header['RA_PNT']
        dec_center_fov = self.header['DEC_PNT']

        # Define half the FOV in degrees
        half_fov = self.fov_size / 2.0  # degrees

        # Calculate RA and Dec ranges
        ra_min = ra_center_fov - 5 * half_fov
        ra_max = ra_center_fov + 6 * half_fov
        dec_min = dec_center_fov - 1.8 * half_fov
        dec_max = dec_center_fov + 1.8 * half_fov

        # Create figure and WCSAxes
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection=self.wcs)

        # Plotting using hist2d
        self.h = self.ax.hist2d(
            self.ra_wcs, self.dec_wcs,
            bins=self.num_bins,
            cmap=self.cmap,
            norm=norm,
            zorder=1  # Background
        )

        # Apply zoom to the calculated ranges
        self.ax.set_xlim(ra_max, ra_min)  # Reverse RA axis if necessary
        self.ax.set_ylim(dec_min, dec_max)

        # Show grid lines
        self.ax.grid(color='white', ls='dotted')

        # Add colorbar
        cbar = plt.colorbar(self.h[3], ax=self.ax, label='Counts')

        # Set axis labels
        self.ax.set_xlabel('Right Ascension (J2000)')
        self.ax.set_ylabel('Declination (J2000)')

    def add_circle(self, ra_center, dec_center, radius, color='black', lw=2, linestyle='-', label=None, num_points=360):
        """
        Overlay a circle on the FOV plot.

        Parameters:
        - ra_center (float): Right Ascension of the circle center in degrees.
        - dec_center (float): Declination of the circle center in degrees.
        - radius (float): Radius of the circle in degrees.
        - color (str): Color of the circle line. Default is 'black'.
        - lw (float): Line width of the circle. Default is 2.
        - linestyle (str): Style of the circle line (e.g., '-', '--'). Default is '-'.
        - label (str): Label for the circle (for legend). Default is None.
        - num_points (int): Number of points to define the circle perimeter. Default is 360.
        """
        # Generate angles from 0 to 2pi radians
        theta = np.linspace(0, 2 * np.pi, num_points)

        # Calculate RA and Dec offsets
        # Adjust RA by cos(dec_center) to account for spherical geometry
        delta_ra = (radius / np.cos(np.deg2rad(dec_center))) * np.cos(theta)
        delta_dec = radius * np.sin(theta)

        # Calculate the RA and Dec coordinates of the circle's perimeter
        ra_circle = ra_center + delta_ra
        dec_circle = dec_center + delta_dec

        # Handle RA wrap-around (0 to 360 degrees)
        ra_circle = np.mod(ra_circle, 360)

        # Plot the circle
        self.ax.plot(
            ra_circle, dec_circle,
            color=color, lw=lw, linestyle=linestyle, label=label, zorder=5
        )

        # Update legend if label is provided
        if label:
            self.ax.legend(loc='upper right')

    def add_annuli(self, ra_center, dec_center, radii_array, units='degrees', color='black', lw=2, linestyle='-', label=None, num_points=360):
        """
        Overlay multiple annuli (circles) on the FOV plot.

        Parameters:
        - ra_center (float): Right Ascension of the annuli center in degrees.
        - dec_center (float): Declination of the annuli center in degrees.
        - radii_array (list or array): List of radii.
        - units (str): Units of radii ('degrees', 'arcminutes', or 'arcseconds'). Default is 'degrees'.
        - color (str): Color of the circles. Default is 'black'.
        - lw (float): Line width of the circles. Default is 2.
        - linestyle (str): Style of the circle lines (e.g., '-', '--'). Default is '-'.
        - label (str): Label for the circles (for legend). Default is None.
        - num_points (int): Number of points to define the circle perimeter. Default is 360.
        """
        # Convert radii to degrees if necessary
        if units == 'arcseconds':
            radii_in_degrees = [radius / 3600.0 for radius in radii_array]
        elif units == 'arcminutes':
            radii_in_degrees = [radius / 60.0 for radius in radii_array]
        else:
            radii_in_degrees = radii_array  # Assume degrees

        for radius in radii_in_degrees:
            self.add_circle(
                ra_center=ra_center,
                dec_center=dec_center,
                radius=radius,
                color=color,
                lw=lw,
                linestyle=linestyle,
                label=label,
                num_points=num_points
            )
            # After adding the first circle, set label to None to avoid duplicate entries in the legend
            label = None

    def show(self):
        """Display the plot."""
        plt.show()

    def save(self, output_path, dpi=300):
        """
        Save the plot to a file.

        Parameters:
        - output_path (str): Path to the output file.
        - dpi (int): Dots per inch (resolution). Default is 300.
        """
        self.fig.savefig(output_path, dpi=dpi)