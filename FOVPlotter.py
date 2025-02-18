import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

def get_exposure_time(fits_path):
    exposure_time = []
    for filename in os.listdir(fits_path):
        if filename.endswith("A01_cl.evt"):
            path = os.path.join(fits_path, filename)
            try:
                with fits.open(path) as fits_file:
                    header = fits_file['EVENTS'].header
                    exposure_time.append(header['EXPOSURE'])
            except KeyError:
                logging.error(f"EXPOSURE keyword not found in {path}")
            except Exception as e:
                logging.error(f"Error reading {path}: {e}")
    return exposure_time

class FOVPlotter:
    def __init__(self, fits_path, num_bins=200, cmap='inferno', log_scale=False, fov_size=0.2, legend_fontsize=12):
        self.fits_path = fits_path
        self.num_bins = num_bins
        self.cmap = cmap
        self.log_scale = log_scale
        self.fov_size = fov_size
        self.circles = []
        self.legend_fontsize = legend_fontsize

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
        self.x = x             # Store x as an instance variable
        self.y = y             # Store y as an instance variable
        self.ra_wcs = ra_wcs
        self.dec_wcs = dec_wcs
        self.header = header
        self.wcs = wcs_obj

    def create_plot(self):
        """Create the initial FOV plot."""
        from astropy.coordinates import SkyCoord

        # Handle zero counts for log scale
        if self.log_scale:
            norm = LogNorm()
        else:
            norm = None

        # Create figure and WCSAxes with the WCS projection
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection=self.wcs)

        # Plotting using hist2d in pixel coordinates
        # Plotting using hist2d in pixel coordinates
        self.h = self.ax.hist2d(
            self.x, self.y,  # Use pixel coordinates
            bins=self.num_bins,
            cmap=self.cmap,
            norm=norm,
            zorder=1,
            transform=self.ax.get_transform('pixel'),  # Specify transform
            edgecolors='none',     # Remove edges around bins
            linewidths=0,           # Set line widths to zero
            rasterized=True        # Rasterize the histogram
        )


        # **Adjust Axis Limits**

        # Extract pointing coordinates from header
        ra_center_fov = self.header['RA_PNT']
        dec_center_fov = self.header['DEC_PNT']
        print(f"RA center: {ra_center_fov}, Dec center: {dec_center_fov}")

        # Convert center coordinates to pixel coordinates
        sky_center = SkyCoord(ra_center_fov, dec_center_fov, unit='deg')
        x_center, y_center = self.wcs.world_to_pixel(sky_center)

        # Get pixel scale (degrees per pixel)
        pixel_scale_x = self.wcs.wcs.cdelt[0]  # degrees per pixel in x
        pixel_scale_y = self.wcs.wcs.cdelt[1]  # degrees per pixel in y

        # Compute delta in pixels corresponding to FOV/2 degrees
        delta_x = (self.fov_size / 1.1) / abs(pixel_scale_x)
        delta_y = (self.fov_size / 1.1) / abs(pixel_scale_y)

        # Set axis limits centered on the center coordinates
        x_min = x_center - delta_x
        x_max = x_center + delta_x
        y_min = y_center - delta_y +delta_y/10
        y_max = y_center + delta_y +delta_y/10
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        # Show grid lines (in RA/Dec)
        self.ax.grid(color='white', ls='dotted')

        # Add colorbar
        cbar = plt.colorbar(self.h[3], ax=self.ax, label='Counts')

        # Set axis labels
        self.ax.set_xlabel('Right Ascension (J2000)')
        self.ax.set_ylabel('Declination (J2000)')
        self.fig.subplots_adjust(left=0.15)  # Increase the left margin (adjust the value as needed)


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
        delta_ra = (radius / np.cos(np.deg2rad(dec_center))) * np.cos(theta)
        delta_dec = radius * np.sin(theta)

        # Calculate the RA and Dec coordinates of the circle's perimeter
        ra_circle = ra_center + delta_ra
        dec_circle = dec_center + delta_dec

        # Plot the circle in world coordinates
        self.ax.plot(
            ra_circle, dec_circle,
            color=color, lw=lw, linestyle=linestyle, label=label, zorder=5,
            transform=self.ax.get_transform('world')  # Specify the transform
        )

        # Update legend if label is provided
        if label:
            self.ax.legend(loc='upper right')

    def add_annuli(self, ra_center, dec_center, radii_array, units='degrees', lw=2, linestyle='-', label='Annulus Regions', num_points=360
                   ,color1= 'white', color2='red'):
        """
        Overlay multiple annuli (circles) on the FOV plot with a custom legend entry.
        """
        # Convert radii to degrees if necessary
        color_legend= "black"
        if units == 'arcseconds':
            radii_in_degrees = [radius / 3600.0 for radius in radii_array]
        elif units == 'arcminutes':
            radii_in_degrees = [radius / 60.0 for radius in radii_array]
        else:
            radii_in_degrees = radii_array  # Assume degrees

        # Generate colors for each annulus
        n_annuli = len(radii_in_degrees)
        cmap = mcolors.LinearSegmentedColormap.from_list('WhiteBlue', [color1, color2])
        colors = [cmap(i / (n_annuli - 1)) for i in range(n_annuli)]

        for idx, radius in enumerate(radii_in_degrees):
            color = colors[idx]
            self.add_circle(
                ra_center=ra_center,
                dec_center=dec_center,
                radius=radius,
                color=color,
                lw=lw,
                linestyle=linestyle,
                label=None,  # Prevent automatic legend entries
                num_points=num_points
            )

        # Create a custom legend handle
        annulus_legend_handle = Circle(
            (0, 0),
            radius=0.6,  # Radius is arbitrary for legend symbol
            edgecolor=color_legend,  # Representative color
            facecolor='none',  # No fill
            lw=lw
        )

        # Define the custom legend handler
        def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            return Circle(
                (width / 2, height / 2),  # Center of the circle
                width / 4,                # Radius
                edgecolor=orig_handle.get_edgecolor(),
                facecolor=orig_handle.get_facecolor(),
                lw=orig_handle.get_lw()
            )

        handler_dict = {annulus_legend_handle: HandlerPatch(patch_func=make_legend_circle)}

        # Add the custom legend handle
        self.ax.legend(
            handles=[annulus_legend_handle],
            labels=[label],
            handler_map=handler_dict,
            loc='upper right'
        )
    def add_polygon(self, coords, color='black', lw=2, linestyle='-', label=None):
        """
        Overlay a polygon on the FOV plot.

        Parameters:
        - coords (list): List of (RA, Dec) pairs in degrees, or flat list of alternating RA and Dec values.
        - color (str): Color of the polygon line. Default is 'black'.
        - lw (float): Line width of the polygon. Default is 2.
        - linestyle (str): Style of the polygon line (e.g., '-', '--'). Default is '-'.
        - label (str): Label for the polygon (for legend). Default is None.
        """
        # Determine if coords is a flat list or list of tuples
        if all(isinstance(c, tuple) and len(c) == 2 for c in coords):
            # It's a list of (RA, Dec) tuples
            ra_coords, dec_coords = zip(*coords)
        elif len(coords) % 2 == 0:
            # It's a flat list of alternating RA and Dec values
            ra_coords = coords[::2]
            dec_coords = coords[1::2]
        else:
            raise ValueError("Coordinates must be a list of (RA, Dec) tuples or a flat list of even length.")

        # Convert RA to [0, 360) if desired; note that this can cause "straight line" artifacts
        ra_coords = np.mod(ra_coords, 360)

        # Close the polygon by appending the first point to the end (if you want a fully closed polygon)
        ra_coords = list(ra_coords) + [ra_coords[0]]
        dec_coords = list(dec_coords) + [dec_coords[0]]

        # Plot the polygon, making sure to specify the world transform
        self.ax.plot(
            ra_coords, dec_coords,
            color=color,
            lw=lw,
            linestyle=linestyle,
            label=label,
            zorder=5,
            transform=self.ax.get_transform('world')  # <--- This fixes the coordinate system
        )

        # Update legend if label is provided
        if label:
            self.ax.legend(loc='upper right', fontsize=self.legend_fontsize)

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
        self.fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
