import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.dates
import numpy
import math

# Configure to always fore latex and siuintx
plt.rcParams['text.latex.preamble'] = [
    r'\usepackage{siunitx}',    # i need upright \micro symbols, but you need...
    r'\sisetup{detect-all}',    # ...this to force siunitx to actually use your fonts
    r'\usepackage{helvet}',     # set the normal font here
    r'\usepackage{sansmath}',   # load up the sansmath so that math -> helvet
    r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
]

plt.rc('text', usetex=True)
font_size = 10


class Plot:
    def __init__(self):
        width = 600.0 * 0.0138889   # pt
        height = width / (1 + 5**0.5) * 2
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(width, height), dpi=300)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=None, wspace=None, hspace=None)
        self.ax2 = None
        self.setup_plot()
        self.legend = []
        self.legend_loc = None
        self.lines = []
        self.marker_size = None
        
    def set_marker_size(self, marker_size):
        self.marker_size = marker_size

    def plot(self, x, y, series_name, linestyle='solid', marker=None):
        line = self.ax.plot(x, y, linestyle=linestyle, marker=marker, markersize=self.marker_size)[0]
        self.lines.append(line)
        self.legend.append(series_name)
        self.set_xlim(min(x), max(x))

    def plot_pandas(self, series, linestyle='solid', marker=None, series_name=None):
        line = self.ax.plot(series, linestyle=linestyle, marker=marker, markersize=self.marker_size)[0]
        self.x_label(series.index.name)
        self.lines.append(line)
        self.legend.append(series.name)

    def setup_plot(self):
        self.ax.grid('on')
        self.ax.tick_params(labelsize=font_size, pad=10)

    def add_arrows(self):
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        hw = 1. / 20. * (ymax - ymin)
        hl = 1. / 20. * (xmax - xmin)
        lw = 1.  # axis line width
        ohg = 0.3  # arrow overhang
        dps = self.fig.dpi_scale_trans.inverted()
        bbox = self.ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height
        yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
        yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height
        self.ax.arrow(xmin, ymin, 1.03 * (xmax - xmin), 0., fc='k', ec='k', lw=lw,
                      head_width=hw, head_length=hl, overhang=ohg,
                      length_includes_head=False, clip_on=False)
        self.ax.arrow(xmin, ymin, 0., 1.03 * (ymax - ymin), fc='k', ec='k', lw=lw,
                      head_width=yhw, head_length=yhl, overhang=ohg,
                      length_includes_head=False, clip_on=False)

    def show(self):
        self.make_legend()
        plt.show()

    def save_fig(self, filename):
        self.make_legend()
        plt.savefig(filename)

    def x_label(self, x_label):
        self.ax.set_xlabel(x_label, fontsize=font_size, labelpad=20)
        self.ax.xaxis.set_label_coords(0.9, -0.13)

    def y_label(self, y_label):
        self.ax.set_ylabel(y_label, fontsize=font_size, labelpad=20, rotation=0)
        self.ax.yaxis.set_label_coords(-0.15, 0.87)

    def y_label2(self, y_label):
        self.ax2.set_ylabel(y_label, fontsize=font_size, labelpad=20, rotation=0)
        self.ax2.yaxis.set_label_coords(1.1, 0.9)

    def make_legend(self):
        self.ax.legend(handles=self.lines, labels=self.legend, fontsize=font_size, loc=self.legend_loc)

    def set_legend_loc(self, location):
        self.legend_loc = location

    def title(self, title):
        self.ax.set_title(title, fontsize=font_size)

    def set_xlim(self, x_min, x_max):
        self.ax.set_xlim([x_min, x_max])

    def set_ylim(self, y_min, y_max):
        self.ax.set_ylim([y_min, y_max])

    def vertical_line(self, x, label):
        self.ax.axvline(x=x, ymin=0, ymax=50, color='r', label=label)

    def set_xoffset(self, offset):
        self.ax.get_xaxis().get_major_formatter().set_useOffset(offset)

    def close(self):
        plt.close(self.fig)

    def set_date_axis_format(self, date_format):
        self.ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(date_format))
        
    def set_axis_timezone(self, tz):
        plt.gca().xaxis_date(tz)

    def set_daily_ticks(self):
        self.ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
        
    def rotate_ticks(self, rotation):
        plt.xticks(rotation=rotation)

    def plot_second_y(self, x, y, series_name):
        self.ax2 = self.ax.twinx()
        self.ax2.plot(x, y, '-')
        self.legend.append(series_name)

    def add_second_y(self):
        self.ax2 = self.ax.twinx()
        self.ax2.tick_params(labelsize=font_size, pad=10)
        plt.subplots_adjust(right=0.85)
        #self.ax2.set_prop_cycle = self.ax._get_lines.prop_cycler    # Set the cycler to the same state
        for line in self.lines:
            next(self.ax2._get_lines.prop_cycler)
            
    def plot_second_y(self, x, y, series_name):
        if not self.ax2:
            self.add_second_y()
        line = self.ax2.plot(x, y, '-')[0]
        self.lines.append(line)
        self.legend.append(series_name)

    def plot_pandas_second_y(self, series, linestyle='-', marker='None'):
        if not self.ax2:
            self.add_second_y()
        line = self.ax2.plot(series, linestyle=linestyle, marker=marker, markersize=self.marker_size)[0]
        self.lines.append(line)
        self.legend.append(series.name)

    def scientific_notation(self, x=True, y=True):
        if x:
            self.ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        if y:
            self.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    def x_log(self):
        self.ax.set_xscale('log')
      
    def y_log(self):
        self.ax.set_yscale('log')
        
            
class Histogram(Plot):
  
  def plot_pandas(self, series, bins=10):
      series.hist(ax=self.ax, bins=bins)      

class GeoPlot(Plot):

    def add_collection(self, collection):
        self.ax.add_collection(collection)

    def save_fig(self, filename):
        plt.savefig(filename)


class Plot3D(Plot):
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 9), dpi=300)
        self.ax = Axes3D(self.fig)
        self.setup_plot()
        self.color_bar = None
        self.legend = []

    def plot(self, x, y, z):
        surf = self.ax.plot_surface(x, y, z, antialiased=False, cmap=cm.coolwarm, linewidth=0.1)
        self.color_bar = self.fig.colorbar(surf, shrink=0.5, aspect=10)

    def z_label(self, z_label):
        self.ax.set_zlabel(z_label, fontsize=font_size, labelpad=20)
        self.color_bar.set_label(z_label, fontsize=font_size)

    def view(self, x_angel, y_angel):
        self.ax.view_init(x_angel, y_angel) 


class SpherePlot:
    def __init__(self, radius):
        self.radius = radius
        self.fig = plt.figure(figsize=(15, 15), dpi=300)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.color_counter = 1
        self.current_color = None
        self.boundaries = None
        self.line_width = 1
        self.colormap = cm.Set1
        plt.tight_layout()

    def set_line_width(self, line_width):
        self.line_width = line_width

    def scale_line_width(self, factor):
        self.line_width *= factor

    def next_color(self):
        self.color_counter += 0.1
        if self.color_counter >= 1:
            self.color_counter = 0
        self.current_color = self.colormap(self.color_counter)
        return self.current_color

    def set_color_from_index(self, index, max_index):
        self.current_color = self.colormap((index+1)/max_index)

    def set_color(self, color):
        self.current_color = color

    def make_sphere(self, finess=20, wire=False, zoom=1, only_north=False, only_front=False):
        finess = complex(0, finess)
        lat, lon = numpy.mgrid[-numpy.pi / 2:numpy.pi / 2:finess, 0:2 * numpy.pi:finess]

        if only_north:
            lat, lon = numpy.mgrid[0:numpy.pi / 2:finess, 0:2 * numpy.pi:finess]
        elif only_front:
            lat, lon = numpy.mgrid[-numpy.pi / 2:numpy.pi / 2:finess, -numpy.pi / 2:numpy.pi / 2:finess]

        x = (self.radius * numpy.cos(lat) * numpy.cos(lon)) * zoom
        y = (self.radius * numpy.cos(lat) * numpy.sin(lon)) * zoom
        z = (self.radius * numpy.sin(lat)) * zoom
        if wire:
            self.ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='black', alpha=0.5, linewidth=self.line_width)
        else:
            self.ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=1, linewidth=0.1, shade=False)

    def make_divider(self):
        normal = numpy.array([0, 0, self.radius])
        point = numpy.array([0, 0, 0])
        d = -point.dot(normal)
        xx, yy = numpy.meshgrid(range(-1, 2), range(-1, 2))
        xx = xx * numpy.float64(self.radius)
        yy = yy * numpy.float64(self.radius)
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
        self.ax.plot_surface(xx, yy, z, rstride=1, cstride=1, color='black', alpha=1, linewidth=1, shade=False)

    def plot_point_coordinates(self, lat, lon):
        x, y, z = self.coord_to_carthesian(lat, lon)
        self.ax.scatter(x, y, z, color=self.current_color, marker='.', alpha=1, s=1, depthshade=False)

    def plot_point_radians(self, lat, lon):
        x, y, z = self.radians_to_carthesian(lat, lon)
        self.ax.scatter(x, y, z, color=self.current_color, marker='x', alpha=1, s=self.line_width*100, depthshade=False)

    def plot_point_carthesian(self, x, y, z):
        self.ax.scatter(x, y, z, color=self.current_color, marker='.', alpha=1, s=1, depthshade=False)

    def coord_to_carthesian(self, lat, lon):
        lat = math.radians(lat)
        lon = math.radians(lon)
        x = self.radius * numpy.cos(lat) * numpy.cos(lon)
        y = self.radius * numpy.cos(lat) * numpy.sin(lon)
        z = self.radius * numpy.sin(lat)
        return x, y, z

    def radians_to_carthesian(self, lat, lon):
        x = self.radius * numpy.cos(lat) * numpy.cos(lon)
        y = self.radius * numpy.cos(lat) * numpy.sin(lon)
        z = self.radius * numpy.sin(lat)
        return x, y, z

    def plot_geo_point(self, geo_point):
        lat = geo_point.latitude
        lon = geo_point.longitude
        self.plot_point_radians(lat, lon)

    def plot_geo_line(self, points):
        xs = []
        ys = []
        zs = []
        for point in points:
            lat = point.latitude
            lon = point.longitude
            x, y, z = self.radians_to_carthesian(lat, lon)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        self.ax.plot(xs=xs, ys=ys, zs=zs, color=self.current_color, alpha=1, linewidth=self.line_width)

    def plot_geo_line_carthesian(self, xs, ys, zs):
        self.ax.plot(xs=xs, ys=ys, zs=zs, color=self.current_color, alpha=1, linewidth=self.line_width)

    def plot_straight_line_from_geopoints(self, geopoint_a, geopoint_b):
        vector_a = geopoint_a.to_ecef_vector().pvector()
        vector_b = geopoint_b.to_ecef_vector().pvector()
        self.plot_straight_line_from_pvector(vector_a, vector_b)

    def nvector_to_pvector(self, nvector):
        return nvector * self.radius

    def plot_straight_line_from_nvector(self, nvector_a, nvector_b):
        pvector_a = self.nvector_to_pvector(nvector_a)
        pvector_b = self.nvector_to_pvector(nvector_b)
        self.plot_straight_line_from_pvector(pvector_a, pvector_b)

    def plot_straight_line_from_pvector(self, vector_a, vector_b):
        if self.geopoint_inside_view(vector_a) or self.geopoint_inside_view(vector_b):
            self.ax.plot(xs=(vector_a[0], vector_b[0]),
                         ys=(vector_a[1], vector_b[1]),
                         zs=(vector_a[2], vector_b[2]),
                         color=self.current_color, alpha=1, linewidth=self.line_width)

    def plot_straight_line_from_coordinates(self, point1, point2):
        x1, y1, z1 = self.coord_to_carthesian(point1[0], point1[1])
        x2, y2, z2 = self.coord_to_carthesian(point2[0], point2[1])
        self.plot_geo_line_carthesian((x1, x2), (y1, y2), (z1, z2))

    def plot_axes(self):
        self.ax.plot(xs=[0, 2 * self.radius], ys=[0, 0], zs=[0, 0])
        self.ax.plot(xs=[0, 0], ys=[0, 2 * self.radius], zs=[0, 0])
        self.ax.plot(xs=[0, 0], ys=[0, 0], zs=[0, 2 * self.radius])

    def set_view(self, elevation, azimuth):
        # Elev in degrees (i.e. latitude)
        self.ax.view_init(elev=elevation, azim=azimuth)

    def zoom_to(self, lat, lon, field_size, elevation=None, set_azimuth=False):
        if not elevation:
            elevation = lat
        x, y, z = self.coord_to_carthesian(lat, lon)
        self.boundaries = (x-field_size*2, x+field_size*2, y-field_size*2, y+field_size*2, z-field_size*2, z+field_size*2)
        self.ax.set_xlim([x-field_size, x+field_size])
        self.ax.set_ylim([y-field_size, y+field_size])
        self.ax.set_zlim([z-field_size, z+field_size])
        if set_azimuth:
            self.set_view(elevation=elevation, azimuth=lon)
        else:
            self.set_view(elevation=elevation, azimuth=0)
        self.line_width = 1e4/field_size

    def geopoint_inside_view(self, pvector):
        x, y, z = pvector[0], pvector[1], pvector[2]
        if not self.boundaries:
            return True
        elif x < self.boundaries[0]:
            return False
        elif x > self.boundaries[1]:
            return False
        elif y < self.boundaries[2]:
            return False
        elif y > self.boundaries[3]:
            return False
        elif z < self.boundaries[4]:
            return False
        elif z > self.boundaries[5]:
            return False
        else:
            return True

    def default_zoom(self):
        zoom = 1
        self.ax.set_xlim([-self.radius / zoom, self.radius / zoom])
        self.ax.set_ylim([-self.radius / zoom, self.radius / zoom])
        self.ax.set_zlim([-self.radius / zoom, self.radius / zoom])

    def save_fig(self, filename):
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        plt.savefig(filename, dpi=300)

    def show(self):
        plt.show()


if __name__ == '__main__':
    import nvector
    a = nvector.GeoPoint(10, 10, degrees=True)
    b = nvector.GeoPoint(20, 20, degrees=True)
    path = nvector.GeoPath(a, b)
    xs = []
    ys = []
    zs = []
    n_steps = 5
    for step in range(n_steps + 1):
        pvector = path.interpolate(step / n_steps).to_ecef_vector().pvector
        xs.append(pvector[0][0])
        ys.append(pvector[1][0])
        zs.append(pvector[2][0])
    plt = SpherePlot(radius=10)

