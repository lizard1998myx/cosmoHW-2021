import numpy
import matplotlib as ml
import matplotlib.pyplot as plt
import gyoto.core
import gyoto.std
from xml.dom.minidom import parseString
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# example1: 3D photon trajectories of Schwarzchild BH
# bhs = Pt(spin=0,inc=45,mass=1)
# bhs.corn(size=0.07, n_points=10)

# example2: shadow of an extreme Kerr BH
# bhek = Pt(spin=1,inc=90,mass=1)
# bhek.shadow_image(fov=0.15, res=64)
# Pt(spin=1,inc=90,mass=1).shadow_image(fov=0.15, res=64)

class Pt():
    def __init__(self, spin, inc, mass=1, shadow=False):
        self.spin=spin  # spin in geometrical units (L), spin < mass
        if inc==0:
            print('Gyoto do not accept inc=0, use 0.001 instead')
            inc = 0.001  # this will give similar results
        self.inc=inc  # inclination in degrees (=theta in spherical coordinates)
        self.mass=mass  # mass in geometrical units (L)
        if self.spin > self.mass:
            print('Spin is larger than mass, this is not physical')
        self.shadow=shadow  # useless
        self.sc=None  # screen object
        self.ph=None  # photon object
        self.distance=100.  # distance of the observer, geometrical units (L)
        self.horizon=self.mass+np.sqrt(self.mass**2-self.spin**2)
        self.get_ph()  # initiate screen and phonton object

    # set up the metric, screen and photon
    def get_ph(self):
        ### Create a metric
        metric = gyoto.std.KerrBL()
        metric.spin(self.spin)
        # metric.mass(4e6, "sunmass")
        metric.mass(self.mass)
        # metric.mass(mass*6.78e-4, "sunmass")

        ### Create screen
        screen = gyoto.core.Screen()
        screen.metric(metric)
        screen.resolution(128)
        screen.time(1000., "geometrical_time")
        screen.distance(self.distance, "geometrical")
        screen.fieldOfView(30. / 100.)
        screen.inclination(self.inc, "degree")
        screen.PALN(0., "degree")

        gridshape = numpy.asarray((1, 3, 11), numpy.uint64)
        pgridshape = gyoto.core.array_size_t_fromnumpy1(gridshape)

        opacity = numpy.zeros(gridshape)
        popacity = gyoto.core.array_double_fromnumpy3(opacity)
        opacity[:, 0::2, 0::2] = 100.
        opacity[:, 1::2, 1::2] = 100.

        intensity = opacity * 0. + 1.
        pintensity = gyoto.core.array_double_fromnumpy3(intensity)

        # Create PatternDisk, attach grids, set some parameters
        pd = gyoto.std.PatternDisk()
        pd.copyIntensity(pintensity, pgridshape)
        pd.copyOpacity(popacity, pgridshape)
        pd.innerRadius(28)
        pd.outerRadius(28)
        pd.repeatPhi(8)
        # pd.redshift(False)
        pd.metric(metric)
        pd.showshadow(self.shadow)
        pd.rMax(50)

        sc = gyoto.core.Scenery()
        sc.metric(metric)
        sc.screen(screen)
        sc.astrobj(pd)
        sc.nThreads(8)

        ph=gyoto.core.Photon()
        # ph.setInitialCondition(sc.metric(), sc.astrobj(), sc.screen(), 0., 0.)
        # ph.hit()
        # n=ph.get_nelements()

        # Create NumPy arrays
        # t=numpy.ndarray(n)
        # r=numpy.ndarray(n)
        # theta=numpy.ndarray(n)
        # phi=numpy.ndarray(n)

        # Call Gyoto method that takes these arrays as argument:
        # ph.get_t(t)
        # ph.getCoord(t, r, theta, phi)

        # plt.plot(t, r, )
        # plt.show()
        self.sc = sc
        self.ph = ph

    # set the initial direction of the photon
    def print_initial(self, x, y):
        print('x=%.2f, y=%.2f' % (x, y))
        self.ph.setInitialCondition(self.sc.metric(), self.sc.astrobj(), self.sc.screen(), x, y)
        x_a = parseString(str(self.ph))
        print(x_a.firstChild.childNodes[-2].toprettyxml().replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' '))
            
    # bool: whether the photon (from x, y) escape the BH or not
    def escape(self, x, y, r_limit=3.):
        self.ph.setInitialCondition(self.sc.metric(), self.sc.astrobj(), self.sc.screen(), x, y)
        self.ph.hit()
        n = self.ph.get_nelements()
        t = numpy.ndarray(n)
        r = numpy.ndarray(n)
        theta = numpy.ndarray(n)
        phi = numpy.ndarray(n)
        self.ph.get_t(t)
        self.ph.getCoord(t, r, theta, phi)
        self.horizon=self.mass+np.sqrt(self.mass**2-self.spin**2)
        return r[0] >= r_limit
        
    # a image of shadow
    def shadow_array(self, fov, res):
        step = fov/res
        rad_list = numpy.arange(-0.5*fov, 0.5*fov+step, step)
        results = []
        for y in rad_list:
            results.append([])
            for x in rad_list:
                results[-1].append(self.escape(x, y))
        return np.array(results)
        
    # a image of horizon
    def horizon_array(self, fov, res):
        step = fov/res
        rad_list = numpy.arange(-0.5*fov, 0.5*fov+step, step)
        results = []
        for y in rad_list:
            results.append([])
            for x in rad_list:
                r = np.sqrt(x**2+y**2)*self.distance
                results[-1].append(r<=self.horizon)
        return np.array(results)
    
    # plot the shadow and horizon
    # fov is the field of view, resolution is number of pixels in each side
    def shadow_image(self, fov=0.15, res=100, ax=None):
        title='Black hole shadow a/m=%.1f inc=%.0f' % (self.spin/self.mass, self.inc)
        filename = 'shadowRes-%i_a-%.1f_i-%.0f.png' % (res, self.spin/self.mass, self.inc)
        sha=self.shadow_array(fov, res)*1.
        hor=self.horizon_array(fov, res)*1.
        indices_unit_length = (1/self.distance)*(res/fov)
        max_l = int(self.distance*fov/2)//2*2
        loc = []
        label = []
        for i in range(-1*max_l, max_l+1, 2):
            label.append(i)
            loc.append((res/2)+indices_unit_length*i)
        if ax is None:
            plt.imshow(sha-hor, origin='lower')
            plt.title(title)
            plt.xticks(loc, label)
            plt.yticks(loc, label)
            plt.grid(True)
            plt.savefig(filename)
        else:
            ax.imshow(sha-hor, origin='lower')
            ax.set_xticks(loc)
            ax.set_yticks(loc)
            ax.set_xticklabels(label)
            ax.set_yticklabels(label)
            ax.grid(True)

    # get coordinates of the trajectory of photon
    def get_coords(self):
        n = self.ph.get_nelements()
        t = numpy.ndarray(n)
        r = numpy.ndarray(n)
        theta = numpy.ndarray(n)
        phi = numpy.ndarray(n)
        self.ph.get_t(t)
        self.ph.getCoord(t, r, theta, phi)
        xx = r * np.sin(theta) * np.sin(phi)
        yy = r * np.sin(theta) * np.cos(phi)
        zz = r * np.cos(theta)
        return -1*xx, yy, -1*zz  # changes applied to the coordinates

    # plot single trajectory
    def plotter(self, x, y):
        ax=self.set_ax()
        self.add(x, y, ax)
        ax.set_title('%.2f %.2f' % (x, y))
        plt.show()

    # set up matplotlib axis
    def set_ax(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_zlim(-15, 15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        # ax.legend()
        return ax

    # add a trajectory into the axis
    def add(self, x, y, ax):
        self.print_initial(x, y)
        self.ph.hit()
        crs = self.get_coords()
        ax.plot(*crs, label='%.2f %.2f' % (x, y), color='grey')

    # add the horizon of BH into the 3D-axis
    def circle(self, ax, delta_phi=0.1):
        r = self.horizon
        for phi in np.arange(0, 2*np.pi, delta_phi):
            the = np.arange(0, np.pi, 0.01)
            x1 = r * np.sin(the) * np.sin(phi)
            y1 = r * np.sin(the) * np.cos(phi)
            z1 = r * np.cos(the)
            ax.plot(x1, y1, z1, color='black')

    # plot a set of photons (rectangularly distributed)
    def grid(self, i_list=np.arange(-0.1, 0.1, 0.02), j_list=[0]):
        ax = self.set_ax()
        for i in i_list:
            for j in j_list:
                self.add(i, j, ax)
        self.circle(ax=ax)
        plt.show()

    # grid(np.arange(-0.1, 0.1, 0.01), [0])
    # grid(np.arange(-0.1, 0.1, 0.01), np.arange(-0.1, 0.1, 0.01))

    # plot a corn of phontons
    def corn(self, size, n_points):
        ax = self.set_ax()
        def r_func():
            return size*np.random.randn()
        # r_func = np.random.random
        for i in range(n_points):
            self.add(r_func(), r_func(), ax)
        # ax.legend()
        self.circle(ax=ax)
        plt.show()
        
    # plot a sector of photons
    def shan(self, size, n_points):
        ax = self.set_ax()
        def r_func():
            return size*np.random.randn()
        # r_func = np.random.random
        for i in range(n_points):
            self.add(0, r_func(), ax)
        # ax.legend()
        self.circle(ax=ax)
        plt.show()
