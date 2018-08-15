#import libraries for radar visualization
import numpy as np
import datetime
import glob
import json
import os
import netCDF4
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#suppress deprecation warnings
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

class goes16:

    def __init__(self):
        with open('config.json') as json_data:
            config = json.load(json_data)
        self.path = config.get('path')
        self.savedir = config.get('savedir','./')
        self.plot = config.get('plot', False)

    def make_cmap(self,colortable, position=None, bit=False):
       #Create full name
       file_path = colortable+'.txt'
       #Read file
       colors = np.loadtxt(file_path,skiprows=1)

       bit_rgb = np.linspace(0,1,256)
       if position == None:
           position = np.linspace(0,1,len(colors))
       else:
           if len(position) != len(colors):
               sys.exit("position length must be the same as colors")
           elif position[0] != 0 or position[-1] != 1:
               sys.exit("position must start with 0 and end with 1")
       if bit:
           for i in range(len(colors)):
               colors[i] = (bit_rgb[colors[i][0]],
                            bit_rgb[colors[i][1]],
                            bit_rgb[colors[i][2]])
       cdict = {'red':[], 'green':[], 'blue':[]}
       for pos, color in zip(position, colors):
           cdict['red'].append((pos, color[0], color[0]))
           cdict['green'].append((pos, color[1], color[1]))
           cdict['blue'].append((pos, color[2], color[2]))

       cmap = mpl.colors.LinearSegmentedColormap(os.path.basename(colortable),cdict,256)
       return cmap

    def truncate_cmap (self,cmap,n_min=0,n_max=256):
       color_index = np.arange(n_min,n_max).astype(int)
       colors = cmap(color_index)
       name = "truncated_{}".format(cmap.name)
       return plt.matplotlib.colors.ListedColormap(colors,name=name)

    def get_cmap(self,infile):
        if 'sat_IR' in infile:
            self.max_val = 300
            self.min_val = 170
            min_val = self.min_val ; max_val = self.max_val
            position = [0,(200.-min_val)/(max_val-min_val),
                 (208.-min_val)/(max_val-min_val),
                 (218.-min_val)/(max_val-min_val),
                 (228.-min_val)/(max_val-min_val),
                 (245.-min_val)/(max_val-min_val),
                 (253.-min_val)/(max_val-min_val),
                 (258.-min_val)/(max_val-min_val),1]

        new_cmap = self.make_cmap(infile, position=position)
        plt.register_cmap(cmap=new_cmap)
        self.cmap = new_cmap

    def get_files(self, path):
        files = sorted(glob.glob(path + '/*'))
        return files

    def read_nc_data(self,fname):
        self.ncId = netCDF4.Dataset(fname, 'r')

    def create_projection(self):
        rad2deg = 180./np.pi
        x1 = self.ncId.variables['x'][:]
        y1 = self.ncId.variables['y'][:]
        goes_imager_projection = self.ncId.variables['goes_imager_projection']
        req = goes_imager_projection.semi_major_axis
        rpol = goes_imager_projection.semi_minor_axis
        H = goes_imager_projection.perspective_point_height + req
        lon0 = goes_imager_projection.longitude_of_projection_origin

        x,y = np.meshgrid(x1,y1)

        #Calculate variables
        a = (np.sin(x))**2 + (np.cos(x)**2)*((np.cos(y)**2) + (req**2/rpol**2)*(np.sin(y)**2))
        b = -2.*H*np.cos(x)*np.cos(y)
        c = H**2 - req**2
        rs = (-1.*b - np.sqrt(b**2 - 4*a*c))/(2*a)
        sx = rs*np.cos(x)*np.cos(y)
        sy = -1.*rs*np.sin(x)
        sz = rs*np.cos(x)*np.sin(y) 
             
        #Calculate lat/lon
        self.lat = np.arctan((req**2/rpol**2)*(sz/np.sqrt((H-sx)**2 + sy**2))) * rad2deg
        self.lon = lon0 - np.arctan(sy/(H-sx)) * rad2deg

        geospatial_lat_lon_extent = self.ncId.variables['geospatial_lat_lon_extent']
        self.lat0 = geospatial_lat_lon_extent.geospatial_lat_center
        self.lon0 = geospatial_lat_lon_extent.geospatial_lon_center

        self.ll_lon = geospatial_lat_lon_extent.geospatial_westbound_longitude
        self.ur_lon = geospatial_lat_lon_extent.geospatial_eastbound_longitude
        self.ll_lat = geospatial_lat_lon_extent.geospatial_southbound_latitude
        self.ur_lat = geospatial_lat_lon_extent.geospatial_northbound_latitude

    def get_tb(self):
        rad = self.ncId.variables['Rad'][:]
        #Read in the constants from the netCDF file
        fk1 = self.ncId.variables['planck_fk1'][0]
        fk2 = self.ncId.variables['planck_fk2'][0]
        bc1 = self.ncId.variables['planck_bc1'][0]
        bc2 = self.ncId.variables['planck_bc2'][0]

        #Calculate brightness temperature [K]
        self.pvar = (fk2 / (np.log((fk1/rad)+1)) - bc1) / bc2

    def get_time(self):
        t = np.mean(self.ncId.variables['t'][:])
        start = datetime.datetime(2000,1,1,12,0,0)
        dt = datetime.timedelta(seconds=t)
        self.time = start + dt

    def plot_tb(self, it, cmap = 'jet', bbox = None):
        #set up a 1x1 figure for plotting
        fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(8,8),dpi=100)
        #set up a basemap with a cylindrical projection centered
        if bbox:
            self.ll_lon = bbox[0] ; self.ll_lat = bbox[1]
            self.ur_lon = bbox[2] ; self.ur_lat = bbox[3]
        m = Basemap(projection='cyl',lat_0=self.lat0,lon_0=self.lon0, 
                   llcrnrlat=self.ll_lat, llcrnrlon=self.ll_lon,
                   urcrnrlat = self.ur_lat, urcrnrlon=self.ur_lon,
                   resolution='i',area_thresh=1500.)

        shp = m.readshapefile("/rstor/freitagb/floods/urban_files/cb_2016_us_ua10_500k",'cities',drawbounds=False)
        for info,shape in zip(m.cities_info,m.cities):
            if 'Huntsville' in info['NAME10']:
        #    if 'Houston' in info['NAME10']:
                x,y = zip(*shape)
                m.plot(x,y,color='k',linewidth=1.0, ax=ax)

        levs = np.linspace(180,300,61,endpoint=True)
        ticks = np.linspace(180,300,7, endpoint=True)
        label = 'Brightness Temperature (K)'
        #define the plot axis to the be axis defined above
        ax = axes
        #normalize the colormap based on the levels provided above
        #norm = mpl.colors.BoundaryNorm(levs,256)
        norm = mpl.colors.Normalize(vmin=self.min_val,vmax=self.max_val)
        #create a colormesh of the reflectivity using with the plot settings defined above
        cs = m.pcolormesh(self.lon,self.lat,self.pvar,norm=norm,cmap=cmap,ax=ax,latlon=True)
        #add geographic boundaries and lat/lon labels
        m.drawparallels(np.arange(20,70,2.5),labels=[1,0,0,0],fontsize=12,
                        color='k',ax=ax,linewidth=0.001)
        m.drawmeridians(np.arange(-150,-50,2.5),labels=[0,0,1,0],fontsize=12,
                        color='k',ax=ax,linewidth=0.001)
        #m.drawcounties(linewidth=0.5,color='gray',ax=ax)
        m.drawstates(linewidth=0.25,color='k',ax=ax)
        m.drawcountries(linewidth=0.5,color='k',ax=ax)
        m.drawcoastlines(linewidth=0.5,color='k',ax=ax)
        fig.text(0.5,0.95,'GOES16 Band 13 at ' + self.time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                horizontalalignment='center',fontsize=16)
        #add the colorbar axes and create the colorbar based on the settings above
        cax = fig.add_axes([0.075,0.075,0.85,0.025])
        cbar = plt.colorbar(cs,cax=cax, norm=norm, ticks=ticks,orientation='horizontal')
        cbar.set_label(label,fontsize=12)
        cbar.ax.tick_params(labelsize=11)
        plt.savefig(self.savedir + '/GOES_IR_'+ self.time.strftime("%Y%m%d_%H%M%S") + '.png')
        plt.close()

    def GOES16(self):
        files = self.get_files(path=self.path)
        latlon = False
        for i,file in enumerate(files):
            self.read_nc_data(file)
            if not latlon:
                self.create_projection()
                latlon=True
            self.get_tb()
            self.get_time()
            if self.plot:
                self.get_cmap('goes_colortables/sat_IR')
                self.plot_tb(i, cmap=self.cmap, bbox=[-98,24,-90,32])
            self.ncId.close()
 
