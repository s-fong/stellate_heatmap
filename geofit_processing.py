# copy of trace_geofit_outline.py, except made to use zinc functions instead of hacks using readline() etc

# 29 Sep 20: flip ONCE.
# 29 Sep 20: merged with fid_xi_average_shape.py
# 30 sep 20: tidy up. Read and write elements file from source data, with only coordinates being changed.
# 5 oct 20: nerve junctions don't have heatmaps. Soma have heatmaps.
# 8 oct 20: with concave hull of meshes, externally fit points on scaffoldfitter to form a mesh that heatmap will be built on.
# 14 Oct: including units, so approximate soma from fragment files can be included

import sys
import os
import pylab as pl
from descartes import PolygonPatch
from shapely.ops import cascaded_union, polygonize
from shapely.geometry import mapping, shape
import math
from scipy.signal import fftconvolve
import scipy.optimize as opt
import numpy as np
from math import *
from math import floor, ceil, sqrt
from copy import deepcopy
from rasterio import Affine, features
import matplotlib.pyplot as plt
from stellate_heatmap_tools import flip_system, translate_system, rotate_points, twoD_Gaussian, create_colormap, cart2pol, extrude_3D_in_1D, xml_soma_nerve_connections, write_heatmap_com_file_single, refine_mesh, zinc_read_exf_file, zinc_find_ix_from_real_coordinates, zinc_write_element_xi_file, write_parameter_set_string_values
from concave_hull_tools import alpha_shape, plot_polygon
from find_somanerve_distance import main_somanerve
from write_rotate_exf_marker_file import main_writerotatefile
from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field
from opencmiss.zinc.node import Node
from opencmiss.zinc.element import Element, Elementbasis
from opencmiss.utils.zinc.field import findOrCreateFieldCoordinates, findOrCreateFieldGroup, findOrCreateFieldStoredString, findOrCreateFieldFiniteElement, findOrCreateFieldNodeGroup

# default magnification of the whole body is 10x

def refactor_zinc_arrays(x,y,z,xyz_all_dict,xyz_marker, dir):
    # purpose: to fix correspondence of [points], {xyz_all_dict}, xyz_DA after possible transformation
    points = np.array([[x[i], y[i], z[i]] for i in range(len(x))])
    for i, key in enumerate(xyz_all_dict):
        xyz_all_dict[key] = points[i]
    inorth, ieast, iwest = dir
    xyz_DA = xyz_marker[iwest]
    xyz_TST = xyz_marker[ieast]
    xyz_VA = xyz_marker[inorth]

    return(points, xyz_all_dict, xyz_DA, xyz_TST, xyz_VA)


def gaussian_blur(in_array, gt, size):
    """Gaussian blur, returns tuple `(ar, gt2)` that have been expanded by `size`"""
    # expand in_array to fit edge of kernel; constant value is zero
    padded_array = np.pad(in_array, size, 'constant')
    # build kernel
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(size)))
    g = (g / g.sum()).astype(in_array.dtype)
    # do the Gaussian blur
    ar = fftconvolve(padded_array, g, mode='full')
    # convolved increased size of array ('full' option); update geotransform
    gt2 = Affine(
        gt.a, gt.b, gt.xoff - (2 * size * gt.a),
        gt.d, gt.e, gt.yoff - (2 * size * gt.e))
    return ar, gt2

def average_polygon(shapes, plot_concave_hull):
    max_shape = cascaded_union([shape(s) for s in shapes])
    minx, miny, maxx, maxy = max_shape.bounds
    dx = dy = 2#0.25*4  # grid resolution; this can be adjusted
    lenx = dx * (ceil(maxx / dx) - floor(minx / dx))
    leny = dy * (ceil(maxy / dy) - floor(miny / dy))
    assert lenx % dx == 0.0
    assert leny % dy == 0.0
    nx = int(lenx / dx)
    ny = int(leny / dy)
    gt = Affine(dx, 0.0, dx * floor(minx / dx), 0.0, -dy, dy * ceil(maxy / dy))

    pa = np.zeros((ny, nx), 'd')
    for s in shapes:
        r = features.rasterize([s], (ny, nx), transform=gt)
        pa[r > 0] += 1
    pa /= len(shapes)  # normalise values

    size = 3
    spa, sgt = gaussian_blur(pa, gt, size)

    thresh = 0.5  # median nominal: 0.5
    pm = np.zeros(spa.shape, 'B')
    pm[spa > thresh] = 1

    poly_shapes = []
    for sh, val in features.shapes(pm, transform=sgt):
        if val == 1:# or True:
            poly_shapes.append(shape(sh))
    if not any(poly_shapes):
        raise ValueError("could not find any shapes")

    ###################################################################
    # find polygon with largest area to keep
    # keep = poly_shapes[0]
    parea = [p.area for p in poly_shapes]
    imax = parea.index(int(max([p.area for p in poly_shapes])))
    poly_shapes = poly_shapes[imax]
    ###################################################################

    avg_poly = cascaded_union(poly_shapes)
    simp_poly = avg_poly.simplify(sqrt(dx ** 2 + dy ** 2))
    simp_shape = mapping(simp_poly)

    # alpha shape of avg_poly if needed
    if plot_concave_hull:
        fig = pl.figure(figsize=(10, 10))
        margin = 0.3
        ax = fig.add_subplot(121)
        patch = PolygonPatch(simp_poly, fc='#999999', ec='#000000', fill=True, zorder=-1)
        ax.add_patch(patch)
        x_min, y_min, x_max, y_max = simp_poly.bounds
        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([y_min - margin, y_max + margin])
        ax.axis('equal')
        ax.set_title('polygon of largest area post cascaded_union')

    keep = simp_shape['coordinates'][0]
    for p in simp_shape['coordinates'][1:]:
        if len(p)> len(keep):
            keep = p

    pk = np.array(list([list(i) for i in keep]))
    simp_shape['coordinates'] = keep
    # smooth the signal

    alpha = 1.5/1e3
    concave_hull, _ = alpha_shape(pk, alpha=alpha)
    simp_poly = concave_hull.simplify(sqrt(dx ** 2 + dy ** 2))
    simp_shape = mapping(simp_poly)

    if plot_concave_hull:
        ax = fig.add_subplot(122)
        patch = PolygonPatch(simp_poly, fc='#999999', ec='#000000', fill=True, zorder=-1)
        ax.add_patch(patch)
        ax.plot([x[0] for x in simp_shape['coordinates'][0]], [x[1] for x in simp_shape['coordinates'][0]], 'x')
        x_min, y_min, x_max, y_max = simp_poly.bounds
        ax.set_xlim([x_min - margin, x_max + margin])
        ax.set_ylim([y_min - margin, y_max + margin])
        ax.axis('equal')
        ax.set_title('concave hull')
        pl.show()

    return simp_shape


def average_shape(files, data_path, ic, meshFile, plot_graphs, all_fids):

    plot_individual_hulls, plot_concave_hull,  _ = plot_graphs

    x_all = []
    y_all = []
    z_all = []
    xe_all = []
    ye_all = []
    xyz_DA_all = []
    xyz_VA_all = []
    xyz_TST_all = []
    xyz_fid_dim = {c:[] for c in all_fids}
    chull_all = []
    shapes = []

    xyz_all_samples_dict = {}
    dxyz_all_samples_dict = {}
    # marks for transformation of points
    west = 'Dorsal ansa subclavia'
    east = 'Thoracic sympathetic nerve trunk'
    north = 'Ventral ansa subclavia'

    first_fid = []
    # first_fid = True if not ic else False

    for iif, f in enumerate(files):#[0:2]:
        print(f)

        derv_present = 1
        marker_present = 1
        all_node_num, xyz_all, dxyz_single, xyzGroups, element_list, xyz_marker, marker_names, marker_nodenum, marker_elemxi = \
        zinc_read_exf_file(data_path + f, 0, derv_present, marker_present, [],[], 3)

        iwest = marker_names.index(west)
        ieast = marker_names.index(east)
        inorth = marker_names.index(north)
        xyz_DA = xyz_marker[iwest]
        xyz_TST = xyz_marker[ieast]
        xyz_VA = xyz_marker[inorth]

        xyz_all_dict = {}
        dxyz_all_dict = {}
        nodenum = list(range(1, len(xyz_all)+1))
        for ir, row in enumerate(xyz_all):
            newdict = {nodenum[ir]: row}
            xyz_all_dict.update(newdict)
        for ir, row in enumerate(dxyz_single):
            newdict = {nodenum[ir]: row}
            dxyz_all_dict.update(newdict)

        # transform the coordinates:
        # 1. Flip if TST.x < DA.x                       (if centroid.x < DA.x)
        # 2. Translate by fid:DA, so DA lies on (0,0)
        # 3. Rotate, so DA and TST lie on (0,0)         (by long axis)
        all_node_num = [key for key in xyz_all_dict]
        points = np.array(xyz_all)
        pointsxy = np.array([[x for x in points[p][:2]] for p in range(len(points))])
        x = [p[0] for p in pointsxy]
        y = [p[1] for p in pointsxy]
        z = [p[2] for p in points]
        centroid = [np.mean(x), np.mean(y)]
        [x, y, z, flip] = flip_system(x, y, z, [], [], [], [xyz_DA, xyz_VA, xyz_TST], [], centroid, [], [], [], [], 0)

        # transform derivatives in dxyz_single - FLIP: change signs for all component of that direction!
        if flip:
            for i, dnode in enumerate(dxyz_single):
                for j, row in enumerate(dnode):
                    dxyz_single[i][j][1] *= -1
            [xm, ym, zm, _] = flip_system([xm[0] for xm in xyz_marker], [xm[1] for xm in xyz_marker], [xm[2] for xm in xyz_marker], [], [], [], [xyz_DA, xyz_VA, xyz_TST], [], centroid, [], [], [], [], 1)
            xyz_marker = [[xm[i], ym[i], zm[i]] for i in range(len(xm))]

            points, xyz_all_dict, xyz_DA, xyz_TST, xyz_VA = refactor_zinc_arrays(x,y,z,xyz_all_dict, xyz_marker, [inorth, ieast, iwest])


         # if TST is to the left of DA, do a 180deg prerotation
        if xyz_DA[0] > xyz_TST[0]:
            theta = pi
            [pointsRot, _] = rotate_points(xyz_all_dict, theta, [])
            [xyz_marker, _] = rotate_points(xyz_marker, theta, [])
            for id, dnode in enumerate(dxyz_single):
                [dxyz_singlenode, _] = rotate_points(dnode, theta, [])
                dxyz_single[id] = dxyz_singlenode
            points = np.array(pointsRot)
            for i, key in enumerate(xyz_all_dict):
                xyz_all_dict[key] = points[i]
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z = [p[2] for p in points]
            points, xyz_all_dict, xyz_DA, xyz_TST, xyz_VA = refactor_zinc_arrays(x,y,z,xyz_all_dict, xyz_marker, [inorth, ieast, iwest])

        fid = [xyz_DA, xyz_TST]
        [pointsRot, theta] = rotate_points(xyz_all_dict, [], fid)
        [xyz_marker, _] = rotate_points(xyz_marker, theta, [])
        for id, dnode in enumerate(dxyz_single):
            [dxyz_singlenode, _] = rotate_points(dnode, theta, [])
            dxyz_single[id] = dxyz_singlenode

        points = np.array(pointsRot)
        for i, key in enumerate(xyz_all_dict):
            xyz_all_dict[key] = points[i]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        z = [p[2] for p in points]
        points, xyz_all_dict, xyz_DA, xyz_TST, xyz_VA = refactor_zinc_arrays(x, y, z, xyz_all_dict, xyz_marker,
                                                                                [inorth, ieast, iwest])

        # BELOW: correct, as order is preserved.
        [x, y, z] = translate_system(x,y,z,centroid) # pzero: xyz_DA
        [xm, ym, zm] = translate_system(xyz_marker, _,_,centroid)  # pzero: xyz_soma5
        xyz_marker = [[xm[i], ym[i], zm[i]] for i in range(len(xm))]

        points = np.array([[x[i], y[i], z[i]] for i in range(len(x))])
        for i, key in enumerate(xyz_all_dict):
            xyz_all_dict[key] = points[i]
        points, xyz_all_dict, xyz_DA, xyz_TST, xyz_VA = refactor_zinc_arrays(x, y, z, xyz_all_dict, xyz_marker,
                                                                                [inorth, ieast, iwest])

        if False: # check plots
            fig, ax = plt.subplots(1, 2)
            ax[0].plot([p[0] for p in pointsxy],[p[1] for p in pointsxy], '.')
            ax[0].axis('equal')
            ax[0].set_title(f+'original')
            ax[1].plot(x,y,'-.')
            ax[1].axis('equal')
            ax[1].set_title('transformed')
            plt.show()

        # pointsxy = np.array([[x for x in points[p][:2]] for p in range(len(points))])
        zmean = np.mean(points[:,2])
        halfpoints = [p[:2] for p in points if p[2] >= zmean]
        xp = [p[0] for p in halfpoints]
        yp = [p[1] for p in halfpoints]
        halflen = len(halfpoints)

        alpha = 0.51/1e3 # 2.5/1e3  # bigger alpha = tighter fit, more nodes.
        found = False
        iters = 0
        while not found:
            # take only one layer in the z direction
            concave_hull, edge_points =  alpha_shape(np.array(halfpoints), alpha) #alpha_shape(points[:,[0,1]], alpha)
            try:
                xe = [c for c in concave_hull.exterior.coords.xy[0]]
                ye = [c for c in concave_hull.exterior.coords.xy[1]]
                if len(xe)<halflen+1:
                    alpha += 0.1/1e3
                else:
                    found = True
                if iters > 2:
                    found = True
                    print('exceeded 2 iterations for finding alpha shape')
            except:
                alpha -= 0.1/1e3
            iters += 1

        # plotting individual concave hulls
        if plot_individual_hulls:
            plot_polygon(concave_hull)
            pl.plot(xp,yp,'o', color='#f16824')
            pl.plot(xe,ye,'x', markersize=10)
            pl.title(f)
            pl.axis('Equal')
            pl.show()

        ptup = [[tuple([xe[i], ye[i]]) for i in range(len(xe))]]
        sh = {'type': 'Polygon', 'coordinates': ptup}
        shapes.append(dict(sh))

        xe_all.append(xe)#[:-1])
        ye_all.append(ye)#[:-1])
        xyz_DA_all.append(xyz_DA)
        xyz_VA_all.append(xyz_VA)
        xyz_TST_all.append(xyz_TST)
        chull_all.append(concave_hull)

        # find coordinates of all fiducials and soma in dimensional x units
        for n in all_fids:
            im = [idx for idx, s in enumerate(marker_names) if n in s][0]
            xyz_fid_dim[n].append(xyz_marker[im])


        sampledict = {f: xyz_all_dict}
        xyz_all_samples_dict.update(sampledict)
        if iif == 0:
            dxyz_all_samples_dict = {n: [] for n in dxyz_all_dict.keys()}
        for nkey in dxyz_all_dict.keys():
            dxyz_all_samples_dict[nkey].append(dxyz_all_dict[nkey])

        # find maximum length and height of sample
        print('xlen = ' + str(max(x) - min(x)))
        print('ylen = ' + str(max(y) - min(y)))


    for key in xyz_all_samples_dict.keys():
        x_all.append([xyz_all_samples_dict[key][node][0] for node in xyz_all_samples_dict[key].keys()])
        y_all.append([xyz_all_samples_dict[key][node][1] for node in xyz_all_samples_dict[key].keys()])
        z_all.append([xyz_all_samples_dict[key][node][2] for node in xyz_all_samples_dict[key].keys()])


    ################################################
    # FID AVG MARKERS (geometric average exf)
    ################################################
    x_avg = list(np.average(x_all, axis=0))
    y_avg = list(np.average(y_all, axis=0))
    z_avg = list(np.average(z_all, axis=0))
    # dxyz_avg = np.zeros([len(x_avg),len(dxyz_single[0]),len(dxyz_single[0][0])])
    dxyz_avg = []
    for nkey in dxyz_all_samples_dict.keys():
        dm = dxyz_all_samples_dict[nkey]
        dxyz_avg.append([[np.mean([dm[i][i1][i2] for i in range(len(dm))]) for i2 in range(len(dm[0][0]))] for i1 in range(len(dm[0]))])

    # -------- averaged nodes -------- they are not used.
    if first_fid and False: #not meshFile.is_file():
        # with data's region, modify the coordinates. Then write out with nothing else changed.
        #   ############### SOME REPEATS ###############
        outputRegion = context.getDefaultRegion()
        outputRegion.readFile(data_path + f)
        oldChild = outputRegion.findChildByName('raw_data')
        outputRegion.removeChild(oldChild)
        fmOut = outputRegion.getFieldmodule()
        fmOut.beginChange()
        cache = fmOut.createFieldcache()
        coordinates = findOrCreateFieldCoordinates(fmOut)

        nodes = fmOut.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)

        # nodeIdentifier = 1
        nodeIter = nodes.createNodeiterator()
        node = nodeIter.next()
        nodeIdentifier = 1
        while node.isValid() and nodeIdentifier <= nodes.getSize()-len(all_fids):
        # for ix, xval in enumerate(x_avg):
            # node = nodes.createNode(nodeIdentifier, nodetemplate)
            ix = nodeIdentifier - 1
            cache.setNode(node)
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, [x_avg[ix], y_avg[ix], z_avg[ix]])
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dxyz_avg[ix][0]+[0])
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, dxyz_avg[ix][1]+[0])
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, dxyz_avg[ix][2]+[0])
            nodeIdentifier += 1
            node = nodeIter.next()

        fmOut.endChange()
        out_file = meshFile
        outputRegion.writeFile(out_file)

        # -------- MARKERS -------- written out to different file. Only write the xi coords and the marker_name string.
        # the below assumes you have a FieldGroup named 'marker'
        if False:
            fmOut.endChange()
            outputRegion.getFieldmodule().defineAllFaces()  # add 1D and 2D elements  # must define the mesh3D manually first
            outputRegion.writeFile(path + 'test\\TEST_write_within_function.exf')


    ###########################################
    # AVERAGE POLYGON
    ###########################################
    print('Entering average_polygon routine.')
    simp_shape = average_polygon(shapes, plot_concave_hull)
    shape_list = []
    for j in range(len(simp_shape['coordinates'][0])):
        xy = list(simp_shape['coordinates'][0][j])
        shape_list.append([xy[0], xy[1]])

    return (shape_list, xyz_fid_dim, [], [min(z), max(z)], all_node_num, xyz_marker, marker_names)



def twoD_gauss_soma(chosen_fid, xy, xyfid, marker_mean, gauss_bounds, cmaps, hb, xlen, normalise, plot_heatmap):

    print('Entering heatmap routine for ' + chosen_fid)

    chosen_fid = [n for n in list(marker_mean.keys())] if chosen_fid == 'all' else [chosen_fid]
    xybody = xy.copy()
    N = 20 #30 #200
    x = np.linspace(hb[0][0], hb[0][1], N)
    y = np.linspace(hb[1][0], hb[1][1], N)
    xarr = x.copy()
    yarr = y.copy()
    x, y = np.meshgrid(x, y)
    xdata_tuple = (x, y)

    fig, ax = plt.subplots(1, 1)
    ax.plot([x[0] for x in xybody], [x[1] for x in xybody], '.-', linewidth=2, color='silver')
    sd = 0.512*(xlen*(not normalise) + normalise)    #0.12
    heatmaps = {}
    for j, key in enumerate(chosen_fid):
        # print(key)
        param = (1, marker_mean[key][0], marker_mean[key][1], sd, sd, 0)
        initial_guess = deepcopy(param)
        data = np.zeros([N,N])
        if isinstance(xyfid[key][0], float) or isinstance(xyfid[key][0], int):
            row = xyfid[key]
            xind = (np.abs(xarr - row[0])).argmin()
            yind = (np.abs(yarr - row[1])).argmin()
            data[xind][yind] = 1
            ax.plot(row[0], row[1], 'x', color=cmaps[j].split('s_')[0])
        else:
            for row in xyfid[key]:
                xind = (np.abs(xarr - row[0])).argmin()
                yind = (np.abs(yarr - row[1])).argmin()
                data[xind][yind] = 1
                ax.plot(row[0], row[1], 'x', color=cmaps[j].split('s_')[0])

        data = data.flatten('F')
        if 'soma' in key:
            gauss_bounds = ((-np.inf, -np.inf, -np.inf, 0, 0, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
        popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess, maxfev=1000000, ftol = 1e-3, bounds=gauss_bounds)
        # print(popt)
        data_fitted = twoD_Gaussian((x, y), *popt)
        data_resh = data_fitted.reshape(N,N)
        # data_resh = np.transpose(data_resh)
        newdict = {key:data_resh}
        heatmaps.update(newdict)
        pa = ax.imshow(data_resh, interpolation='nearest', cmap=cmaps[j], alpha = 0.9, origin='bottom', extent=(x.min(), x.max(), y.min(), y.max()))
        ax.text(marker_mean[key][0],marker_mean[key][1], key, color=cmaps[j].split('s_')[0])
        plt.title('Fiducial heatmap')

        plt.axis('equal')
        if False:
            cba = plt.colorbar(pa, shrink=0.25)
            cba.set_clim(0, param[0])
            cba.set_label('positive')
    if plot_heatmap:
        plt.show()
    return(xdata_tuple, heatmaps)


def make_heatmap(fitter_path, do_heatmap, chosen_fid, meshFile, geoMeshFile, heatmap_bounds, all_nerves, plot_graphs, shapes, projected_markers, marker_mean):

    plot_individual_hulls, plot_concave_hull,  plot_heatmap = plot_graphs
    two_layer = True
    files = os.listdir(fitter_path)
    files = [f for f in files if f.endswith('.exf')]

    cmaps = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys', 'Reds']
    cmaps *= 3
    cmaps = [create_colormap(c) for c in cmaps]

    if shapes:
        first_fid = 0
        simp_shape = shapes['simp_shape']
        xyz_fid = shapes['xyz_fid']
        elemdict = shapes['elemdict']
        zbound = shapes['zbound']
        nodenums = shapes['nodenums']
        shapes_o = shapes
        xyz_marker_dim = []
    else:
        first_fid = 1
        simp_shape, xyz_fid, elemdict, zbound, nodenums, xyz_marker_dim, marker_names = average_shape(files, fitter_path, 0, meshFile, plot_graphs, all_nerves) #all_soma
        shapes_o = dict()
        shapes_o['simp_shape'] = simp_shape
        shapes_o['xyz_fid'] = xyz_fid
        shapes_o['zbound'] = zbound
        shapes_o['nodenums'] = nodenums
        shapes_o['elemdict'] = elemdict

    xy_avg = [[x[0], x[1]] for x in simp_shape]
    xyz_avg = [[x[0], x[1], zbound[0]] for x in xy_avg]
    xyz_fid_avg = {c:[] for c in list(xyz_fid.keys())}
    x = [x[0] for x in xyz_avg]
    y = [x[1] for x in xyz_avg]
    z = [x[2] for x in xyz_avg]
    centroid = [np.mean(x), np.mean(y), np.mean(z)]

    for key in xyz_fid:
        if not isinstance(xyz_fid[key][0],float) and not isinstance(xyz_fid[key][0],int):
            avgx = np.mean([row[0] for row in xyz_fid[key]])
            avgy = np.mean([row[1] for row in xyz_fid[key]])
            avgz = np.mean([row[2] for row in xyz_fid[key]])
        else:
            avgx = xyz_fid[key][0]
            avgy = xyz_fid[key][1]
            avgz = xyz_fid[key][2]
        xyz_fid_avg[key] = [avgx, avgy, avgz]

    if first_fid and False:
        plt.figure()
        plt.plot([x[0] for x in simp_shape], [x[1] for x in simp_shape], '.')
        for key in xyz_fid_avg.keys():
            plt.plot(xyz_fid_avg[key][0], xyz_fid_avg[key][1], 'rx')
        plt.show()

    if two_layer:
        # change 2D image to 3D (extruded and through-z elements added, in 1D)
        xyz_avg = extrude_3D_in_1D(xyz_avg, zbound[1])
    nodelist = list(range(1, len(xyz_avg)+1))

                #######################################################################################
                # STOP! # STOP! # STOP! # STOP! # STOP! # STOP! # STOP! # STOP! # STOP! # STOP! # STOP!
                #######################################################################################
                # after making concave hull, write to .ex data file.
                # Externally fit on scaffoldfitter.
                # use the output geofit mesh as inputs for the remaining heatmap code below.
                #######################################################################################

    concaveFileName = 'average_concave_hull_mesh'
    if first_fid:
        concaveMeshFile = path + 'concave_hull\\%s.ex' %(concaveFileName)
        context = Context("Example")
        region = context.getDefaultRegion()
        xyzlist = [list(x) for x in xyz_avg]
        markers = {}
        # arbitrary positions for markers: for scaffoldfitter
        indexmarker = {}
        indexmarker['ICN'] = y.index(int(max(y)))
        indexmarker['DA'] = x.index(int(min(x)))
        th_C8_1,_ = cart2pol(xyzlist[indexmarker['DA']+1], centroid)
        th_C8_2,_ = cart2pol(xyzlist[indexmarker['DA']-1], centroid)

        markers['Thoracic sympathetic nerve trunk'] = xyzlist[x.index(int(max(x)))]
        markers['Dorsal ansa subclavia'] = xyzlist[indexmarker['DA']]
        # markers['Cervical spinal nerve 8'] = xyzlist[indexmarker['DA'] +1] if \
        #     abs(th_C8_1)<abs(th_C8_2) else xyzlist[indexmarker['DA'] -1]
        markers['Inferior cardiac nerve'] = xyzlist[indexmarker['ICN']]
        markers['Ventral ansa subclavia'] = xyzlist[indexmarker['ICN']-1] if \
            xyzlist[indexmarker['ICN']-1][0]<xyzlist[indexmarker['ICN']][0] else xyzlist[indexmarker['ICN']+1]
        print('*** Writing '+concaveMeshFile+' ***')
        zinc_write_element_xi_file(concaveMeshFile, xyzlist, markers)

        # call write_rotate to add LR markers to the concave hull mesh file, prior to scaffoldfitting:
        rotate_data_path = path+"concave_hull\\"
        print('*** Writing transformed LR concave hull mesh file ***')
        main_writerotatefile(rotate_data_path)

    # normalise all data by xlen
    normalise = False
    x = [x[0] for x in simp_shape]
    xlen = max(x)
    if first_fid:
        print('avg xlen '+str(xlen))
    if normalise:
        xy_avg = [[x[0]/xlen, x[1]/xlen] for x in simp_shape]
        for key in xyz_fid.keys():
            xyz_fid[key] = [[x/xlen for x in row] for row in xyz_fid[key]] #/xlen
    else:
        heatmap_bounds = [[h*xlen for h in row] for row in heatmap_bounds]

    # heatmap of fiducials
    if do_heatmap:
        # find real xyz of theser points - do evaluateReal and hostMesh.
        if projected_markers: # for soma. otherwise use the fiducial markers (nervejunctions)
            marker_names = []
            xyz_marker = []
            xyz_fid = {c:[] for c in projected_markers.keys()}
            for key in projected_markers:
                context = Context("Example")
                region = context.getDefaultRegion()
                region.readFile(geoMeshFile)
                fm = region.getFieldmodule()
                mesh = fm.findMeshByDimension(3)
                cache = fm.createFieldcache()
                coordinates = findOrCreateFieldCoordinates(fm)
                for ie, exi in enumerate(projected_markers[key]['elementID']):
                    element = mesh.findElementByIdentifier(exi)
                    xi = projected_markers[key]['xi'][ie]
                    cache.setMeshLocation(element, xi)
                    result, x = coordinates.evaluateReal(cache, 3)
                    marker_names.append(key)
                    xyz_marker.append(x)
                    xyz_fid[key].append(x)

        if first_fid:
            marker_mean = {c: [] for c in xyz_fid.keys()}
            for key in xyz_fid.keys():
                if isinstance(xyz_fid[key][0],float) or isinstance(xyz_fid[key][0],int):
                    marker_mean[key] = xyz_fid[key]
                else:
                    marker_mean[key] = np.mean(np.array(xyz_fid[key]), 0)

            for key in xyz_fid.keys(): # recalculate means
                if isinstance(xyz_fid[key][0], float) or isinstance(xyz_fid[key][0], int):
                    marker_mean[key] = xyz_fid[key]
                else:
                    marker_mean[key] = np.mean(np.array(xyz_fid[key]), 0)

        # param = [] #(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset)
        min_lim = 0.025
        gauss_bounds = ((-np.inf, -np.inf, -np.inf, min_lim, min_lim, -np.inf), (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
        xdata_tuple, heatmaps = twoD_gauss_soma(chosen_fid, xy_avg, xyz_fid, marker_mean, gauss_bounds, cmaps, heatmap_bounds, xlen, normalise, plot_heatmap)
        marker_mean = {c:list(marker_mean[c]) for c in marker_mean.keys()}
        return (xdata_tuple, heatmaps, nodenums, shapes_o, marker_mean, concaveFileName)

    # if plot_individual_hulls:
    #     pl.show()

    return


def probability_per_node_zinc(io_files, xtup, nn, heatmaps, unloc_dist, nerve_position):

    nerve_title = io_files[1].split('\\')[-1].split('.')[0]
    node_probability = {c: 0 for c in nn}

    refinedFile_template = io_files[0]
    refinedFile_withProbability = io_files[1]

    # first read of input file: for the xyz node values for finding nodal probabilities
    node_xyz = {}
    marker_node_number = {}
    marker_xdx = {}
    all_node_num, xyz_all, dxyz_single, xyzGroups, element_list, xyz_marker, marker_names, marker_nodenum, marker_elemxi = \
        zinc_read_exf_file(file=refinedFile_template, raw_data = 0, derv_present=1, marker_present=1,  otherFieldNames = [], groupNames = [], mesh_dimension=3)
    for ia, num in enumerate(all_node_num):
        node_xyz.update({num:xyz_all[ia]})
    for im, marker in enumerate(marker_names):
        marker_node_number.update({marker: marker_nodenum[im]})
        marker_xdx.update({marker: xyz_marker[im]})

    # calculate nodal probability
    if heatmaps:
        gridxs = xtup[0][0]
        gridys = xtup[1][:,0]
        for fid in heatmaps.keys():
            for node in node_xyz:
                x0,y0,z0 = node_xyz[node]
                ix = np.where(gridxs == min(gridxs, key=lambda x: abs(x - x0)))[0][0]
                iy = np.where(gridys == min(gridys, key=lambda x: abs(x - y0)))[0][0]
                node_probability[node] += heatmaps[fid][iy][ix]

        # outputRegion = region.createRegion()
        probability_group_name = 'probability_' + nerve_title
    else:
        # if within locus determined by unloc_dist, probability = 1 else 0
        for node in node_xyz:
            x0, y0, z0 = node_xyz[node]
            th,r = cart2pol([x0,y0], nerve_position)
            node_probability[node] = 1 if r <= unloc_dist else 0
        probability_group_name = 'locus_' + nerve_title

    context = Context("Example")
    outputRegion = context.getDefaultRegion()
    outputRegion.readFile(refinedFile_template)
    fmOut = outputRegion.getFieldmodule()
    fmOut.beginChange()
    cache = fmOut.createFieldcache()
    coordinates = findOrCreateFieldCoordinates(fmOut)
    probability = findOrCreateFieldFiniteElement(fmOut, probability_group_name, 1)
    nodes = fmOut.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

    nodetemplate = nodes.createNodetemplate()
    nodetemplate.defineField(coordinates)
    nodetemplate.defineField(probability)
    nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
    nodetemplate.setValueNumberOfVersions(probability, -1, Node.VALUE_LABEL_VALUE, 1)

    for id, nodeIdentifier in enumerate(node_xyz.keys()):
        xyz_val = node_xyz[nodeIdentifier]
        node = nodes.findNodeByIdentifier(nodeIdentifier)
        cache.setNode(node)
        node.merge(nodetemplate)
        result1 = coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyz_val)
        result2 = probability.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, node_probability[nodeIdentifier])


    if False: # proper openCMISS way - incorrect right now.
        mesh = fmOut.findMeshByDimension(3)
        # mesh.destroyAllElements() # this messes up the markers?
        destroyElements = True
        basis = fmOut.createElementbasis(3, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)

        if True: # mabelle's way
            eft = mesh.createElementfieldtemplate(basis)
            elementtemplate = mesh.createElementtemplate()
            elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
            result = elementtemplate.defineField(coordinates, -1, eft)

        eftProb = mesh.createElementfieldtemplate(basis)
        elementtemplateProb = mesh.createElementtemplate()
        elementtemplateProb.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        # result = elementtemplateProb.defineField(coordinates, -1, eftProb)
        result = elementtemplateProb.defineField(probability, -1, eftProb)

        for eID, enids in enumerate(element_list):
            elementIdentifier = eID + 1 #+ 1000
            if destroyElements:
                mesh.destroyElement(mesh.findElementByIdentifier(elementIdentifier))
                element = mesh.createElement(elementIdentifier, elementtemplate)
                result = element.setNodesByIdentifier(eft, enids)
                element.merge(elementtemplateProb)
                result = element.setNodesByIdentifier(eftProb, enids)
            else:
                element = mesh.createElement(elementIdentifier, elementtemplateProb)
                if element.isValid():
                    result = element.setNodesByIdentifier(eftProb, enids)
                    element.merge(elementtemplateProb)

        if destroyElements: # remake markers - find their node numbers
            pass
        fmOut.endChange()
        fmOut.defineAllFaces()  # add 1D and 2D elements  # must define the mesh3D manually first
        outputRegion.writeFile(refinedFile_withProbability)

    else:  # HACK: append the probability preamble to elements.
        fmOut.endChange()
        outputRegion.writeFile(refinedFile_withProbability)

        lines = []
        pattern = False
        elementPatternLines = []
        found_mesh_3d = False
        with open(refinedFile_withProbability,'r') as r_in:
            for line in r_in:
                if '!#mesh mesh3d, dimension=3' in line:
                    found_mesh_3d = True
                if 'Fields' in line and found_mesh_3d:
                    line = '#Fields=2\n'
                if 'x.' in line and found_mesh_3d:
                    pattern = True
                if 'y.' in line and found_mesh_3d:
                    pattern = False
                if pattern:
                    eline = line
                    if 'x' in line:
                        eline = line.replace('x', '1')
                        elementPatternLines.append('2) %s, field, rectangular cartesian, real, #Components=1\n' %(probability_group_name))
                    elementPatternLines.append(eline)
                if 'Element: 1\n' in line and found_mesh_3d:
                    for eline in elementPatternLines:
                        lines.append(eline)
                    found_mesh_3d = False
                lines.append(line)
        with open (refinedFile_withProbability,'w') as r_out:
            for line in lines:
                r_out.write(line)

    comlines = []

    return (lines, refinedFile_withProbability, marker_elemxi, marker_xdx, comlines)


def refine_mesh_write_out(path, source_file, ic, heatmaps, pref, out_files, nodelist, unloc_dist, nerve_position):
    rf_path = path + 'refined\\'
    out_name = source_file.split('\\')[-1]
    refinedfile = rf_path + pref + out_name
    if ic==0:
        subdiv = 4
        context = Context("Example")
        region = context.getDefaultRegion()
        region.readFile(source_file)
        if not region.readFile(source_file):
            print('File not readable for refinement')

        outputRegion = region.createRegion()  # or context.createRegion()
        fmSource = region.getFieldmodule()
        sourceMesh = fmSource.findMeshByDimension(3)
        sourceNodes = fmSource.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodeIdentifier = fmSource.findNodesetByName('nodes').getSize() + 1

        fm = outputRegion.getFieldmodule()
        print('Refining ')
        outputRegion, nodeIdentifier = refine_mesh(region, outputRegion, subdiv)
        outputRegion.getFieldmodule().defineAllFaces()  # add 1D and 2D elements
        mesh = fm.findMeshByDimension(3)

        outputRegion.writeFile(refinedfile)

        nodes = fm.findNodesetByName('nodes')
        nodelist = []
        nodeIter = nodes.createNodeiterator()
        node = nodeIter.next()
        while node.isValid():
            nodeID = node.getIdentifier()
            nodelist.append(nodeID)
            node = nodeIter.next()

    # add probability field to refined mesh, using heatmaps
    out_files[0] = refinedfile
    _, refinedFile_withProbability, _,_, ctemp = probability_per_node_zinc(out_files, xtup, nodelist, heatmaps, unloc_dist, nerve_position)

    clines = []
    for line in ctemp:
        if 'surfaces' in line:
            line = "gfx modify g_element \"/\" surfaces domain_mesh2d coordinate coordinates face all tessellation default LOCAL select_on material default data probability spectrum default selected_material default_selected render_shaded;\n"
        clines.append(line)
    return clines, nodelist



if __name__ == "__main__":

    up_path = 'scaffoldfitter_output\\'
    path = up_path + 'processed_exf\\'
    com_folder = up_path + 'com_files\\'

    do_heatmap = True
    single_com = True
    testing = False
    plot_individual_hulls = False
    plot_concave_hull = False
    plot_heatmap = False
    plot_graphs = [plot_individual_hulls, plot_concave_hull,  plot_heatmap]
    heatmap_bounds = [[-0.3,1],[-0.3,0.3]]
    cols = ['yellow','red','cyan','magenta','orange','green','gold','silver']

    all_nerves = ['Inferior cardiac nerve', 'Ventral ansa subclavia', 'Dorsal ansa subclavia',
                  'Cervical spinal nerve 8',
                  'Thoracic spinal nerve 1', 'Thoracic spinal nerve 2', 'Thoracic spinal nerve 3',
                  'Thoracic sympathetic nerve trunk'] #, 'Thoracic ansa']
    all_soma = ['Soma_'+a for a in all_nerves]
    pref = 'TEST_' if testing else ''
    comlines = []

    # fitter_path = 'C:\\Users\\sfon036\\Google Drive\\SPARC_work\\codes\\mapclient_workflows\\11_stellate3arm\\'
    fitter_path = up_path + 'scaffoldfitter_geofit_folder\\'
    geofit_output_path = fitter_path + 'geofit_outputs\\'
    raw_path = fitter_path + 'exp_data\\'

    # read in all raw data files to calculate projections of soma
    fes = os.listdir(raw_path)
    fes = [f for f in fes if f.endswith('.exf')]
    suf = '_'+fes[0].split('_')[-1]
    raw_data_all = {c.split('_')[0]:{} for c in fes}
    projected_markers_all = {}
    for fe in fes:
        file_name = raw_path + fe
        raw_node_num, raw_xyz_all, _, xyzGroups, _, raw_xyz_marker, raw_marker_names, raw_marker_nodenum, _ = \
            zinc_read_exf_file(file=file_name, raw_data = 1, derv_present=0, marker_present=1, otherFieldNames = [], groupNames = [], mesh_dimension=0)
        newdict = {'nodes': raw_node_num, 'xyz':raw_xyz_all, 'marker':{'name': raw_marker_names, 'xyz': raw_xyz_marker, 'node': raw_marker_nodenum}}
        raw_data_all[fe.split('_')[0]].update(newdict)

    for sample in raw_data_all.keys():
        # SOMA - find xi coordinates in that given mesh
        points_single = zinc_find_ix_from_real_coordinates(geofit_output_path+sample+'.exf', raw_path+sample+suf)
        if points_single:
            for fid in points_single.keys():
                if 'soma' in fid.lower():
                    if fid not in projected_markers_all.keys():
                        projected_markers_all.update({fid:{}})
                        for key in points_single[fid].keys():
                            projected_markers_all[fid][key] = [points_single[fid][key]]
                    else:
                        for key in points_single[fid].keys():
                            projected_markers_all[fid][key].append(points_single[fid][key])


    # find connections between soma and the nerveorigin: from reading xml files
    # cx_dict = xml_soma_nerve_connections('..\\stellate_xml_scaffold\\xml_ex_files\\Finalized Stellate Dataset')
    cx_dict = xml_soma_nerve_connections('xml_ex\\Finalized Stellate Dataset')
    soma_cx = {c:[] for c in list(cx_dict.keys())}
    for soma_name in cx_dict.keys():
        for somaid in cx_dict[soma_name].keys():
            if cx_dict[soma_name][somaid]:
                nerve = cx_dict[soma_name][somaid][0]#.replace(' ','').lower()
                soma_cx[soma_name.replace('"','')].append(nerve)
    # count number of soma-icn connections for all nerves
    soma_nerve_cx = {c:0 for c in all_nerves if 'soma' not in c}
    for somakey in soma_cx:
        for nerve in soma_cx[somakey]:
            soma_nerve_cx[nerve] += 1

    # count number of cx
    num_soma_nerve_cx = {c:0 for c in all_nerves if 'soma' not in c}
    for somakey in soma_cx:
        for nerve in soma_cx[somakey]:
            num_soma_nerve_cx[nerve] += 1

    # add soma of unknown location
    nerve_dists = main_somanerve(get_scales=False)


    ###############################################
    # HEATMAP
    ###############################################

    meshFile = path + pref + 'geometric_average_mesh.exf'
    geoMeshFile = geofit_output_path + 'geofit_average_concave_hull_mesh.exf'

    # which markers to make a heatmap for?
    # chosen_fid = all_nerves.copy() + ['all']# + ['Soma_'+c for c in all_nerves]
    chosen_fid = list(projected_markers_all.keys())

    for ic, c in enumerate(chosen_fid+['all']):
        suf = '_' + c.replace(' ', '')
        if not do_heatmap:
            make_heatmap(do_heatmap, geofit_output_path, c, meshFile, geoMeshFile, heatmap_bounds, all_nerves, plot_graphs, [], [], [])
        else:
            if ic==0:
                xtup, heatmaps, nn, shapes, marker_mean, concaveMeshFile= make_heatmap(geofit_output_path, do_heatmap, c, meshFile, geoMeshFile, heatmap_bounds, all_nerves, plot_graphs, [], projected_markers_all, [])
            else:
                xtup, heatmaps, nn, shapes, marker_mean, concaveMeshFile = make_heatmap(geofit_output_path,do_heatmap, c, meshFile, geoMeshFile, heatmap_bounds, all_nerves, plot_graphs, shapes, projected_markers_all, marker_mean)

        # refine mesh 4x4x1
        out_files = [[],
                     path + pref+'refined_mesh_with_probability%s.exf' %(suf),
                     com_folder + pref+'view%s.com' %(suf)]
        out_files[1] = path + pref+'%s.exf' %(suf[1:])
        out_files[-1] = com_folder + pref+'heatmap%s.com' %(suf)

        if ic == 0:
            nodelist = []

        paramOutFile = path + 'nodal_param_mean_mesh.txt'
        write_parameter_set_string_values(geoMeshFile, paramOutFile)

        comline, nodelist = refine_mesh_write_out(path, geoMeshFile, ic, heatmaps, pref, out_files, nodelist,[],[])
        comlines.append(comline)


    # ###### CONNECTED NODES ######
    # # read in all.exf to modify it with the extra fields
    all_node_num, xyz_all, _, xyzGroups,_, xyz_marker, marker_names, marker_nodenum, _ = \
        zinc_read_exf_file(file=path+'all.exf', raw_data=0, derv_present=0, marker_present=1,  otherFieldNames = ['probability_all'],groupNames = [], mesh_dimension=3)
    centroid = [sum([x[i] for x in xyz_all])/len(xyz_all) for i in range(len(xyz_all[0]))]


    # repeat the refine_mesh routine for extra unknown location soma (unknown_soma_names)
    unknown_soma_names = ['Soma_' + sn for sn in nerve_dists.keys() if
                      'Soma_' + sn not in marker_mean.keys()]
    for sn in unknown_soma_names:
        nerve_origin_name = sn.split('_')[-1]
        suf = '_' + sn.replace(' ', '')
        out_files = [[], \
                     path + pref+'refined_mesh_with_probability%s.exf' %(suf), \
                     com_folder + pref+'view%s.com' %(suf)]
        out_files[1] = path + pref+'%s.exf' %(suf[1:])
        out_files[-1] = com_folder + pref+'heatmap%s.com' %(suf)
        comline, nodelist = refine_mesh_write_out(path, geoMeshFile, 1, [], pref, out_files, nodelist, nerve_dists[nerve_origin_name], xyz_marker[marker_names.index(nerve_origin_name)])



    allFile =path+'all.exf'
    context = Context('Example')
    outputRegion = context.getDefaultRegion()
    outputRegion.readFile(allFile)
    fmOut = outputRegion.getFieldmodule()
    fmOut.beginChange()
    cache = fmOut.createFieldcache()
    coordinates = findOrCreateFieldCoordinates(fmOut, 'marker_data_coordinates')
    # *** insert soma_centroid datapoints ***
    nodes = fmOut.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    markerName = findOrCreateFieldStoredString(fmOut, name='marker_data_name')
    somamarkerGroup = findOrCreateFieldGroup(fmOut, 'known_location_marker')
    somamarkerUnLocGroup = findOrCreateFieldGroup(fmOut, 'unknown_location_marker')
    markerPoints = findOrCreateFieldNodeGroup(somamarkerGroup, nodes).getNodesetGroup()
    markerUnLocPoints = findOrCreateFieldNodeGroup(somamarkerUnLocGroup, nodes).getNodesetGroup()
    datamarkerTemplate = markerPoints.createNodetemplate()
    datamarkerTemplate.defineField(coordinates)
    datamarkerTemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
    datamarkerTemplate.defineField(markerName)
    datamarkerUnLocTemplate = markerUnLocPoints.createNodetemplate()
    datamarkerUnLocTemplate.defineField(coordinates)
    datamarkerUnLocTemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
    datamarkerUnLocTemplate.defineField(markerName)

    nodeIdentifier = max(all_node_num[-1], marker_nodenum[-1])+1
    for key in marker_mean.keys():
        # soma
        node = markerPoints.createNode(nodeIdentifier, datamarkerTemplate)
        cache.setNode(node)
        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, marker_mean[key])
        markerName.assignString(cache, key)
        marker_nodenum.append(nodeIdentifier)
        marker_names.append(key)
        xyz_marker.append(marker_mean[key])
        nodeIdentifier += 1
        # nerve
        nerve = key.split('Soma_')[-1]
        node = markerPoints.createNode(nodeIdentifier, datamarkerTemplate)
        cache.setNode(node)
        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyz_marker[marker_names.index(nerve)])
        markerName.assignString(cache, nerve)
        nodeIdentifier += 1

    # add soma of unknown location
    # xscale, yscale = [sum([a[j] for a in all_file_scales])/len(all_file_scales) for j in range(2)]
    # fullScaleOffset = [1/xscale, 1/yscale, 1/xscale] # invert to get dimensionless, just like the rest of the points
    somaNerveOffsetOverride = [150]*3  #if need

    for nerve_name in num_soma_nerve_cx.keys():
        if num_soma_nerve_cx[nerve_name]:
            soma_name = 'Soma_' + nerve_name
            if soma_name not in marker_names:
                inerve = marker_names.index(nerve_name)
                xn = xyz_marker[inerve]
                vnorm = math.sqrt(sum([(centroid[i]-xn[i])**2 for i in range(3)]))
                CNunit = [(xn[i]-centroid[i])/vnorm for i in range(3)]
                try:
                    xyzUnLoc = [xn[i] - (CNunit[i]*nerve_dists[nerve_name]) for i in range(3)]
                except:
                    xyzUnLoc = [xn[i] - (CNunit[i]*somaNerveOffsetOverride[i]) for i in range(3)]
                    print('OVERRIDE soma position for: ' + soma_name)
                marker_names.append(soma_name)
                node = markerUnLocPoints.createNode(nodeIdentifier, datamarkerUnLocTemplate)
                cache.setNode(node)
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyzUnLoc)
                markerName.assignString(cache, soma_name)
                marker_nodenum.append(nodeIdentifier)
                xyz_marker.append(xyzUnLoc)
                nodeIdentifier += 1
                # nerve
                node = markerPoints.createNode(nodeIdentifier, datamarkerTemplate)
                cache.setNode(node)
                coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1,
                                              xyz_marker[marker_names.index(nerve_name)])
                markerName.assignString(cache, nerve_name)
                nodeIdentifier += 1

    fmOut.endChange()

    elementIdentifier = 1
    for nerve_name in num_soma_nerve_cx.keys():
        if num_soma_nerve_cx[nerve_name]:
            soma_name = 'Soma_' + nerve_name
            elnodes = []

            soma_name_ns = soma_name.replace(' ','')
            childRegion = outputRegion.createChild(soma_name_ns)
            fmCh = childRegion.getFieldmodule()
            fmCh.beginChange()
            cache = fmCh.createFieldcache()
            mesh1d = fmCh.findMeshByDimension(1)
            basis1d = fmCh.createElementbasis(1, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
            eft1d = mesh1d.createElementfieldtemplate(basis1d)
            soma_coordinates = findOrCreateFieldCoordinates(fmCh,soma_name_ns+'_coordinates')
            markerName = findOrCreateFieldStoredString(fmCh, name=soma_name_ns+'_name')
            markerPoints = fmCh.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

            somamarkertemplate = markerPoints.createNodetemplate()
            somamarkertemplate.defineField(soma_coordinates)
            somamarkertemplate.defineField(markerName)
            somamarkertemplate.setValueNumberOfVersions(soma_coordinates, -1, Node.VALUE_LABEL_VALUE, 1)

            soma_node_number = marker_nodenum[marker_names.index(soma_name)]

            nerve_origin_number = marker_nodenum[marker_names.index(nerve_name)]
            cxnames = [soma_name, nerve_name]

            nodeIdentifier = 1
            for ic, name in enumerate(cxnames):
                node = markerPoints.createNode(nodeIdentifier, somamarkertemplate)
                result = cache.setNode(node)
                result = soma_coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyz_marker[marker_names.index(name)])
                result = markerName.assignString(cache, name)
                elnodes.append(nodeIdentifier)
                nodeIdentifier += 1

            # elements
            elementtemplate = mesh1d.createElementtemplate()
            elementtemplate.setElementShapeType(Element.SHAPE_TYPE_LINE)
            result = elementtemplate.defineField(soma_coordinates, -1, eft1d)
            element = mesh1d.createElement(elementIdentifier, elementtemplate)
            result = element.setNodesByIdentifier(eft1d, elnodes)
            elementIdentifier += 1

            fmCh.endChange()

    outputRegion.writeFile(allFile)

    # make a giant com file to view all individual, refined heatmaps
    if single_com:
        source_file = com_folder + pref+'all_heatmaps_single_window.com'
        write_heatmap_com_file_single(source_file, chosen_fid, soma_nerve_cx, unknown_soma_names, cols)

