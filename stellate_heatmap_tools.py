import os
from scipy.spatial.distance import pdist, squareform
from numpy import nanmax, argmax, unravel_index
from math import *
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import xml.etree.ElementTree as ElTree
from opencmiss.zinc.context import Context
from opencmiss.zinc.node import Node
from opencmiss.utils.zinc.field import Field, findOrCreateFieldFiniteElement, findOrCreateFieldCoordinates, findOrCreateFieldGroup, \
    findOrCreateFieldNodeGroup, findOrCreateFieldStoredMeshLocation, findOrCreateFieldStoredString
from opencmiss.utils.zinc.finiteelement import getElementNodeIdentifiersBasisOrder
from scaffoldmaker.utils.meshrefinement import MeshRefinement


def get_raw_tag(element):
    element_tag = element.tag
    if '}' in element_tag:
        element_tag = element.tag.split('}', 1)[1]
    return element_tag


def coordinates_opencmiss_to_list(cache, nodes, coordinates, derv):
    xyzlist = []
    dxyzlist = []
    valid_nodes = []
    ccount = coordinates.getNumberOfComponents()

    # for n in nodelist:
    nodeIter = nodes.createNodeiterator()
    node = nodeIter.next()
    while node.isValid():
        nodeID = node.getIdentifier()
        cache.setNode(node)
        result, v1 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, ccount )
        if result == 1:
            xyzlist.append(v1)
            valid_nodes.append(nodeID)
        if derv:
            result, d1 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, ccount )
            result2, d2 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, ccount )
            result3, d3 = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, ccount )
            if result == 1:
                dxyzlist.append([[d1[i], d2[i]] for i in range(len(d1))])
        node = nodeIter.next()

    if not dxyzlist:
        dxyzlist = [[[0,0]]*3]*len(xyzlist)

    if derv:
        return valid_nodes, xyzlist, dxyzlist
    else:
        return valid_nodes, xyzlist, []


def zinc_read_exf_file(file, raw_data, derv_present, marker_present, otherFieldNames, groupNames, mesh_dimension):
    from_xml = True if raw_data == 2 else False
    context = Context("Example")
    region = context.getDefaultRegion()
    region.readFile(file)
    if not region.readFile(file):
        print('File not readable for zinc')
    fm = region.getFieldmodule()
    cache = fm.createFieldcache()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    coords_name = "data_coordinates" if raw_data == 1 else "coordinates"
    coordinates = findOrCreateFieldCoordinates(fm, coords_name)
    otherFields = [findOrCreateFieldFiniteElement(fm, n, 1, component_names=("1"), type_coordinate=False) for n in otherFieldNames]
    mesh = fm.findMeshByDimension(mesh_dimension)

    xyzGroups = []
    if groupNames:
        xyzGroups = {c:[] for c in groupNames}
        for subgroup in groupNames:
            group = fm.findFieldByName(subgroup).castGroup()
            nodeGroup = group.getFieldNodeGroup(nodes)
            if nodeGroup.isValid():
                gnodes = nodeGroup.getNodesetGroup()
                nodeIter = gnodes.createNodeiterator()
                node = nodeIter.next()
                groupSize = 0
                while node.isValid():
                    cache.setNode(node)
                    result, x = coordinates.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
                    xyzGroups[subgroup].append(x)
                    node = nodeIter.next()
                    groupSize += 1
            else:
                del xyzGroups[subgroup]

    all_node_num, xyz_all, dxyz_single = coordinates_opencmiss_to_list(cache, nodes, coordinates, derv_present)
    for oth in otherFields:
        _, ff, _ = coordinates_opencmiss_to_list(cache, nodes, oth, 0)

    element_list = []
    elementIter = mesh.createElementiterator()
    element = elementIter.next()
    while element.isValid():
        eft = element.getElementfieldtemplate(coordinates, -1)  # assumes all components same
        nodeIdentifiers = getElementNodeIdentifiersBasisOrder(element, eft)
        element_list.append(nodeIdentifiers)
        element = elementIter.next()

    if marker_present:
        # raw_data = True if not derv_present else raw_data
        nodes = fm.findNodesetByName('datapoints') if raw_data else fm.findNodesetByName('nodes')
        marker_names = []
        xyz_marker = []
        marker_nodenum = []
        marker_elemxi = {}
        marker_string = "marker_data" if raw_data == 1 else "marker"
        markerNamesField = fm.findFieldByName(marker_string+"_name")
        markerLocation = fm.findFieldByName(marker_string+"_location")
        hostCoordinates = fm.createFieldEmbedded(coordinates, markerLocation)
        if raw_data and not from_xml:
            coordinates = findOrCreateFieldCoordinates(fm, 'marker_'+coords_name)
        fieldcache = fm.createFieldcache()
        nodeIter = nodes.createNodeiterator()
        node = nodeIter.next()
        while node.isValid():
            nodeID = node.getIdentifier()
            fieldcache.setNode(node)
            markerName = markerNamesField.evaluateString(fieldcache)
            if markerName is not None:
                markerName = markerName.split('Origin of ')[-1].capitalize()
                if raw_data:
                    result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
                else:
                    result, x = hostCoordinates.evaluateReal(fieldcache, 3)
                marker_names.append(markerName)
                xyz_marker.append(x)
                marker_nodenum.append(node.getIdentifier())
                element, xi = markerLocation.evaluateMeshLocation(fieldcache, 3)
                if element.isValid():
                    marker_elemxi.update({markerName: {'elementID': element.getIdentifier(), 'xi': xi}})
            node = nodeIter.next()

        return all_node_num, xyz_all, dxyz_single, xyzGroups, element_list, xyz_marker, marker_names, marker_nodenum, marker_elemxi
    else:
        return all_node_num, xyz_all, dxyz_single, xyzGroups


def zinc_find_ix_from_real_coordinates(modelFile, dataFile):
    context = Context("Example")
    region = context.getDefaultRegion()
    # load model
    region.readFile(modelFile)
    if not region.readFile(modelFile):
        print('File not readable for zinc')
        return []
    region.readFile(dataFile)
    if not region.readFile(dataFile):
        print('File not readable for zinc')
        return []
    fm = region.getFieldmodule()
    cache = fm.createFieldcache()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    datapoints = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    dataNamesField = fm.findFieldByName("marker_data_name")
    coordinates = findOrCreateFieldCoordinates(fm, "coordinates")
    data_coordinates = findOrCreateFieldCoordinates(fm, "marker_data_coordinates")
    mesh = fm.findMeshByDimension(3)

    found_mesh_location = fm.createFieldFindMeshLocation(data_coordinates, coordinates, mesh)
    found_mesh_location.setSearchMode(found_mesh_location.SEARCH_MODE_NEAREST)
    xi_projected_data = {}
    nodeIter = datapoints.createNodeiterator()
    node = nodeIter.next()
    while node.isValid():
        cache.setNode(node)
        element, xi = found_mesh_location.evaluateMeshLocation(cache, 3)
        marker_name = dataNamesField.evaluateString(cache)
        if element.isValid():
            addProjection = {marker_name:{"elementID": element.getIdentifier(), "xi": xi}}
            xi_projected_data.update(addProjection)
        node = nodeIter.next()
    return xi_projected_data


def zinc_write_element_xi_marker_file(outFile, region, allMarkers, nodeIdentifier):
    fm = region.getFieldmodule()
    fm.beginChange()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    mesh = fm.findMeshByDimension(3)
    cache = fm.createFieldcache()
    markerGroup = findOrCreateFieldGroup(fm, "marker")
    markerName = findOrCreateFieldStoredString(fm, name="marker_name")
    markerLocation = findOrCreateFieldStoredMeshLocation(fm, mesh, name="marker_location")
    markerPoints = findOrCreateFieldNodeGroup(markerGroup, nodes).getNodesetGroup()
    markerTemplate = nodes.createNodetemplate()
    markerTemplate.defineField(markerName)
    markerTemplate.defineField(markerLocation)

    for key in allMarkers:
        xi = allMarkers[key]["xi"]
        addMarker = {"name": key+" projected", "xi": allMarkers[key]["xi"]}

        markerPoint = markerPoints.createNode(nodeIdentifier, markerTemplate)
        cache.setNode(markerPoint)
        markerName.assignString(cache, addMarker["name"])
        elementID = allMarkers[key]["elementID"]
        element = mesh.findElementByIdentifier(elementID)
        result = markerLocation.assignMeshLocation(cache, element, addMarker["xi"])
        nodeIdentifier += 1

    fm.endChange()
    region.writeFile(outFile)
    return


def zinc_write_element_xi_file(outFile, xyz, allMarkers):
    context = Context("Example")
    outputRegion = context.getDefaultRegion()
    fm = outputRegion.getFieldmodule()
    fm.beginChange()
    cache = fm.createFieldcache()
    coordinates = findOrCreateFieldCoordinates(fm)

    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = nodes.createNodetemplate()
    nodetemplate.defineField(coordinates)
    nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
    markerNodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    markerGroup = findOrCreateFieldGroup(fm, "marker")
    markerName = findOrCreateFieldStoredString(fm, name="marker_name")
    markerPoints = findOrCreateFieldNodeGroup(markerGroup, markerNodes).getNodesetGroup()
    markerTemplate = markerPoints.createNodetemplate()
    markerTemplate.defineField(markerName)
    markerTemplate.defineField(coordinates)

    nodeIdentifier = 1
    for ix in xyz:
        node = nodes.createNode(nodeIdentifier, nodetemplate)
        cache.setNode(node)
        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, ix)
        nodeIdentifier += 1

    nodeIdentifier = 1
    for key in allMarkers:
        addMarker = {"name": key, "xyz": allMarkers[key]}
        node = markerPoints.createNode(nodeIdentifier, markerTemplate)
        cache.setNode(node)
        coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, addMarker['xyz'])
        markerName.assignString(cache, addMarker["name"])
        nodeIdentifier += 1

    fm.endChange()
    outputRegion.writeFile(outFile)


def write_parameter_set_string_values(infile, outfile):
    all_node_num, xyz_all, dxyz, _, element_list, xyz_marker, marker_names, marker_nodenum, marker_elemxi = zinc_read_exf_file(infile, 0, 1, 1, [], [], 3)
    with open(outfile,'w') as wo:
        wo.write('[ ')
        for i, x in enumerate(xyz_all):
            # wo.write(str(x) + ', ' + str([d[0] for d in dxyz[i]]) + ', ' + str([d[1] for d in dxyz[i]]) + ' ],\n')
            if i < all_node_num[-1]-1:
                wo.write('[ [%0.4g, %0.4g, %0.4g], ' %(x[0], x[1], x[2]) + \
                         '[%0.4g, %0.4g, %0.4g], ' %(dxyz[i][0][0], dxyz[i][1][0], dxyz[i][2][0]) + \
                         '[%0.4g, %0.4g, %0.4g] ], ' %(dxyz[i][0][1], dxyz[i][1][1], dxyz[i][2][1]) + '\n')
            else:
                wo.write('[ [%0.4g, %0.4g, %0.4g], ' % (x[0], x[1], x[2]) + \
                         '[%0.4g, %0.4g, %0.4g], ' % (dxyz[i][0][0], dxyz[i][1][0], dxyz[i][2][0]) + \
                         '[%0.4g, %0.4g, %0.4g] ] ] ' % (dxyz[i][0][1], dxyz[i][1][1], dxyz[i][2][1]) + '\n')
    return


def xml_soma_nerve_connections(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        files = [f for f in files if ('.ex' not in f and '.xml' in f)]
    else: # single xml file
        files = [path]
    soma_dict = {f: {} for f in files}
    soma_name_list = []
    for f in files:
        somaIDlist = []
        found_soma = False
        file_name = path+'\\'+f if os.path.isdir(path) else path
        with open(file_name,'r') as f_in:
            tree = ElTree.parse(file_name)
            root = tree.getroot()
            for tree_root in root:
                raw_tag = get_raw_tag(tree_root)
                if raw_tag == "contour":
                    cname = tree_root.get("name")
                    if 'soma' in cname.lower():
                        if cname not in soma_name_list:
                            soma_name_list.append(cname)
                        for property_set in tree_root.iter():
                            if property_set.get("name") == "Set":
                                for child in property_set.iter():
                                    if "}" in child.tag:
                                        tag = child.tag.split('}')
                                        somaID = "".join(child.itertext())
                                        if len(somaID) > 3 and 'x' not in somaID:
                                        # if ('10x' not in tag) and ('40x' not in tag) and ('63x' not in tag):
                                            newdict = {cname: {somaID: []}}
                                            if cname not in list(soma_dict[f].keys()):
                                                soma_dict[f].update(newdict)
                                            elif somaID not in soma_dict[f][cname] and 'All Cells' not in somaID:
                                                soma_dict[f].update(newdict)
                if raw_tag == "tree":
                    soma_name = []
                    found_innerve = False
                    found_soma_name = False
                    for property_set in tree_root.iter():
                        if property_set.get("name") == "Set":
                            for child in property_set.iter():
                                if "}" in child.tag:
                                    tag = child.tag.split('}')
                                    tline = "".join(child.itertext())
                                    if "s" in tag:
                                        if 'innervates' in tline:
                                            innerve = tline.split('innervates: ')[-1].capitalize()
                                            if 'ramus of first' in innerve:
                                                innerve = 'Thoracic spinal nerve 1'
                                            found_innerve = True
                                        else:# tline in list(soma_dict[f].keys()): # assume it is hte soma id
                                            try:
                                                if 'All Cells' not in tline:
                                                    somaID = tline
                                                    soma_name = [n for n in soma_dict[f].keys() if tline in soma_dict[f][n]][0]
                                                    found_soma_name = True
                                            except:
                                                j = 10
                                        if found_soma_name and found_innerve:
                                            # if 'All Cells' not in somaID and 'unknown' not in innerve.lower() and soma_name:
                                            # if 'unknown' not in innerve.lower() and soma_name:
                                            if 'unknown' not in innerve.lower() and soma_name.lower():
                                                if innerve not in soma_dict[f][soma_name][somaID]:
                                                    soma_dict[f][soma_name][somaID].append(innerve)
                                                    found_innerve = False
                                                    found_soma_name = False
    # join all samples' results in one dict - join Soma 1 to Soma 1
    cx_dict = {c:{} for c in soma_name_list}
    for f in soma_dict.keys():
        for nsoma in soma_dict[f].keys():
            cx_dict[nsoma].update(soma_dict[f][nsoma])
    return cx_dict


def cart2pol(p, centroid):

    x = p[0] - centroid[0]
    y = p[1] - centroid[1]

    return ( np.arctan2(y, x),  np.sqrt(x**2 + y**2))


def matrix_multiply_2D(x, theta):
    x_rot = x[0]*cos(theta) + x[1]*sin(theta)
    y_rot = -x[0]*sin(theta) + x[1]*cos(theta)
    if len(x) > 4:
        return [x_rot, y_rot, x[2], x[3], x[4]]
    if len(x) > 3:
        return [x_rot, y_rot, x[2], x[3]]
    elif len(x) > 2:
        return [x_rot, y_rot, x[2]]
    else:
        return [x_rot, y_rot]

def rotate_points(xyz, theta, fid):
    # rotate points according to axis defined by vector [am, bm]
    # [am bm] become the new x axis [0 1]

    if isinstance(xyz, dict):
        xyz = [xyz[key] for key in xyz.keys()]

    if fid: # rotate by fids (DA and TST sit on x=0)
        pDA = fid[0]
        pTST = fid[1]
        theta = atan((pTST[1] - pDA[1]) / (pTST[0] - pDA[0]))

    else:
        if not theta:
            D = pdist(xyz)
            D = squareform(D)
            N, [a, b] = nanmax(D), unravel_index(argmax(D), D.shape)
            am = xyz[a]
            bm = xyz[b]
            theta = atan((bm[1] - am[1]) / (bm[0] - am[0]))

    xyz_rot = [matrix_multiply_2D(p, theta) for p in xyz]

    return (xyz_rot, theta)


def parse_xml_find_soma_ID(xml_file):

    soma_dict = {}
    soma_name_list = []
    somaIDlist = []
    found_soma = False
    with open(xml_file,'r') as f_in:
        tree = ElTree.parse(xml_file)
        root = tree.getroot()
        for tree_root in root:
            raw_tag = get_raw_tag(tree_root)
            if raw_tag == "contour":
                cname = tree_root.get("name")
                if 'oma' in cname:
                    if cname not in soma_name_list:
                        soma_name_list.append(cname)
                    for property_set in tree_root.iter():
                        if property_set.get("name") == "Set":
                            for child in property_set.iter():
                                if "}" in child.tag:
                                    tag = child.tag.split('}')
                                    somaID = "".join(child.itertext())
                                    if len(somaID) > 3 and 'x' not in somaID:
                                        newdict = {cname: {somaID: []}}
                                        if cname not in list(soma_dict.keys()):
                                            soma_dict.update(newdict)
                                        elif somaID not in soma_dict[cname] and 'All Cells' not in somaID:
                                            soma_dict.update(newdict)
            if raw_tag == "tree":
                soma_name = []
                found_innerve = False
                found_soma_name = False
                for property_set in tree_root.iter():
                    if property_set.get("name") == "Set":
                        for child in property_set.iter():
                            if "}" in child.tag:
                                tag = child.tag.split('}')
                                tline = "".join(child.itertext())
                                if "s" in tag:
                                    if 'innervates' in tline:
                                        innerve = tline.split('innervates: ')[-1].capitalize()
                                        if 'ramus of first thoracic nerve' in innerve:
                                            innerve = 'Thoracic spinal nerve 1'
                                        found_innerve = True
                                    else:  # assume it is hte soma id
                                        try:
                                            if 'All Cells' not in tline:
                                                somaID = tline
                                                soma_name = [n for n in soma_dict.keys() if tline in soma_dict[n]][0]
                                                found_soma_name = True
                                        except:
                                            j = 10
                                    if found_soma_name and found_innerve:
                                        # if 'All Cells' not in somaID and 'unknown' not in innerve.lower() and soma_name:
                                        # if 'unknown' not in innerve.lower() and soma_name:
                                        if 'unknown' not in innerve and soma_name:
                                            if innerve not in soma_dict[soma_name][somaID]:
                                                soma_dict[soma_name][somaID].append(innerve)
                                                found_innerve = False
                                                found_soma_name = False
    return soma_dict


def find_closest_end(xp, yp, zp, target):
    norm = 1e6
    plen = len(xp)
    end_kept = []
    for i in range(plen):
        xyz = [xp[i], yp[i], zp[i]]
        vdf = [target[i] - xyz[i] for i in range(3)]
        norm_raw = LA.norm(vdf)
        if norm_raw < norm:
            end_kept = xyz
            norm = norm_raw
    return end_kept


def find_closest_value_polar(rlist, thlist, zlayers,target, num_layers, N, reltol):
    # perform in polar coords

    keep_ind = []
    count = 0
    layer_vals = list(zlayers.keys())
    layer_kept = []

    for w in range(num_layers):
        istart = count
        iend = count + zlayers[layer_vals[w]]
        keep = [tt for tt in range(istart,iend) if abs((thlist[tt]-target[1])/pi)<reltol[0] and abs((rlist[tt]-target[0])/target[0])<reltol[1]]
        rsemi = [rlist[i] for i in keep]
        try:
            keep_ind.append(rlist.index(min(rsemi, key=lambda x: abs(x - target[0]))))
            layer_kept.append(w)
        except:
            keep_ind.append(88888)
        count += zlayers[layer_vals[w]]

    r_all = []
    th_all = []
    for j in keep_ind:
        try:
            r_all.append(rlist[j])
            th_all.append(thlist[j])
        except:
            r_all = 88888 # dummies
            th_all = 88888

    return (keep_ind, r_all, th_all, layer_kept)


# xyz_stRot is the new stellate points. xyz_smRot_2D is the new soma points.
def mirror_points(x, y, z, centroid, direction):
    if direction == 'x':
        xnew = [(2 * centroid[0]) - ix for ix in x]
        xyz_new = [[xnew[i], y[i], z[i]] for i in range(len(x))]
    elif direction == 'y':
        ynew = [(2 * centroid[0]) - ix for ix in y]
        xyz_new = [[x[i], ynew[i], z[i]] for i in range(len(x))]
    return(xyz_new)


def flip_system(x, y, z, xsm, ysm, zsm, xy_fid, xyz_t3Rot, centroid, num_soma, fdict, xyz_n, sample, override):
    # base the flipping on the order of the points, so future rotation/translation transforms do not require another flip.
    # flip is only in the y direction (after angle offset to make th_TST=0)
    # input OVERRIDE flips based on the boolean (calculated outside this function)
    if sample:
        xyz_sm = []
        newsm = [[]] * num_soma

    # find angles: ICN is between DA (max angle) and TST (min angle)
    # beware negative angles
    x_DA = xy_fid[0]
    x_ICN = xy_fid[1]
    x_TST = xy_fid[2]

    th_DA, _ = cart2pol(x_DA, centroid)
    th_ICN, _ = cart2pol(x_ICN, centroid)
    th_TST, _ = cart2pol(x_TST, centroid)
    angle_offset = th_TST
    th_DA -= angle_offset
    th_ICN -= angle_offset
    th_TST -= angle_offset

    # make it all positive.
    th_DA = ( ((2*pi) + th_DA)*(th_DA<1) ) + ( (th_DA)*(th_DA>1) )
    th_ICN = ( ((2*pi) + th_ICN)*(th_ICN<1) ) + ( (th_ICN)*(th_ICN>1) )

    order = False
    if th_ICN < th_DA and th_ICN > th_TST:
        order = True

    flip = False
    if not order or override:
        print('FLIPPED')
        flip = True
        ind = 1

    xt3 = [k[0] for k in xyz_t3Rot]
    yt3 = [k[1] for k in xyz_t3Rot]
    zt3 = [k[2] for k in xyz_t3Rot]

    if flip:
        xyz_st = mirror_points(x, y, z, centroid, 'y')
        new = [x[ind] for x in xyz_st]

        if sample:
            xyz_t3 = mirror_points(xt3, yt3, zt3, centroid, 'y')
            newt3 = [x[ind] for x in xyz_t3]

        centroid = [np.mean(x), np.mean(y), np.mean(z)]

        if sample:
            for j in range(num_soma):
                out = mirror_points(xsm[j], ysm[j], zsm[j], centroid, 'y')
                out = [[row + [j + 1]][0] for row in out] # add soma ID
                xyz_sm.extend( out)
                newsm[j] = [x[ind] for x in out]
            for key in xyz_n:
                if xyz_n[key]:
                    xyz_n[key] = mirror_points([xyz_n[key][0]], [xyz_n[key][1]], [xyz_n[key][2]], centroid, 'y')[0]
        x_ICN = mirror_points([x_ICN[0]], [x_ICN[1]],[x_ICN[2]], centroid, 'y')
        x_ICN = x_ICN[0]

        y = new
        if sample:
            ysm = newsm
            yt3 = newt3
    else:
        if sample:
            xyz_st = [[x[i], y[i], z[i]] for i in range(len(x))]
            for i in range(num_soma):
                xyz_sm.extend( [xsm[i][k], ysm[i][k], zsm[i][k], i + 1] for k in range(len(xsm[i])) )

    xyz_t3out = [[xt3[i], yt3[i], zt3[i]] for i in range(len(xt3))]

    # Also return status of flip
    if sample:
        return (x, y, z, xsm, ysm, zsm, xyz_st, xyz_sm, xyz_t3out, fdict,xyz_n, x_ICN, flip)
    else:
        return (x, y, z, flip)


def translate_system(x,y,z,pzero):
    if isinstance(x[0], list):
        y = [ix[1] for ix in x]
        z = [ix[2] for ix in x]
        x = [ix[0] for ix in x]
    x = [val-pzero[0] for val in x]
    y = [val-pzero[1] for val in y]

    return (x, y, z)


def extrude_3D_in_1D(xy, zlen):
    num_nodes = len(xy)
    xy2 = np.array(xy.copy())
    xy = np.array(xy)
    xy2[:,2] = zlen
    xyz = list(xy) + list(xy2)
    return xyz


def write_heatmap_com_file_single(source_file, all_nerve_cx, soma_nerve_cx, unknown_soma_cx, cols):
    mult = 3
    with open(source_file, 'w') as w:
        w.write('gfx mod spectrum default linear range 0 0.2 extend_above extend_below rainbow colour_range 1 0 ambient diffuse component 1\n')
        w.write('gfx create spectrum locus_colour\ngfx mod spectrum locus_colour linear range 0 1 extend_above extend_below white_to_blue colour_range 1 0 ambient diffuse component 1\n\n')
        lower_ns = True if ' ' not in all_nerve_cx[0] and all_nerve_cx[0].lower() else False
        for ij, c in enumerate(all_nerve_cx):
            if lower_ns:
                nerve = c if 'soma' not in c else 'soma' + c.split('soma_')[-1]
            else:
                nerve = c.capitalize() if 'Soma' not in c else 'Soma_' + (c.lower().split('soma_')[-1].capitalize())
            nerve = nerve.replace(' ','')
            if ij == 0:
                w.write('gfx read elements "..\processed_exf\%s.exf"\n' % (nerve))
                w.write('gfx modify g_element "/" lines domain_mesh1d coordinate coordinates face all tessellation default LOCAL line line_base_size 0 select_on material grey50 selected_material default_selected render_shaded;\n')
            else:
                w.write('gfx read elements "%s.exf"\n' % (nerve))
            w.write('gfx modify g_element "/" surfaces domain_mesh2d as surf_%s coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_%s spectrum default selected_material default_selected render_shaded;\n\n' % (nerve, nerve))
        for c2 in unknown_soma_cx:
            cxname = c2.replace(' ','')
            w.write('gfx read elements "%s.exf"\ngfx modify g_element "/" surfaces domain_mesh2d as surf_%s coordinate coordinates face all tessellation default LOCAL select_on invisible material default data locus_%s spectrum locus_colour selected_material default_selected render_shaded;\n\n' %(cxname, cxname, cxname))
        w.write('gfx read elements "all.exf"\n\n')
        w.write('gfx define field marker_coordinates embedded element_xi marker_location field coordinates\n\n')
        ii = 0
        for nerve in soma_nerve_cx:
            # if there is a non-zero connection:
            if soma_nerve_cx[nerve]:
                cx = 'Soma_' + nerve.replace(' ','')
                lw = mult * soma_nerve_cx[nerve]
                w.write('gfx modify g_element /%s/ general clear;\n' % (cx))
                w.write('gfx modify g_element /%s/ lines domain_mesh1d coordinate %s_coordinates face all tessellation default LOCAL line_width %d line line_base_size 0 select_on material %s selected_material default_selected render_shaded;\n\n' % (cx, cx, lw, cols[ii]))
                ii += 1
        w.write('gfx modify g_element "/" points domain_nodes subgroup known_location_marker coordinate marker_data_coordinates tessellation default_points LOCAL glyph sphere size "50*50*50" offset 0,0,0 font default label marker_data_name label_offset 1,0,0 select_on material magenta selected_material default_selected render_shaded\n\n')
        w.write('gfx modify g_element "/" points domain_nodes subgroup unknown_location_marker coordinate marker_data_coordinates tessellation default_points LOCAL glyph sphere size "50*50*50" offset 0,0,0 font default label marker_data_name label_offset 1,0,0 select_on material white selected_material default_selected render_shaded\n\n')
        w.write('gfx modify g_element "/" points domain_nodes subgroup marker coordinate marker_coordinates tessellation default_points LOCAL glyph sphere size "50*50*50" offset 0,0,0 font default label marker_name select_on invisible material yellow selected_material default_selected render_shaded\n\n')
        w.write('gfx cre wind;\ngfx mod win 1 background colour 0 0 0;\ngfx edit scene;\n\n')
    return


def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, offset):
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (1/(2*sigma_x**2))
    c = 1/(2*sigma_y**2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + c * ((y - yo) ** 2)))
    return(g.ravel())


def create_colormap(cname):
    ncolors = 256
    color_array = plt.get_cmap(cname)(range(ncolors))
    # change alpha values
    color_array[:, -1] = np.linspace(0, 1, ncolors)
    # create a colormap object and register to matplotlib
    cname_new = cname+'_trans'
    map_object = LinearSegmentedColormap.from_list(name=cname_new, colors=color_array)
    plt.register_cmap(cmap=map_object)
    return cname_new


def refine_mesh(region, outputRegion, dv):
    meshRefinement = MeshRefinement(region, outputRegion)
    meshRefinement.refineAllElementsCubeStandard3d(dv,dv,1)
    fm = outputRegion.getFieldmodule()
    nodeIdentifier = fm.findNodesetByName('nodes').getSize()

    return outputRegion, nodeIdentifier+1
