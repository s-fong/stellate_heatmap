# 28 Jan 2020
# with input ex file (output of mbfxml2ex.py), make another exf file with a marker group to be used in scaffoldfitter
# 3 Feb
# with fiducials from 10X_fiducials_unprocessed.txt - instead of using nerve coordinate there, find the closest point on the stellate points.
# Because marker points must exist on the data points.
# 14 Feb parse through stellate body nodes between selected fiducials of every contour layer, and add to marker group
# 16 Sep modify marker and group names to suit starter meshes generated from scaffoldmaker (not lefttopbottom, but stellate face 1-2 etc)

import os
import shutil
import numpy as np
from math import *
from numpy import nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from stellate_heatmap_tools import cart2pol, rotate_points, find_closest_end, find_closest_value_polar, flip_system, parse_xml_find_soma_ID, zinc_read_exf_file
from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field
from opencmiss.zinc.node import Node
from opencmiss.utils.zinc.field import findOrCreateFieldCoordinates, findOrCreateFieldGroup, \
    findOrCreateFieldStoredString, findOrCreateFieldNodeGroup


def lefttopbottom_by_fiducials(x_arr, y_arr, z_arr, xyz_n, algorithm, sample, plot):
    # there are several layers - don't know how many.
    # in one layer, there may not be any nodes lying close to a fiducial point.
    # compare polar coordinates in either direction - preserve order of input nodes
    # node numbering starts at 1

    centroid = [np.mean(x_arr), np.mean(y_arr)]
    N = len(x_arr)
    num_layers = len(set(z_arr))

    # establish direction of node ordering
    clockwise = False
    theta1, _ = cart2pol([x_arr[0], y_arr[0]], centroid)
    theta2, _ = cart2pol([x_arr[1], y_arr[1]], centroid)
    if theta1 > theta2:
        clockwise = True

    p_DA = xyz_n['Dorsal ansa subclavia']
    if 'Inferior cardiac nerve' in list(xyz_n.keys()):
        p_VA = xyz_n['Inferior cardiac nerve']
        if not p_VA:
            p_VA = xyz_n['Ventral ansa subclavia']
    else:
        p_VA = xyz_n['Ventral ansa subclavia']
    p_TST = xyz_n['Thoracic sympathetic nerve trunk']

    th_TST, r_TST = cart2pol(p_TST, centroid)
    th_DA, r_DA = cart2pol(p_DA, centroid)
    th_VA, r_VA = cart2pol(p_VA, centroid)
    # angle_offset = min(abs(th_TST), abs(th_VA), abs(th_DA))
    angle_offset = th_TST

    th_DA -= angle_offset
    th_VA -= angle_offset
    th_TST -= angle_offset  # this is the zero
    th_DA -= 2 * pi if th_DA > pi else 0
    th_VA -= 2 * pi if th_VA > pi else 0
    th_TST -= 2 * pi if th_TST > pi else 0
    #
    ngroup = {'left': [], 'top': [], 'bottom': []}
    n_order = []

    if algorithm == 'by_order':

        seen = set()
        zvals = [x for x in z_arr if x not in seen and not seen.add(x)]
        zlayers = {zv: 0 for zv in zvals}
        zlayers_nolabel = []
        # zval_cumul = {zv:[0,0] for zv in set(z_arr)}
        count = 0
        for iz, vz in enumerate(z_arr):
            zlayers[vz] += 1
        for iz, key in enumerate(zlayers):
            zlayers_nolabel.append(zlayers[key])
        znodes = [[0] * 2 for _ in range(num_layers)]
        znodes[0][1] = zlayers_nolabel[0] - 1
        for iz in range(1, num_layers):
            znodes[iz][0] = znodes[iz - 1][1] + 1
            znodes[iz][1] += znodes[iz - 1][1] + zlayers_nolabel[iz]

        r_arr = [0] * N
        th_arr = [0] * N
        for m in range(N):
            [th_arr[m], r_arr[m]] = cart2pol([x_arr[m], y_arr[m]], centroid)
            th_arr[m] -= angle_offset
            th_arr[m] -= 2 * pi if th_arr[m] > pi else 0

        th_arr_og = th_arr.copy()
        r_arr_og = r_arr.copy()

        reltol = [0.05, 0.25]
        # reltol = [5, 5]

        [id_DA_a, rr_DA, th1, ikeep1] = find_closest_value_polar(r_arr, th_arr, zlayers, [r_DA, th_DA], num_layers, N, reltol)
        [id_VA_a, rr_VA, th2, ikeep2] = find_closest_value_polar(r_arr, th_arr, zlayers, [r_VA, th_VA], num_layers, N, reltol)
        [id_TST_a, rr_TST, th3, ikeep3] = find_closest_value_polar(r_arr, th_arr, zlayers, [r_TST, th_TST], num_layers, N, reltol)
        if id_DA_a[0] == id_DA_a[1]:
            id_DA_a[1] = id_DA_a[0] + int(len(th_arr) / 2)
            id_VA_a[1] = id_VA_a[0] + int(len(th_arr) / 2)
            id_TST_a[1] = id_TST_a[0] + int(len(th_arr) / 2)
        # ikeep_all = ikeep1 + ikeep2 + ikeep3
        # ikeep_all = list(set(ikeep_all))
        ikeep_all = [ik for ik in ikeep1 if ik in ikeep2 and ik in ikeep3]
        ikeep_all.sort(reverse=False)

        # deal With DISCONTINUITY and nodes that do not encompass whole body well
        # do not use polar coordinates.
        # if present, discard layer.
        rem_index = [0] * num_layers
        for i in range(num_layers):
            istart = znodes[i][0]
            iend = znodes[i][1]

            for node in range(istart, iend - 1):
                dist = sqrt(((x_arr[node + 1] - x_arr[node]) ** 2) + ((y_arr[node + 1] - y_arr[node]) ** 2))
                if dist > 1000:
                    rem_index[i] = 1

            if (max(x_arr) - max(x_arr[istart:iend]) > 2000) or (min(x_arr) - min(x_arr[istart:iend]) < -2000):
                rem_index[i] = 1

        if rem_index:
            ir = 0
            count = 0
            while ir < len(ikeep_all):
                id = rem_index[ir]
                if id:
                    try:
                        delme = ikeep_all.index(count)
                        del ikeep_all[delme]
                        del rem_index[ir]
                        count += 1
                    except:
                        ir += 1
                else:
                    ir += 1

        # TOP: TST to VA
        if len(ikeep_all) > 0:
            for ij, k in enumerate(ikeep_all):
                id_TST = id_TST_a[k] + 1
                id_VA = id_VA_a[k] + 1
                id_DA = id_DA_a[k] + 1
                istart = znodes[k][0] + 1
                iend = znodes[k][1] + 1

                if ij > 0:
                    prev_clockwise = clockwise

                clockwise = False
                if (id_TST > id_VA and id_VA > id_DA) or (id_VA > id_DA and id_DA > id_TST) or (
                        id_DA > id_TST and id_TST > id_VA):
                    clockwise = True

                # ngroupsingle = {'left': [], 'top': [], 'bottom': []}
                if clockwise:
                    # print('    layer: clockwise')
                    ngroup['bottom'].extend(list(range(id_TST, iend)) + list(range(istart, id_DA)) if (id_TST > id_VA and id_TST > id_DA) else list(range(id_TST, id_DA)))
                    ngroup['left'].extend(list(range(id_DA, iend)) + list(range(istart, id_VA)) if (id_DA > id_VA and id_DA > id_TST) else list(range(id_DA, id_VA)))
                    ngroup['top'].extend(list(range(id_VA, iend)) + list(range(istart, id_TST)) if (id_VA > id_TST and id_VA > id_DA) else list(range(id_VA, id_TST)))
                else:
                    # print('    layer: ANTIclock')
                    ngroup['bottom'].extend(list(range(id_DA, iend)) + list(range(istart, id_TST)) if (id_DA > id_VA and id_DA > id_TST) else list(range(id_DA, id_TST)))
                    ngroup['left'].extend(list(range(id_VA, iend)) + list(range(istart, id_DA)) if (id_VA > id_TST and id_VA > id_DA) else list(range(id_VA, id_DA)))
                    ngroup['top'].extend(list(range(id_TST, iend)) + list(range(istart, id_VA)) if (id_TST > id_VA and id_TST > id_DA) else list(range(id_TST, id_VA)))

            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(121, polar=True)
                rtest = []
                thtest = []
                ax.plot(th_arr, r_arr, '.', color='lightgrey')
                substep = 1
                th_sub = [th_arr[q] for q in range(istart, iend, substep)]
                r_sub = [r_arr[q] for q in range(istart, iend, substep)]
                ax.plot(th_sub, r_sub, 'o-')
                for q in range(0, len(r_sub), 20):
                    ax.text(th_sub[q], r_sub[q], str(q), fontweight='bold')
                ax.set_title(sample + '\n' + 'clockwise = ' + str(clockwise))
                ax.grid(False)

                # fig = plt.figure()
                ax = fig.add_subplot(122, polar=True)
                ax.plot(th_arr, r_arr, '.', color='lightgrey')
                thleft = []
                rleft = []
                thtop = []
                rtop = []
                thbottom = []
                rbottom = []
                id_TST -= 1
                id_DA -= 1
                id_VA -= 1
                for i in ngroup['left']:
                    thleft.append(th_arr[i])
                    rleft.append(r_arr[i])
                for i in ngroup['top']:
                    thtop.append(th_arr[i])
                    rtop.append(r_arr[i])
                for i in ngroup['bottom']:
                    thbottom.append(th_arr[i])
                    rbottom.append(r_arr[i])
                ax.plot(thleft, rleft, '.', color='magenta')
                ax.plot(thtop, rtop, '.', color='cyan')
                ax.plot(thbottom, rbottom, '.', color='yellow')

                ax.plot(th_arr[id_TST], r_arr[id_TST], 'bx', markersize=15)
                ax.text(th_arr[id_TST], r_arr[id_TST], 'TST ' + str(id_TST), fontsize=15, color='b')
                ax.plot(th_arr[id_DA], r_arr[id_DA], 'rx', markersize=15)
                ax.text(th_arr[id_DA], r_arr[id_DA], 'DA ' + str(id_DA), fontsize=15, color='r')
                ax.plot(th_arr[id_VA], r_arr[id_VA], 'mx', markersize=15)
                ax.text(th_arr[id_VA], r_arr[id_VA], 'VA ' + str(id_VA), fontsize=15, color='m')
                plt.show()

        print('\nRELTOL = ' + str(reltol[0]))

    else:
        print('NO ALGORITHM FOR GROUPING POINTS')

    nleft = ngroup['left']
    ntop = ngroup['top']
    nbottom = ngroup['bottom']
    return (nleft, ntop, nbottom)


# ****************************************************************************************************************
#                                               MAIN
# *****************************************************************************************************************
def main_writerotatefile(data_path):
    plot = False
    concave_hull = True if 'concave_hull' in data_path else False
    # fiducial points are in the new xml file with separate marker field
    fids_in_xml = False if concave_hull else True

    files = os.listdir(data_path)
    files = [f for f in files if f.endswith('.ex')]

    max_soma = 5
    all_nerves = ['Inferior cardiac nerve', 'Ventral ansa subclavia', 'Dorsal ansa subclavia', 'Cervical spinal nerve 8', 'Thoracic spinal nerve 1', 'Thoracic spinal nerve 2', 'Thoracic spinal nerve 3', 'Thoracic sympathetic nerve trunk', 'Thoracic ansa']
    bodyname = ['Cervicalganglion', 'stellate']

    fdict = {sample: {} for sample in files}

    write_this = False
    for sample in files:
        print(sample)
        lr_groups = True
        output_suffix = '_LR.exf'
        short_name = sample.split('_')[0] if not concave_hull else sample
        fmag = sample.split('_')[1].upper()
        full_body = True
        xyz_sm = []
        txt = []
        in_file = data_path + sample
        if not os.path.exists(data_path + 'exf_output\\'):
            os.mkdir(data_path + 'exf_output\\')
        mapclient_workflow_path = 'C:\\Users\\sfon036\\Google Drive\\SPARC_work\\codes\\mapclient_workflows\\11_stellate3arm\\exp_data\\'

        with open(in_file, 'r') as f_in:
            if '10x' not in sample.lower() and not concave_hull:
                full_body = False
            out_file = data_path + 'exf_output\\' + short_name + ('_fragment_%s'%(fmag)*(not full_body))+output_suffix
            possibleGroups = [] if concave_hull else all_nerves + ['Soma %d'%(i) for i in range(1,6)] + ['T3 paravertebral ganglion', 'Cervicothoracic ganglion']
            all_node_num, xyz_all, _, xyzGroups, _, xyz_marker, marker_names, marker_nodenum, _ = zinc_read_exf_file(
                data_path + sample, 2, 0, 1, [], possibleGroups, 3)
            xyz_n = {m: xyz_marker[im] for im, m in enumerate(marker_names)}
            xyz_all_dict = {n: xyz_all[n - 1] for n in all_node_num}
            if not full_body:
                nervename = []
                for key in xyzGroups.keys():
                    if key in all_nerves:
                        nervename.append(key)

                for nn in nervename:
                    xyz_n_0 = xyzGroups[nn]
                    xyz_n[nn] = xyz_n_0
            try:
                xyz_st = xyzGroups['Cervicothoracic ganglion']
            except:
                xyz_st = [xyz_all_dict[key] for key in xyz_all_dict.keys()]
            xyz_t3 = []
            xyz_t3Rot = []
            try:
                xyz_t3 = xyzGroups['T3 paravertebral ganglion']
                print('T3 paravertebral ganglion present')
            except:
                pass
            # there may be more than one soma
            num_soma = 0
            if not concave_hull:
                for key in xyzGroups:
                    if 'soma' in key.lower():
                        num_soma += 1

            for i in range(num_soma):
                # print('soma '+str(i+1))
                crd = xyzGroups['Soma ' + str(i + 1)]
                xyz_sm.extend([[c[0], c[1], c[2], i + 1] for c in crd])

            if full_body:
                if not concave_hull:
                    x = [(ix[0]) for ix in xyz_st]
                    y = [(ix[1]) for ix in xyz_st]
                    z = [(ix[2]) for ix in xyz_st]
                    centroid = [np.mean(x), np.mean(y), np.mean(z)]
                    xsm = [[]] * num_soma
                    ysm = [[]] * num_soma
                    zsm = [[]] * num_soma
                    isoma = [[]] * num_soma
                    for i in range(num_soma):
                        xsm[i] = [(ix[0]) for ix in xyz_sm if ix[3] == i + 1]
                        ysm[i] = [(ix[1]) for ix in xyz_sm if ix[3] == i + 1]
                        zsm[i] = [(ix[2]) for ix in xyz_sm if ix[3] == i + 1]
                        isoma[i] = i + 1
                    try:
                        xy_ICN = xyz_n['Inferior cardiac nerve']
                    except:
                        # print('no iCN. using Ventral ansa for system rotation')
                        xy_ICN = xyz_n['Ventral ansa subclavia']
                    xy_nerve_fids = [xyz_n['Dorsal ansa subclavia'], xy_ICN,
                                     xyz_n['Thoracic sympathetic nerve trunk']]

                    [x, y, z, xsm, ysm, zsm, xyz_st, xyz_sm, xyz_t3, fdict, xyz_n, xy_nerve_end,
                     _] = flip_system(x, y, z, xsm, ysm, zsm, xy_nerve_fids, xyz_t3, centroid, num_soma, fdict,
                                      xyz_n, sample, 0)
                    centroid = [np.mean(x), np.mean(y), np.mean(z)]

                [xyz_stRot, theta] = rotate_points(xyz_st, [], [])
                [xyz_smRot, theta] = rotate_points(xyz_sm, theta, [])
                if xyz_t3:
                    [xyz_t3Rot, theta] = rotate_points(xyz_t3, theta, [])
                xyz_nRot = xyz_n.copy()
                for key in xyz_n:
                    if xyz_n[key]:
                        [nrot, theta] = rotate_points([xyz_n[key]], theta, [])
                        xyz_nRot[key] = nrot[0]
                        try:
                            [newcoord, theta] = rotate_points([fdict[sample][key]], theta, [])
                            fdict[sample][key] = newcoord[0]
                            for i in range(num_soma):
                                [sout, theta] = rotate_points([fdict[sample]['Soma%s' % (str(i + 1))]], theta, [])
                                fdict[sample]['Soma%s' % (str(i + 1))] = sout[0]
                        except:
                            pass

                x = [(ix[0]) for ix in xyz_stRot]
                y = [(ix[1]) for ix in xyz_stRot]
                z = [(ix[2]) for ix in xyz_stRot]
                centroid = [np.mean(x), np.mean(y), np.mean(z)]
                xsm = [[]] * num_soma
                ysm = [[]] * num_soma
                zsm = [[]] * num_soma
                isoma = [[]] * num_soma
                for i in range(num_soma):
                    xsm[i] = [(ix[0]) for ix in xyz_smRot if ix[3] == i + 1]
                    ysm[i] = [(ix[1]) for ix in xyz_smRot if ix[3] == i + 1]
                    zsm[i] = [(ix[2]) for ix in xyz_smRot if ix[3] == i + 1]
                    isoma[i] = i + 1

            else:
                xyz_smRot_2D = xyz_sm  # [[xsm[i], ysm[i], zsm[i]] for i in range(len(xsm))]
                xyz_stRot = xyz_st.copy()
                xyz_nRot = xyz_n.copy()
                x = [(ix[0]) for ix in xyz_stRot]
                y = [(ix[1]) for ix in xyz_stRot]
                z = [(ix[2]) for ix in xyz_stRot]
                xsm = [[]] * num_soma
                ysm = [[]] * num_soma
                zsm = [[]] * num_soma
                isoma = [[]] * num_soma
                for i in range(num_soma):
                    xsm[i] = [(ix[0]) for ix in xyz_smRot_2D if ix[3] == i + 1]
                    ysm[i] = [(ix[1]) for ix in xyz_smRot_2D if ix[3] == i + 1]
                    zsm[i] = [(ix[2]) for ix in xyz_smRot_2D if ix[3] == i + 1]
                    isoma[i] = i + 1
                lr_groups = False

        # soma cx to nerves - INNERVATIONS
        cx_soma = {}
        if not concave_hull:
            xname = sample.split('.ex')[0]
            xml_path = data_path + 'xml\\'
            cx_soma = parse_xml_find_soma_ID(xml_path + xname)

        if lr_groups:
            cx = centroid[0]
            cy = centroid[1]

            if full_body:
                # parse through all stellate nodes between selected fiducials of every contour layer
                # LEFT: Dorsal ansa / C8 to Ventral ansa / inferior cardiac nerve
                # TOP: iCN/VA to trunk (or rightmost top nodes of rotated stellate)
                # BOTTOM: C8/DA to trunk or rightmost bottom nodes
                [nleft, ntop, nbottom] = lefttopbottom_by_fiducials(x, y, z, xyz_nRot, 'by_order', sample, plot)

        if lr_groups or not full_body:
            # write with zinc
            context = Context('Example')
            region = context.getDefaultRegion()
            fm = region.getFieldmodule()
            fm.beginChange()
            nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            coordinates = findOrCreateFieldCoordinates(fm, 'data_coordinates')
            nodetemplate = nodes.createNodetemplate()
            nodetemplate.defineField(coordinates)
            nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
            # stellate face groups
            armGroup12 = findOrCreateFieldGroup(fm, 'stellate face 1-2')
            armGroup23 = findOrCreateFieldGroup(fm, 'stellate face 2-3')
            armGroup31 = findOrCreateFieldGroup(fm, 'stellate face 3-1')
            armGroup12Name = findOrCreateFieldStoredString(fm, name='stellate face 1-2')
            armGroup23Name = findOrCreateFieldStoredString(fm, name='stellate face 2-3')
            armGroup31Name = findOrCreateFieldStoredString(fm, name='stellate face 3-1')
            armGroup12Nodes = findOrCreateFieldNodeGroup(armGroup12, nodes).getNodesetGroup()
            armGroup23Nodes = findOrCreateFieldNodeGroup(armGroup23, nodes).getNodesetGroup()
            armGroup31Nodes = findOrCreateFieldNodeGroup(armGroup31, nodes).getNodesetGroup()

            markerNodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            markerGroup = findOrCreateFieldGroup(fm, 'marker')
            markerName = findOrCreateFieldStoredString(fm, name='marker_data_name')
            markerPoints = findOrCreateFieldNodeGroup(markerGroup, markerNodes).getNodesetGroup()
            marker_coordinates = findOrCreateFieldCoordinates(fm, 'marker_data_coordinates')
            markerTemplate = markerPoints.createNodetemplate()
            markerTemplate.defineField(marker_coordinates)
            markerTemplate.setValueNumberOfVersions(marker_coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
            markerTemplate.defineField(markerName)

            cache = fm.createFieldcache()

            for nID0, xyz in enumerate(xyz_stRot):
                nID = nID0 + 1
                if nID in nleft:
                    node = armGroup23Nodes.createNode(nID, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyz)
                elif nID in ntop:
                    node = armGroup12Nodes.createNode(nID, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyz)
                elif nID in nbottom:
                    node = armGroup31Nodes.createNode(nID, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyz)
                else:  # nothing
                    node = nodes.createNode(nID, nodetemplate)
                    cache.setNode(node)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyz)

            nID = 1000001
            for i, key in enumerate(xyz_n):
                if xyz_n[key] and 'junction' not in key.lower():
                    if full_body:
                        xyz = find_closest_end(x, y, z, xyz_nRot[key])
                    else:  # use junction point
                        for nkey in xyz_nRot:
                            if 'junction' in nkey.lower() and key.lower() in nkey.lower():
                                xyz = xyz_nRot[nkey]
                    node = markerPoints.createNode(nID, markerTemplate)
                    cache.setNode(node)
                    marker_coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xyz)
                    markerName.assignString(cache, key)
                    nID += 1
            written_soma_connections = []
            for j in range(num_soma):
                soma_id = 'Soma ' + str(j + 1)
                for key in cx_soma[soma_id]:
                    possible_nerve = cx_soma[soma_id][key]
                    if possible_nerve:
                        soma_nerve = possible_nerve[0]
                        if (soma_nerve not in written_soma_connections):
                            soma_inn_name = 'Soma_' + soma_nerve
                            xys = [np.mean(xsm[j]), np.mean(ysm[j]), np.mean(zsm[j])]
                            node = markerPoints.createNode(nID, markerTemplate)
                            cache.setNode(node)
                            marker_coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xys)
                            markerName.assignString(cache, soma_inn_name)
                            written_soma_connections.append(soma_nerve)
                            nID += 1

            fm.endChange()
            region.writeFile(out_file)

            # write to mapclient workflow
            m_out = mapclient_workflow_path + short_name + output_suffix
            shutil.copyfile(out_file, m_out)

        else:  # do not write anything for files without LR groups
            pass

    return


if __name__ == '__main__':
    data_path = 'xml_ex\\'  # 'xml_ex\\' 'scaffoldfitter_output\\data_mesh\\processed_exf\\concave_hull\\'
    main_writerotatefile(data_path)
