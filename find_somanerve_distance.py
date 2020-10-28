# 13 oct: parse through fragment xml xfiles and find the absolute distance between soma centre and nerve origin centre in um
# xml parser codeblocks copied from mbfxml2ex.py

import os
import math
import xml.etree.ElementTree as ElTree
from tools import find_closest_end, zinc_read_exf_file
import sys
sys.path.insert(1, "C:\\Users\\sfon036\\Google Drive\\SPARC_work\\codes\\python_packages\\mbfxml2ex_shelley\\src\\")
import mbfxml2ex_shelley


def find_nerve_soma_from_ex(ex_path, efiles, data):
    nerve_names = []
    for sample in data.keys():
        nerve = data[sample.split('.ex')[0]]['nerve']
        if nerve not in nerve_names:
            nerve_names.append(nerve)
    nerve_dists = {c:[] for c in nerve_names}
    groupName = []
    for ef in data.keys(): #efiles
        efile = ef + '.ex'
        print(efile)
        ekey = efile.split('.ex')[0]
        soma_names = [data[ekey][d] for d in data[ekey].keys() if 'Soma' in d]
        for skey in soma_names:
            groupName.append(skey)
        all_node_num, xyz_all, _, xyzGroups, _, xyz_marker, marker_names, marker_nodenum, _ = zinc_read_exf_file(
            ex_path + efile, 2, 0, 1, [], groupName, 3)
        soma_centroid = {c:[] for c in soma_names}
        for skey in soma_names:
            soma_centroid[skey] = [sum([x[i] for x in xyzGroups[skey]]) / len(xyzGroups[skey]) for i in range(3)]
        try:
            nerve_marker = marker_names[0].split('between ')[-1].split(' and')[0].capitalize()
        except:
            nerve_marker = []
            data[ekey]['nerve'] = 0
        if nerve_marker == data[ekey]['nerve']:
            soma_name = data[ekey]['Axon']
            nerve_centroid = xyz_marker[0]
            xdist, ydist = [(nerve_centroid[i] - soma_centroid[soma_name][i]) for i in range(2)] # there could be more than one soma
            xdist, ydist = [xdist / data[ekey]['xy_scale'][0], ydist / data[ekey]['xy_scale'][1]]
            vnorm = math.sqrt(xdist ** 2 + ydist ** 2)

            print('distance from ' + nerve_marker + ' to soma ' + soma_name + ' = ' + str(vnorm) + ' um')
            nerve_dists[nerve_marker].append(vnorm)
        else:
            print('non matching nerve junction with group marker in .ex file')

    for nerve in nerve_dists.keys():
        nerve_dists[nerve] = sum(nerve_dists[nerve])/len(nerve_dists[nerve])

    return nerve_dists


def find_xml_image_scale(xml_path, xfiles):
    # find the average scale for body files
    fulldata = {c:{} for c in xfiles}
    data_list = []
    for xf in xfiles:
        contents = mbfxml2ex_shelley.read_xml(xml_path+xf)
        scales = contents._images[0]['scale']
        fulldata[xf].update({'xy_scale': scales})
        data_list.append(scales)
    return data_list

##############################################################
#                           MAIN
##############################################################

def main_somanerve(get_scales=False):
    # ex_path = 'C:\\Users\\sfon036\\Google Drive\\SPARC_work\\GANGLIA\\STELLATE\\stellate_xml_scaffold\\xml_ex_files\\Finalized Stellate Dataset\\'
    ex_path = 'xml_ex\\Finalized Stellate Dataset\\'

    # parse xml for scale info
    # xml_path = ex_path + "xml\\"
    xml_path = ex_path
    allxfiles = os.listdir(xml_path)
    xfiles = [xf for xf in allxfiles if '10x' not in xf.lower() and xf.endswith('.xml')]
    fullxfiles = [xf for xf in allxfiles if '10x' in xf.lower() and xf.endswith('.xml')]

    data = {c:{} for c in xfiles}
    for xf in xfiles:
        contents = mbfxml2ex_shelley.read_xml(xml_path+xf)
        soma_axon_dendrite = {}
        for contour_root in contents.get_contours():
            if 'soma' in contour_root['name'].lower():
                soma_name = contour_root['name']
                dendriteID = contour_root['Dendrite ID']
                if soma_name not in soma_axon_dendrite.keys():
                    soma_axon_dendrite.update({soma_name: dendriteID})
        for tree_root in contents.get_trees():
            if 'axon' in tree_root['type'].lower():
                axon_name = tree_root['type']
                dendriteID = tree_root['Dendrite ID']
                if axon_name not in soma_axon_dendrite.keys():
                    soma_axon_dendrite.update({axon_name: dendriteID})
        for mark in contents.get_markers():
            junction_string = mark['name']
            nerve = junction_string.split('between ')[-1].split(' and')[0].capitalize()
            data[xf].update({"nerve": nerve})

        if 'nerve' in data[xf].keys():
            data[xf].update(soma_axon_dendrite)
            data[xf].update({'xy_scale': contents._images[0]['scale']})
        else:
            del data[xf]

    if get_scales:
        full_scales = find_xml_image_scale(xml_path, fullxfiles)

    # parse ex file for soma nerve pixel coordinates
    # nerve position is the point of nerve closest to the ganglion
    efiles = os.listdir(ex_path)
    efiles = [f for f in efiles if f.endswith('.ex') and '10x' not in f.lower()]

    nerve_dists = find_nerve_soma_from_ex(ex_path, efiles, data)

    if get_scales:
        return nerve_dists, full_scales
    else:
        return nerve_dists

if __name__ == "__main__":
    get_scales = False
    if get_scales:
        nerve_dists, full_scales = main_somanerve(get_scales=False)
    else:
        nerve_dists = main_somanerve(get_scales=False)





