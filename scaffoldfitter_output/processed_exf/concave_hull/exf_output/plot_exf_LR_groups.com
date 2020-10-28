gfx read elements "average_concave_hull_mesh.ex_LR.exf"

gfx modify g_element "/" general clear;
gfx modify g_element "/" points domain_nodes coordinate data_coordinates tessellation default_points LOCAL glyph point size "1*1*1" offset 0,0,0 font default select_on material default selected_material default_selected render_shaded;
gfx modify g_element "/" points domain_nodes subgroup "stellate face 2-3.nodes" coordinate data_coordinates tessellation default_points LOCAL glyph sphere size "50*50*50" offset 0,0,0 font default select_on material default selected_material default_selected render_shaded;
gfx modify g_element "/" points domain_datapoints subgroup marker coordinate marker_data_coordinates tessellation default_points LOCAL glyph point size "1*1*1" offset 0,0,0 font default label marker_data_name label_offset 0,0,0 select_on material default selected_material default_selected render_shaded;

gfx cre win
gfx edit scene