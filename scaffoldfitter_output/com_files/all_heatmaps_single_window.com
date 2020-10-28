gfx mod spectrum default linear range 0 0.2 extend_above extend_below rainbow colour_range 1 0 ambient diffuse component 1
gfx create spectrum locus_colour
gfx mod spectrum locus_colour linear range 0 1 extend_above extend_below white_to_blue colour_range 1 0 ambient diffuse component 1

gfx read elements "..\processed_exf\Soma_Inferiorcardiacnerve.exf"
gfx modify g_element "/" lines domain_mesh1d coordinate coordinates face all tessellation default LOCAL line line_base_size 0 select_on material grey50 selected_material default_selected render_shaded;
gfx modify g_element "/" surfaces domain_mesh2d as surf_Soma_Inferiorcardiacnerve coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Soma_Inferiorcardiacnerve spectrum default selected_material default_selected render_shaded;

gfx read elements "Soma_Ventralansasubclavia.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Soma_Ventralansasubclavia coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Soma_Ventralansasubclavia spectrum default selected_material default_selected render_shaded;

gfx read elements "Soma_Dorsalansasubclavia.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Soma_Dorsalansasubclavia coordinate coordinates face all tessellation default LOCAL select_on invisible material default data locus_Soma_Dorsalansasubclavia spectrum locus_colour selected_material default_selected render_shaded;

gfx read elements "all.exf"

gfx define field marker_coordinates embedded element_xi marker_location field coordinates

gfx modify g_element /Soma_Inferiorcardiacnerve/ general clear;
gfx modify g_element /Soma_Inferiorcardiacnerve/ lines domain_mesh1d coordinate Soma_Inferiorcardiacnerve_coordinates face all tessellation default LOCAL line_width 27 line line_base_size 0 select_on material yellow selected_material default_selected render_shaded;

gfx modify g_element /Soma_Ventralansasubclavia/ general clear;
gfx modify g_element /Soma_Ventralansasubclavia/ lines domain_mesh1d coordinate Soma_Ventralansasubclavia_coordinates face all tessellation default LOCAL line_width 12 line line_base_size 0 select_on material red selected_material default_selected render_shaded;

gfx modify g_element /Soma_Dorsalansasubclavia/ general clear;
gfx modify g_element /Soma_Dorsalansasubclavia/ lines domain_mesh1d coordinate Soma_Dorsalansasubclavia_coordinates face all tessellation default LOCAL line_width 12 line line_base_size 0 select_on material cyan selected_material default_selected render_shaded;

gfx modify g_element /Soma_Thoracicspinalnerve1/ general clear;
gfx modify g_element /Soma_Thoracicspinalnerve1/ lines domain_mesh1d coordinate Soma_Thoracicspinalnerve1_coordinates face all tessellation default LOCAL line_width 3 line line_base_size 0 select_on material magenta selected_material default_selected render_shaded;

gfx modify g_element "/" points domain_nodes subgroup known_location_marker coordinate marker_data_coordinates tessellation default_points LOCAL glyph sphere size "50*50*50" offset 0,0,0 font default label marker_data_name label_offset 1,0,0 select_on material magenta selected_material default_selected render_shaded

gfx modify g_element "/" points domain_nodes subgroup unknown_location_marker coordinate marker_data_coordinates tessellation default_points LOCAL glyph sphere size "50*50*50" offset 0,0,0 font default label marker_data_name label_offset 1,0,0 select_on material white selected_material default_selected render_shaded

gfx modify g_element "/" points domain_nodes subgroup marker coordinate marker_coordinates tessellation default_points LOCAL glyph sphere size "50*50*50" offset 0,0,0 font default label marker_name select_on invisible material yellow selected_material default_selected render_shaded

gfx cre wind;
gfx mod win 1 background colour 0 0 0;
gfx edit scene;

