gfx mod spectrum default linear range 0 0.2 extend_above extend_below rainbow colour_range 1 0 ambient diffuse component 1

gfx read elements "..\processed_exf\Inferior Cardiac Nerve.exf"
gfx modify g_element "/" lines domain_mesh1d coordinate coordinates face all tessellation default LOCAL line line_base_size 0 select_on material default selected_material default_selected render_shaded;
gfx modify g_element "/" surfaces domain_mesh2d as surf_Inferior Cardiac Nerve coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Inferior Cardiac Nerve spectrum default selected_material default_selected render_shaded;

gfx read elements "Ventral Ansa Subclavia.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Ventral Ansa Subclavia coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Ventral Ansa Subclavia spectrum default selected_material default_selected render_shaded;

gfx read elements "Dorsal Ansa Subclavia.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Dorsal Ansa Subclavia coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Dorsal Ansa Subclavia spectrum default selected_material default_selected render_shaded;

gfx read elements "Cervical Spinal Nerve 8.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Cervical Spinal Nerve 8 coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Cervical Spinal Nerve 8 spectrum default selected_material default_selected render_shaded;

gfx read elements "Thoracic Spinal Nerve 1.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Thoracic Spinal Nerve 1 coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Thoracic Spinal Nerve 1 spectrum default selected_material default_selected render_shaded;

gfx read elements "Thoracic Spinal Nerve 2.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Thoracic Spinal Nerve 2 coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Thoracic Spinal Nerve 2 spectrum default selected_material default_selected render_shaded;

gfx read elements "Thoracic Spinal Nerve 3.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Thoracic Spinal Nerve 3 coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Thoracic Spinal Nerve 3 spectrum default selected_material default_selected render_shaded;

gfx read elements "Thoracic Sympathetic Nerve Trunk.exf"
gfx modify g_element "/" surfaces domain_mesh2d as surf_Thoracic Sympathetic Nerve Trunk coordinate coordinates face all tessellation default LOCAL select_on invisible material default data probability_Thoracic Sympathetic Nerve Trunk spectrum default selected_material default_selected render_shaded;

gfx read elements "all.exf"

gfx modify g_element /soma_Inferior cardiac nerve/ general clear;
gfx modify g_element /soma_Inferior cardiac nerve/ lines domain_mesh1d coordinate soma_Inferior cardiac nerve_coordinates face all tessellation default LOCAL line_width 33 line line_base_size 0 select_on material yellow selected_material default_selected render_shaded;

gfx modify g_element /soma_Ventral ansa subclavia/ general clear;
gfx modify g_element /soma_Ventral ansa subclavia/ lines domain_mesh1d coordinate soma_Ventral ansa subclavia_coordinates face all tessellation default LOCAL line_width 12 line line_base_size 0 select_on material red selected_material default_selected render_shaded;

gfx modify g_element /soma_Dorsal ansa subclavia/ general clear;
gfx modify g_element /soma_Dorsal ansa subclavia/ lines domain_mesh1d coordinate soma_Dorsal ansa subclavia_coordinates face all tessellation default LOCAL line_width 18 line line_base_size 0 select_on material cyan selected_material default_selected render_shaded;

gfx modify g_element /soma_Thoracic spinal nerve 1/ general clear;
gfx modify g_element /soma_Thoracic spinal nerve 1/ lines domain_mesh1d coordinate soma_Thoracic spinal nerve 1_coordinates face all tessellation default LOCAL line_width 3 line line_base_size 0 select_on material magenta selected_material default_selected render_shaded;

gfx modify g_element "/" points domain_nodes subgroup marker coordinate marker_data_coordinates tessellation default_points LOCAL glyph point size "1*1*1" offset 0,0,0 font default label marker_data_name

gfx cre wind;
gfx mod win 1 background colour 0 0 0;
gfx edit scene;

