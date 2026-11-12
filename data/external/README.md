# Published data in data/external

The CSV files in this folder are readings or digitisations of published data from sources with an open licence. They remain subject to the terms of their sources; the MIT licence of this repository covers the code and the files created for the book, not these data.

## apple_drying_curves_Processes2026.csv

Source: O. Dajbych, A. Kabutey, Č. Mizera, A. Sedláček, D. Herak, "Effect of Diffusion Path Lengths on Effective Moisture Diffusivity and Activation Energy of Red Delicious Apple Slices Under Convective Drying", Processes 14(12) (2026) 2015, https://doi.org/10.3390/pr14122015.
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. The licence of this repository (MIT) does not apply to this file.
Changes: the moisture-ratio curves of Fig. 2e,f were digitised from the vector graphics of the published PDF (extract_apple_drying_data.py) and resampled at 10-min intervals by linear interpolation; values are not the authors' original data. Half-thicknesses (Table 1) are transcribed in the scripts. No figure or table of the paper is reproduced.

## tunnel_oven_electric_air_temperature_Mayo2014.csv

Source: C. Mayo Bayón et al., Sensors 14(8) (2014) 14712-14731, https://doi.org/10.3390/s140814712.
Licence: Creative Commons Attribution 3.0 (CC BY 3.0), https://creativecommons.org/licenses/by/3.0/. The licence of this repository (MIT) does not apply to this file.
Changes: air temperatures along the oven were read by eye from the published figure (uncertainty about 10 degC); values are not the authors' original data. No figure or table of the paper is reproduced.

## Sources without an open licence: files not distributed

Data from publications without an open licence are not redistributed in this repository. The verification scripts that use them stop with a message when a file is absent and leave their stored result files unchanged. To repeat such a check, prepare the file from the original publication (columns below; values read from the cited figure or table) and place it in this folder.

| File | Source | Columns |
|---|---|---|
| uht_plant_heat_transfer_coefficient_Petermeier2002.csv | Petermeier et al., J. Food Eng. 55 (2002) 9-17 | time_h, overall_heat_transfer_coefficient_W_m2K |
| heat_pump_dryer_apple_Tunckal2016.csv | Tunçkal et al., Isı Bilimi ve Tekniği Dergisi 36 (2016) 161-172 | time_h, total_energy_kWh, SMER_kg_per_kWh, COP_system_without_external_condenser, COP_system_with_external_condenser |
| bread_baking_temperatures_Zanoni1993.csv | Zanoni et al., J. Food Eng. 19 (1993) 389-398 | time_min, oven_C, upper_crust_C, lower_crust_C, crumb_1cm_C, crumb_centre_C |
| molten_maize_starch_rheology_Vergnes1987.csv | Vergnes and Villemaire, Rheol. Acta 26 (1987) 570-576 | temperature_C, moisture_wet_basis_pct, power_law_index_m, consistency_K_Pa_s_m |
| acrylamide_wheat_flour_Gokmen2006.csv | Gökmen and Şenyuva, Food Chem. 99 (2006) 238-243 | time_min, acrylamide_150C_ng_g, acrylamide_180C_ng_g, acrylamide_220C_ng_g |
| extrusion_sei_moisture_DellaValle1997.csv | Della Valle et al., J. Food Eng. 31 (1997) 277-295 | moisture_content_wet_basis, sectional_expansion_index_read_from_figure, starch, figure |
| extruder_residence_time_Kumar2006.csv | Kumar et al., J. Food Eng. 75 (2006) 237-244 | barrel_temperature_C, screw_speed_rpm, mean_residence_time_s, variance_s2, number_of_CSTRs, first_tracer_appearance_s_read_from_figure |
| pseudomonas_growth_rate_Koutsoumanis2000.csv | Koutsoumanis et al., Appl. Environ. Microbiol. 66 (2000) 3528-3534 | temperature_C, mu_max_per_h_aerobic_red_mullet_pseudomonads |
| pseudomonas_growth_rate_CO2_4C_Koutsoumanis2000.csv | same | temperature_C, CO2_percent, mu_max_per_h_pseudomonads_red_mullet |
