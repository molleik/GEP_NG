# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:48:13 2022

@author: Majd Olleik
"""
import pandas as pd
import itertools
import copy
import os

from model_v0_6 import UpstreamElecModel

out_dir = 'results/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model = UpstreamElecModel(_file_name='input_file_v5.0.xlsx',
                          _output_dir=out_dir)

"""
re_limits = [0, 0.18, 0.3]
budgets = [1000, 3000, 5000]
discount_rates = [0.07, 0.11, 0.15]
up_cost_factors = [0.8, 1, 1.2]
import_gp_factors = [0.8, 1, 1.2]
re_cost_factors = [1]
first_id = 1
"""


re_limits = [0.18]
budgets = [3000]
discount_rates = [0.11]
up_cost_factors = [1]
import_gp_factors = [1]
first_id = 300

#Added for additional sensitivity on renewable energy capex
re_cost_factors = [0.8, 1, 1.2]




all_comb = [[i, b, re, dr, up, imp, re_c]
            for i, (b, re, dr, up, imp, re_c) in 
            enumerate(itertools.product(
                                        budgets,
                                        re_limits,
                                        discount_rates,
                                        up_cost_factors,
                                        import_gp_factors,
                                        re_cost_factors), 
                      start=first_id)]

no_upstream = False
summary_list = []

base_up_capex = model.up_u_capex
base_up_opex = model.up_u_opex
base_import_gp = copy.deepcopy(model.import_gp)
base_pv_c = model.e_u_capex[2]
base_wind_c = model.e_u_capex[3]
gp_step = 0.1
spg_step = 0.01



for comb in all_comb:
    my_id = int(comb[0])
    b = int(comb[1])
    re = comb[2]
    dr = comb[3]
    up = comb[4]
    imp = comb[5]
    re_c = comb[6]
    
    outfile = 'id_' + str(my_id) + '-b_' + str(b) + '-d_' + str(dr) +  \
        '-up_' + str(up) + '-imp_' + str(imp) + '-re_' \
            + str(re) + 're_c' + str(re_c) + '.xlsx'
            
    model.budget = [b] * model.years
    model.i_s = dr
    model.i_e = dr
    model.up_u_capex = base_up_capex * up
    model.up_u_opex = base_up_opex * up
    model.import_gp = base_import_gp * imp
    
    # Added for RE capex sensitivity
    model.e_u_capex[2] = base_pv_c * re_c
    model.e_u_capex[3] = base_wind_c * re_c
    
    gp_start = model.up_u_capex + model.up_u_opex
    
    if no_upstream:
        gp_max = gp_start
    else:
        if(b == 5000):
            gp_max = 17
        else:
            gp_max = 15
        

    model.iterate(_gp_start=gp_start, _gp_step=gp_step, 
                  _spg_step=spg_step,_gp_max=gp_max, 
                  _re_limit=re, _out_file=outfile)
    
   
    summary_list.append([my_id,b,dr,up,imp,re, re_c
                         model.obj,
                         model.prod_gas_all, 
                         model.pv_emissions,
                         model.added_cap_pv_2030])
    print(str(my_id) + " is done.")
    
summary_df = pd.DataFrame(list(summary_list), 
                                    columns=['ID',
                                             'Budget', 
                                             'Discount Rate', 
                                             'Upstream Cost Factor',
                                             'NG Imp Price Factor', 
                                             'RE Target',
                                             'RE Cost Factor',
                                             'Obj',
                                             'Plateau Gas',
                                             'Emissions_PV',
                                             'PV_Added_2030'
                                             ])
summary_df.to_excel(out_dir+"SUMMARY.xlsx")