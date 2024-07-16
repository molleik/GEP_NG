# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:24:12 2022

@author: Majd Olleik
"""

import numpy as np
import pandas as pd
import math
import timeit
import os
from gurobipy import *

# First year in the model is 2023
START_YEAR = 2023
RET_YEAR = 2025 # retirement year of existing HFO and DO plants
THOUSAND = 1000
MILLION = 1000000
MMBTU_TO_KCF = 1 / 1.037
BTU_TO_KCF = MMBTU_TO_KCF / MILLION
EPSILON = 0.001


class UpstreamElecModel:
    """
    
    """
    def __init__(self, _output_dir:str, _file_name:str=None, _data=None,
                 _threads=0):
        """
        Parameters
        ----------
        _file_name : str
            DESCRIPTION.
        _output_dir : str
            DESCRIPTION.
        _seed : int, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.
        """
        self.file_name = _file_name
        self.supplied_data = _data
        self.output_dir = _output_dir
        self.threads = _threads
        self.load_data()
        
    def load_data(self):
        # Param
        if(self.supplied_data is not None):
            self.data = self.supplied_data
        else:
            self.data = pd.read_excel(self.file_name, sheet_name=None)
            
        self.years = int(self.data['Param']['Y'][0])
        self.days = int(self.data['Param']['D'][0])
        self.hours = int(self.data['Param']['H'][0])
        self.techs = int(self.data['Param']['T'][0])
        self.i_e = self.data['Param']['i_e'][0]
        self.i_up = self.data['Param']['i_up'][0]
        self.i_s = self.data['Param']['i_s'][0]
        self.e_demand_growth = self.data['Param']['ElecDemandGrowth'][0]
        self.d_weight = self.data['Weights']['w'].to_numpy()
        
        self.import_gp = self.data['GP']['Mean'].to_numpy()
        
        # Electricity
        self.tech_name = self.data['Elec']['Name'].to_numpy()
        self.tech_fuel = self.data['Elec']['Fuel'].to_numpy()
        self.build_new = self.data['Elec']['NewCap'].to_numpy()
        self.inst_cap18 = self.data['Elec']['InstCap2018'].to_numpy()
        self.prod_cap18 = self.data['Elec']['ProdCap2018'].to_numpy()
        self.e_u_capex = self.data['Elec']['UnitCapex'].to_numpy()
        self.e_u_fx_opex = self.data['Elec']['FixedOpex'].to_numpy()
        self.e_u_var_opex = self.data['Elec']['VarOpex'].to_numpy() 
        self.e_heat_rate = self.data['Elec']['HeatRate'].to_numpy()
        self.e_yearly_red_prod = self.data['Elec']['RedInProdCap'].to_numpy()
        self.e_lead_time = self.data['Elec']['LeadTime'].to_numpy()
        self.e_u_emissions = self.data['Elec']['Emissions'].to_numpy() #Tonne/MWh
        self.e_life = self.data['Elec']['Life'].to_numpy()
        self.e_min_cap = self.data['Elec']['MinCap'].to_numpy()
        self.e_min_disp = self.data['Elec']['MinDisp'].to_numpy()
        self.gas_techs = np.where(self.tech_fuel == 'NG')[0]
        self.e_disp = self.data['Elec']['Disp'].to_numpy()
        self.disp_techs = np.where(self.e_disp == True)[0]
        self.pv_techs = np.where(self.tech_name == 'PV')[0]
        self.wind_techs = np.where(self.tech_name == 'Wind')[0]
        self.hydro_techs = np.where(self.tech_name == 'Hydro')[0]
        self.re_techs = np.concatenate([self.pv_techs, self.wind_techs,
                                        self.hydro_techs])
        self.no_new_cap_techs = np.where(self.build_new == False)[0]
        
        self.pv_out_per_unit_cap = np.transpose(
            self.data['PVProd'].iloc[:,1:self.days+1].to_numpy()) #UnitH/Unit
        
        self.wind_out_per_unit_cap = np.transpose(
            self.data['WindProd'].iloc[:,1:self.days+1].to_numpy()) #UnitH/Unit
        
        self.e_demand18 = np.transpose(
            self.data['ElecDemand'].iloc[:,1:self.days+1].to_numpy()) #MW
        
        self.budget = self.data['Budget']['Budget'].to_numpy() #MM$
        
        #changing units
        self.e_u_capex = self.e_u_capex / THOUSAND # MM$/MW
        self.e_u_fx_opex = self.e_u_fx_opex / THOUSAND # MM$/MW/year
        self.e_u_var_opex = self.e_u_var_opex / MILLION # MM$/MWh
        self.e_heat_rate = self.e_heat_rate * BTU_TO_KCF * THOUSAND # kcf/MWh
        
        self.inst_cap_start = self.inst_cap18
        self.prod_cap_start = self.prod_cap18 - \
            self.inst_cap18 * self.e_yearly_red_prod * (START_YEAR - 2018)
            
        #### Adding 850 MW of installed offgrid PV capacity by 2023
        self.inst_cap_start[2] = 850
        self.prod_cap_start[2] = 850
        
        self.e_demand_start = self.e_demand18 \
            * (1 + self.e_demand_growth) ** (START_YEAR - 2018)
        
        
        # Upstream
        self.rho = self.data['Up']['rho'][0]
        self.cg = self.data['Up']['CG'][0]
        self.tr = self.data['Up']['TR'][0]
        self.up_u_capex = self.data['Up']['UnitCapex'][0]
        self.up_u_opex = self.data['Up']['UnitOpex'][0]
        self.up_lead_time = int(self.data['Up']['LeadTime'][0])
        self.up_dep_time = int(self.data['Up']['DepTime'][0])
        self.up_u_emissions = self.data['Up']['Emissions'][0]
        self.up_alpha = self.data['Up']['Alpha'][0]
        self.max_yearly_emissions = None
        self.min_re_target = None
        
        # Retirement
        self.ret_inst_cap = np.zeros([self.techs, self.years])
        self.ret_prod_cap = np.zeros([self.techs, self.years])
        self.ret_techs = np.where(np.logical_or(self.tech_name == 'HFO',
                                  self.tech_name == 'DO'))[0]
        for t in self.ret_techs:
            self.ret_inst_cap[t, RET_YEAR - START_YEAR] = \
                self.inst_cap_start[t]
            self.ret_prod_cap[t, RET_YEAR - START_YEAR] = \
                self.prod_cap_start[t] - (self.inst_cap_start[t] * 
                                      self.e_yearly_red_prod[t]) * (RET_YEAR - 
                                                                    START_YEAR
                                                                    - 1)
        
        
        
        
        
    def solve(self, _up_gp=6, _state_profit_gas=0.1, _store_results=False):
        """

        Returns
        -------
        None.

        """
        
        self.up_gp = _up_gp
        self.state_pg = _state_profit_gas
        
        if self.max_yearly_emissions == None:
            self.max_yearly_emissions = 100000 * THOUSAND #Tonnes CO2
        
        
        try:
            # creating a new gurobi environment. Relevant for multithreading
            my_gb_env = Env()
            
            # setting the number of threads to be used.
            if self.threads > 0:
                my_gb_env.setParam("Threads",self.threads)
            my_gb_env.setParam("LogToConsole",0)
            self.mod = Model("deterministic",my_gb_env)
            m = self.mod
            
            # if self.threads > 0:
            #    self.mod.Params.Threads = self.threads
            
            # Electricity related
            # continuous [0,+inf]
            added_cap = m.addVars(self.techs, self.years, name='addedCap') #MW
            inst_cap = m.addVars(self.techs, self.years, name='instCap') # MW
            prod_cap = m.addVars(self.techs, self.years, name='prodCap') # MW
            disp_cap = m.addVars(self.techs, 
                                 self.years, self.days, 
                                 self.hours, name='dispCap') # MW
            e_capex = m.addVars(self.years, name='eCapex') # MM$
            e_fx_opex = m.addVars(self.years, name='eFxOpex') # MM$
            e_var_opex = m.addVars(self.years, name='eVarOpex') # MM$
            prod_elec = m.addVars(self.techs, self.years, name='prodElec') #MWh
            gas_needed = m.addVars(self.years, name='gasNeeded') # bcf
            e_fuel_opex = m.addVars(self.years, name='eFuelOpex') # MM$
            e_resid = m.addVar(name='eResid') # MM$
            e_emissions = m.addVars(self.techs, self.years, name='eEmissions')
            
            # Upstream related
            royalty = m.addVars(self.years, name='royalty') # MM$
            s_profit_gas = m.addVars(self.years, name='stateProfitGas') # MM$
            c_profit_gas = m.addVars(self.years, name='compProfitGas') # MM$
            tax = m.addVars(self.years, name='taxes') # MM$
            disp_gas = m.addVars(self.years, name='dispGas') # MM$
            cost_gas = m.addVars(self.years, name='costGas') # MM$
            up_capex = m.addVars(self.years, name='upCapex') # MM$
            up_dep_capex = m.addVars(self.years, name='upDepCapex') # MM$
            up_opex = m.addVars(self.years, name='upOpex') # MM$
            taxable_amount = m.addVars(self.years, lb=-GRB.INFINITY,
                                       name='taxableAmount') # MM$
            up_comp_rev = m.addVars(self.years, lb=-GRB.INFINITY,
                                    name='upCompRev') # MM$
            
            up_total_capex = m.addVar(name='upTotalCapex') # MM$
            prod_gas = m.addVars(self.years, name='prodGas') # bcf(million kcf)
            prod_gas_all = m.addVar(name='prodGasAll')
            up_emissions = m.addVars(self.years, name='upEmissions') #tonnes
            
            # Binary variables
            b1 = m.addVars(self.years, vtype=GRB.BINARY, name='b1')
            b2 = m.addVars(self.techs, self.years, vtype=GRB.BINARY, name='b2')
            b3 = m.addVars(self.years, vtype=GRB.BINARY, name='b3')
            b4 = m.addVars(self.years, vtype=GRB.BINARY, name='b4')
            
            
            
            #Setting the objective function
            e_costs_npv = quicksum((e_capex[y] + e_fx_opex[y] + e_var_opex[y]
                                    + e_fuel_opex[y]) / ((1+self.i_e) ** (y+1))
                                   for y in range(self.years))
            
            e_resid_npv = e_resid / ((1+self.i_e) ** (self.years))
            
            u_npv = quicksum((royalty[y] + s_profit_gas[y] + tax[y])
                             / ((1+self.i_s) ** (y+1))
                             for y in range(self.years))
            
            m.setObjective(e_costs_npv - u_npv - e_resid_npv, GRB.MINIMIZE)
            
            #print(m.getObjective())
            
            #Setting the constraints
            
            #e_capex:
            m.addConstrs((e_capex[y] == 
                         quicksum(self.e_u_capex[t] / self.e_lead_time[t] 
                                  * added_cap[t,y-z+1]
                                  for t in range(self.techs)
                                  for z in range(1, min(self.e_lead_time[t], 
                                                        y + 1) + 1))
                         for y in range(self.years)),
                         name='eCapex_ctr')
            
            #e_fx_opex:
            m.addConstrs((e_fx_opex[y] == 
                          quicksum(self.e_u_fx_opex[t] * inst_cap[t,y]
                                   for t in range(self.techs))
                         for y in range(self.years)),
                         name='eFxOpex_ctr')
            
            #inst_cap:
            m.addConstrs((inst_cap[t,0] == self.inst_cap_start[t]
                         for t in range(self.techs)),
                         name='intCap0_ctr')
            
            
            m.addConstrs((inst_cap[t,y] == inst_cap[t,y-1] - 
                          self.ret_inst_cap[t,y]
                         for t in range(self.techs)
                         for y in range(1, self.e_lead_time[t])
                         if(self.e_lead_time[t] > 1)),
                         name='instCapLT_ctr')
            
            m.addConstrs((inst_cap[t,y] == inst_cap[t,y-1] + 
                          added_cap[t, y-self.e_lead_time[t]] - 
                          self.ret_inst_cap[t,y]
                         for t in range(self.techs)
                         for y in range(self.e_lead_time[t], self.years)),
                         name='instCap_ctr')
            
            #e_var_opex:
            m.addConstrs((e_var_opex[y] == 
                          quicksum(self.e_u_var_opex[t] * prod_elec[t,y]
                                   for t in range(self.techs))
                          for y in range(self.years)),
                         name='eVarOpex_ctr')
            
            #prod_elec:
            m.addConstrs((prod_elec[t,y] == 
                          quicksum(self.d_weight[d] * 365 * disp_cap[t,y,d,h]
                                   for d in range(self.days)
                                   for h in range(self.hours))
                          for y in range(self.years)
                          for t in range(self.techs)),
                         name='prodElec_ctr')
            
            #gas_needed:
            m.addConstrs((gas_needed[y] == 
                          quicksum(self.d_weight[d] * 365 * disp_cap[t,y,d,h]
                                   * self.e_heat_rate[t] / MILLION
                                   for t in self.gas_techs
                                   for d in range(self.days)
                                   for h in range(self.hours))
                          for y in range(self.years)),
                         name='gasNeeded_ctr')
            
            #e_fuel_opex
            """
            if gas_needed[y] <= prod_gas:
                e_fuel_opex[y] = gas_needed[y] * u_gp
            else: 
                A = prod_gas * u_gp + (gas_needed[y] - prod_gas) 
                                 * import_gp[y]
                e_fuel_opex[y] = A
            
            using b1[y] (binary variable):
            1) e_fuel_opex[y] + (1-b1[y]) * bigM >= gas_needed[y] * u_gp
            2) e_fuel_opex[y] - (1-b1[y]) * bigM <= gas_needed[y] * u_gp
            3) gas_needed[y] + b1[y] * bigM >= prod_gas
            
            4) e_fuel_opex[y] + b1[y] * bigM >= A
            5) e_fuel_opex[y] - b1[y] * bigM <= A
            6) gas_needed[y] - (1 - b1[y]) * bigM <= prod_gas
            
            What is bigM?
            bigM > gas_needed[y] * u_gp
            bigM > A
            bigM > prod_gas
            We also know that prod_gas is bounded by:
                max(gas_needed[y]*(1+alpha))
            so:
            bigM > max(gas_needed[y]) * u_gp * (1+alpha)
            bigM > max(gas_needed[y]) * max(import_gp[y])
            bigM > max(gas_needed[y]) * (1+alpha)
            We can replace max(gas_needed[y]) by the maximum gas demand at
            the end of the model life if all plants run on natural gas.
            As such, the right hand side elements are all inputs 
            (u_gp is a decision variable treated as an input for the sake of
             iterating over it). Therefore, bigM can now be set accordingly.
            """
            # the below should be updated if a prob. approach is followed
            
            max_elec_demand = np.amax(self.e_demand_start) \
                * (1 + self.e_demand_growth) ** self.years * 24 * 365
            max_gas_heat_rate = max(self.e_heat_rate[self.gas_techs])
            max_gas_demand = max_elec_demand * max_gas_heat_rate / MILLION \
                * (1+ self.up_alpha)
            bigM = max_gas_demand * max(self.up_gp, max(self.import_gp)) + 1
            if bigM < max_gas_demand:
                bigM = max_gas_demand + 1
            
            #1: e_fuel_opex[y] + (1-b1[y]) * bigM >= gas_needed[y] * u_gp
            m.addConstrs((e_fuel_opex[y] + (1-b1[y]) * bigM >=
                          gas_needed[y] * self.up_gp
                          for y in range(self.years)),
                         name='fuelOpex1_ctr')
            
            #2: e_fuel_opex[y] - (1-b1[y]) * bigM <= gas_needed[y] * u_gp
            m.addConstrs((e_fuel_opex[y] - (1-b1[y]) * bigM <=
                          gas_needed[y] * self.up_gp
                          for y in range(self.years)),
                         name='fuelOpex2_ctr')
           
            #3: gas_needed[y] + b1[y] * bigM >= prod_gas
            m.addConstrs((gas_needed[y] + b1[y] * bigM >= prod_gas[y]
                          for y in range(self.years)),
                         name='fuelOpex3_ctr')
            
            #4: e_fuel_opex[y] + b1[y] * bigM >= A
            # A = prod_gas * u_gp + (gas_needed[y] - prod_gas) 
            #                     * import_gp[y]
            
            m.addConstrs((e_fuel_opex[y] + b1[y] * bigM >=
                          prod_gas[y] * self.up_gp + 
                          (gas_needed[y] - prod_gas[y]) * self.import_gp[y]
                          for y in range(self.years)),
                         name='fuelOpex4_ctr')
            
            #5: e_fuel_opex[y] - b1[y] * bigM <= A
            m.addConstrs((e_fuel_opex[y] - b1[y] * bigM <=
                          prod_gas[y] * self.up_gp + 
                          (gas_needed[y] - prod_gas[y]) * self.import_gp[y]
                          for y in range(self.years)),
                         name='fuelOpex5_ctr')
            
            #6: gas_needed[y] - (1 - b1[y]) * bigM <= prod_gas
            m.addConstrs((gas_needed[y] - (1-b1[y]) * bigM <= prod_gas[y]
                          for y in range(self.years)),
                         name='fuelOpex6_ctr')
            
            #prod_cap:
            m.addConstrs((prod_cap[t,0] == self.prod_cap_start[t]
                          for t in range(self.techs)),
                         name='prodCap0_ctr')

            
            m.addConstrs((prod_cap[t,y] == prod_cap[t,y-1] 
                          - inst_cap[t,y] * self.e_yearly_red_prod[t]
                          - self.ret_prod_cap[t,y]
                         for t in range(self.techs)
                         for y in range(1, self.e_lead_time[t])
                         if(self.e_lead_time[t] > 1)),
                         name='prodCapLT_ctr')
            
            m.addConstrs((prod_cap[t,y] == prod_cap[t,y-1] 
                          - inst_cap[t,y] * self.e_yearly_red_prod[t]
                          + added_cap[t, y-self.e_lead_time[t]]
                          - self.ret_prod_cap[t,y]
                         for t in range(self.techs)
                         for y in range(self.e_lead_time[t], self.years)),
                         name='prodCap_ctr')
            
            #added_cap:
            #: no added capacity that will not be used in model life
            m.addConstrs((added_cap[t,y] == 0
                          for t in range(self.techs)
                          for y in range(self.years - self.e_lead_time[t], 
                                         self.years)),
                         name='addedCap1_ctr')
            
            """
            added_cap[t,y] > 0 ==> added_cap[t,y] >= self.e_min_cap[t]
            is equivalent to:
            added_cap[t,y] >= self.e_min_cap[t] or added_cap[t,y] <= 0
            introducing the binary variable b2[t,y]:
            1: added_cap[t,y] + b2[t,y] * bigM >= self.e_min_cap[t]
            2: added_cap[t,y] - (1 - b2[t,y]) * bigM <= 0
            
            Where bigM > max(self.e_min_cap) and bigM > max(added_cap[t,y])
            We can set bigM to the multiples (10) of the maximum elec demand.
            """
            bigM2 = 10 * max_elec_demand
            #1: added_cap[t,y] + b2[t,y] * bigM >= self.e_min_cap[t]
            m.addConstrs((added_cap[t,y] + b2[t,y] * bigM2 >=
                         self.e_min_cap[t]
                         for t in range(self.techs)
                         for y in range(self.years)),
                        name='addedCap2_ctr')
            
            #2: added_cap[t,y] - (1 - b2[t,y]) * bigM <= 0
            m.addConstrs((added_cap[t,y] - (1-b2[t,y]) * bigM2 <= 0
                          for t in range(self.techs)
                          for y in range(self.years)),
                         name='addedCap3_ctr')
            
            #3: added_cap[t,y] == 0 for all techs with no new cap allowed
            m.addConstrs((added_cap[t,y] == 0
                          for t in self.no_new_cap_techs
                          for y in range(self.years)),
                         name='addedCap4_ctr')
            
            #resid:
            m.addConstr((e_resid == 
                         quicksum(self.e_u_capex[t] * added_cap[t,y]
                                  * (1 - (self.years-y-self.e_lead_time[t]) 
                                     / self.e_life[t])
                                  for t in range(self.techs)
                                  for y in range(self.years))),
                        name='resid_ctr')
            
            #total disp_cap:
            m.addConstrs((quicksum(disp_cap[t,y,d,h]
                                   for t in range(self.techs)) ==
                          self.e_demand_start[d,h] *(1+self.e_demand_growth)**y
                          for y in range(self.years)
                          for d in range(self.days)
                          for h in range(self.hours)),
                         name='totDispCap_ctr')
            
            #max disp_cap:
            # for dispatchable plants, the maximum dispatch is their prod_cap
            m.addConstrs((disp_cap[t,y,d,h] <= prod_cap[t,y]
                          for t in self.disp_techs
                          for y in range(self.years)
                          for d in range(self.days)
                          for h in range(self.hours)),
                         name='maxDispCap1_ctr')
            
            # for PV it depends on weather conditions
            m.addConstrs((disp_cap[t,y,d,h] <= 
                          prod_cap[t,y] * self.pv_out_per_unit_cap[d,h]
                          for t in self.pv_techs
                          for y in range(self.years)
                          for d in range(self.days)
                          for h in range(self.hours)),
                         name='maxDispCap2_ctr')
            
            # for wind it depends on weather conditions
            m.addConstrs((disp_cap[t,y,d,h] <= 
                          prod_cap[t,y] * self.wind_out_per_unit_cap[d,h]
                          for t in self.wind_techs
                          for y in range(self.years)
                          for d in range(self.days)
                          for h in range(self.hours)),
                         name='maxDispCap3_ctr')
            
            #min disp_cap:
            m.addConstrs((disp_cap[t,y,d,h] >= 
                          prod_cap[t,y] * self.e_min_disp[t]
                          for t in range(self.techs)
                          for y in range(self.years)
                          for d in range(self.days)
                          for h in range(self.hours)),
                         name='minDispCap_ctr')
            
            #budget:
            m.addConstrs((e_capex[y] + e_fx_opex[y] + 
                          e_var_opex[y] + e_fuel_opex[y]
                          - self.e_u_var_opex[7] * prod_elec[7,y] <=
                          self.budget[y]
                          for y in range(self.years)),
                         name='budget_ctr')
            
            #royalty:
            m.addConstrs((royalty[y] == self.rho * prod_gas[y] * self.up_gp
                         for y in range(self.years)),
                         name='royalty_ctr')
            
            #disp_gas:
            m.addConstrs((disp_gas[y] ==(1-self.rho) * prod_gas[y] * self.up_gp
                          for y in range(self.years)),
                         name='dispGas_ctr')
            
            """
            cost_gas:
            cost_gas = min(A,B)
            where A = sum_{till y - 1} (up_capex + up_opex - cost_gas)
            + up_capex[y] + up_opex[y]
            and B = disp_gas[y] * self.cg
            can be written as:
            if(A < B) cost_gas = A
            if(B < A) cost_gas = B
            if(b==A) cost_gas = A = B
            
            1) cost_gas[y] + b3[y] * bigM >= A
            2) cost_gas[y] - b3[y] * bigM <= A
            3) A + (1 - b3[y]) * bigM >= B
            
            4) cost_gas[y] + (1 - b3[y]) * bigM >= B
            5) cost_gas[y] - (1 - b3[y]) * bigM <= B
            6) B + b3[y] * bigM >= A
    
            bigM >= A ==> 
                bigM >= (self.up_u_capex + self.up_u_opex) * max_gas_demand * 
                        years
            bigM >= B ==>
                bigM >= max_gas_demand * self.up_gp
            """
            bigM3 = (self.up_u_capex + self.up_u_opex) \
                * max_gas_demand * self.years + 1
            if (max_gas_demand * self.up_gp > bigM3):
                bigM3 = max_gas_demand * self.up_gp + 1
            
           
            #1: cost_gas[y] + b3[y] * bigM >= A
            m.addConstrs((cost_gas[y] + b3[y] * bigM3 >=
                           up_capex[y] + up_opex[y]
                           + quicksum(up_capex[n] + up_opex[n] - cost_gas[n]
                                      for n in range(0,y))
                           for y in range(self.years)),
                          name='costGas1_ctr')
            
            #2: cost_gas[y] - b3[y] * bigM <= A
            m.addConstrs((cost_gas[y] - b3[y] * bigM3 <=
                           up_capex[y] + up_opex[y]
                           + quicksum(up_capex[n] + up_opex[n] - cost_gas[n]
                                      for n in range(0,y))
                           for y in range(self.years)),
                          name='costGas2_ctr')
            
            #3: A + (1 - b3[y]) * bigM >= B
            m.addConstrs((up_capex[y] + up_opex[y]
                           + quicksum(up_capex[n] + up_opex[n] - cost_gas[n]
                                      for n in range(0,y)) 
                           + (1 - b3[y]) * bigM3 >= disp_gas[y] * self.cg
                           for y in range(self.years)),
                          name='costGas3_ctr')
            
            #4: cost_gas[y] + (1 - b3[y]) * bigM >= B
            m.addConstrs((cost_gas[y] + (1-b3[y]) * bigM3 >= 
                          disp_gas[y] * self.cg
                          for y in range(self.years)),
                          name='costGas4_ctr')
            
            #5: cost_gas[y] - (1 - b3[y]) * bigM <= B
            m.addConstrs((cost_gas[y] - (1-b3[y]) * bigM3 <= 
                          disp_gas[y] * self.cg
                          for y in range(self.years)),
                          name='costGas5_ctr')
            
            #6: B + b3[y] * bigM >= A
            m.addConstrs((disp_gas[y] * self.cg + b3[y] * bigM3 >=
                          up_capex[y] + up_opex[y]
                           + quicksum(up_capex[n] + up_opex[n] - cost_gas[n]
                                      for n in range(0,y))
                           for y in range(self.years)),
                         name='costGas6_ctr')
            
            #State profit_gas:
            m.addConstrs((s_profit_gas[y] == 
                         (disp_gas[y] - cost_gas[y]) * self.state_pg
                         for y in range(self.years)),
                         name='stateProfitGas_ctr')
            
            #Comp profit_gas:
            m.addConstrs((c_profit_gas[y] == 
                         (disp_gas[y] - cost_gas[y]) * (1 - self.state_pg)
                         for y in range(self.years)),
                         name='compProfitGas_ctr')
            
            #Taxable amount:
            m.addConstrs((taxable_amount[y] == 
                          c_profit_gas[y] + cost_gas[y] - up_dep_capex[y]
                          - up_opex[y]
                         for y in range(self.years)),
                         name='taxableAmount_ctr')
            
            #Taxes
            """
            tax[y] = max(self.tr * taxable_amount[y], 0)
            can be written as:
                self.tr * taxable_amount[y] > 0 ==> tax[y] = self.tr * ...
                self.tr * taxable_amount[y] < 0 ==> tax[y] = 0
            using b4[y]:
            1) tax[y] + b4[y] * bigM >= self.tr * taxable_amount[y]
            2) tax[y] - b4[y] * bigM <= self.tr * taxable_amount[y]
            3) self.tr * taxable_amount[y] - (1 - b4[y]) * bigM <= 0
    
            4) tax[y] + (1-b4[y]) * bigM >= 0
            5) tax[y] - (1-b4[y]) * bigM <= 0
            6) self.tr * taxable_amount[y] + b4[y] * bigM >= 0
            
            bigM >= self.tr * taxable_amount[y]
            bigM >= 0
            ==> bigM >= self.tr * max_gas_demand * self.up_gp
            """
            bigM4 = self.tr * max_gas_demand * self.up_gp + 1
            
            #1: tax[y] + b4[y] * bigM >= self.tr * taxable_amount[y]
            m.addConstrs((tax[y] + b4[y] * bigM4 >=
                          self.tr * taxable_amount[y]
                          for y in range(self.years)),
                         name='tax1_ctr')
            
            #2: tax[y] - b4[y] * bigM <= self.tr * taxable_amount[y]
            m.addConstrs((tax[y] - b4[y] * bigM4 <=
                          self.tr * taxable_amount[y]
                          for y in range(self.years)),
                         name='tax2_ctr')
            
            #3: self.tr * taxable_amount[y] - (1 - b4[y]) * bigM <= 0
            m.addConstrs((self.tr * taxable_amount[y] - (1 - b4[y]) * bigM4 <=
                          0
                          for y in range(self.years)),
                         name='tax3_ctr')
    
            #4: tax[y] + (1-b4[y]) * bigM >= 0
            m.addConstrs((tax[y] + (1 - b4[y]) * bigM4 >= 0
                          for y in range(self.years)),
                         name='tax4_ctr')
            
            #5: tax[y] - (1-b4[y]) * bigM <= 0
            m.addConstrs((tax[y] - (1 - b4[y]) * bigM4 <= 0
                          for y in range(self.years)),
                         name='tax5_ctr')
            
            #6: self.tr * taxable_amount[y] + b4[y] * bigM >= 0
            m.addConstrs((self.tr * taxable_amount[y] + b4[y] * bigM4 >= 0
                          for y in range(self.years)),
                         name='tax6_ctr')
            
            #comp_rev:
            m.addConstrs((up_comp_rev[y] ==
                          c_profit_gas[y] + cost_gas[y] - tax[y] - up_capex[y] 
                          - up_opex[y]
                          for y in range(self.years)),
                         name='compRev_ctr')
            
            #IRR:
            m.addConstr((quicksum(up_comp_rev[y]/((1+self.i_up)**(y+1))
                                   for y in range(self.years)) >= 0),
                         name='IRR_ctr')
            
            #up_opex:
            m.addConstrs((up_opex[y] == prod_gas[y] * self.up_u_opex
                          for y in range(self.years)),
                         name='upOpex_ctr')
            
            #total_capex:
            m.addConstr((up_total_capex == self.up_u_capex * 
                         quicksum(prod_gas[y] for y in range(self.years))),
                        name='totalCapex_ctr')
            
            total_prod_gas = quicksum(prod_gas[y] for y in range(self.years))
            total_gas_needed = quicksum(gas_needed[y] 
                                        for y in range(self.years))
            
            #up_capex:
            
            m.addConstrs((up_capex[y] == 
                          up_total_capex / self.up_lead_time
                          for y in range(self.up_lead_time)),
                         name='upCapex1_ctr')
            
            m.addConstrs((up_capex[y] == 0
                          for y in range(self.up_lead_time, self.years)),
                         name='upCapex2_ctr')
            
            #up_dep_capex:
            
            m.addConstrs((up_dep_capex[y] == 
                          up_total_capex / self.up_dep_time
                          for y in range(self.up_lead_time,
                                         self.up_lead_time +
                                         self.up_dep_time)),
                         name='upDepCapex1_ctr')
            
            m.addConstrs((up_dep_capex[y] == 0
                          for y in range(self.up_lead_time)),
                         name='upDepCapex2_ctr')
            
            m.addConstrs((up_dep_capex[y] == 0
                          for y in range(self.up_lead_time +
                                         self.up_dep_time, self.years)),
                         name='upDepCapex3_ctr')
            
            #prod_gas:
            m.addConstrs((prod_gas[y] <= gas_needed[y]
                          for y in range(self.years)),
                         name='prodGas1_ctr')
            
            
            m.addConstrs((prod_gas[y] == 0
                          for y in range(self.up_lead_time)),
                         name='prodGas2_ctr')
            
            
            m.addConstrs((prod_gas[y] <= prod_gas_all * (1 + self.up_alpha)
                          for y in range(self.up_lead_time, self.years)),
                         name='prodGas3_ctr')
            
            m.addConstrs((prod_gas[y] >= prod_gas_all * (1 - self.up_alpha)
                          for y in range(self.up_lead_time, self.years)),
                         name='prodGas4_ctr')
            
            #e_emissions
            m.addConstrs((e_emissions[t,y] == 
                          prod_elec[t,y] * self.e_u_emissions[t]
                          for t in range(self.techs)
                          for y in range(self.years)),
                         name='eEmissions_ctr')
            
            #up_emissions
            m.addConstrs((up_emissions[y] == 
                          prod_gas[y] * self.up_u_emissions
                          for y in range(self.years)),
                         name='upEmissions_ctr')
            
            #max_emissions (starting 2030, i.e. year 7)
            if self.max_yearly_emissions is not None:
                m.addConstrs((up_emissions[y] + 
                              quicksum(e_emissions[t,y] 
                                       for t in range(self.techs))
                              <= self.max_yearly_emissions
                              for y in range(2030 - START_YEAR,self.years)),
                             name='yearlyEmissions_ctr')
                
            # PV of emissions:
                quicksum((e_capex[y] + e_fx_opex[y] + e_var_opex[y]
                                        + e_fuel_opex[y]) / ((1+self.i_e) ** (y+1))
                                       for y in range(self.years))
            pv_emissions = quicksum((up_emissions[y] + 
                                    quicksum(e_emissions[t,y] 
                                             for t in range(self.techs)))
                                    / (1+self.i_s) ** (y+1)
                                    for y in range(self.years))
                
            #min_re (starting 2030 i.e. year 7)
            if self.min_re_target is not None:                
                m.addConstrs((quicksum(prod_elec[t,y]
                                       for t in self.re_techs) >=
                              self.min_re_target * 
                              quicksum(self.d_weight[d] * 365 *
                                       self.e_demand_start[d,h] * (1 +
                                       self.e_demand_growth)**y 
                                       for d in range(self.days)
                                       for h in range(self.hours))
                              for y in range(2030 - START_YEAR,self.years)),
                             name='reTarget_ctr')
            
            
            m.optimize()
            #m.write('out.lp')
            #print(m)
            #m.computeIIS()
            #m.write('IIS.ilp')
            #print("HERE")
            #for v in added_cap.values():
            #    print('{}: {}'.format(v.varName, round(v.X,0)))
            
            self.obj = round(m.getObjective().getValue())
            self.u_npv = round(u_npv.getValue())
            self.e_npv = round(e_resid_npv.getValue() - 
                               e_costs_npv.getValue())
            self.prod_gas_all = round(prod_gas_all.X)
            
            self.total_prod_gas = total_prod_gas.getValue()
            self.total_gas_needed = total_gas_needed.getValue()
            self.pv_emissions = pv_emissions.getValue()/1000 # kilo Tonnes
            
            if(_store_results):
                
                added_cap_list = []
                prod_cap_list = []
                inst_cap_list = []
                prod_elec_list = []
                disp_cap_list = []
                e_cost_list = []
                gas_list = []
                up_list = []
                emissions_list = []
    
                for y in range(self.years):
                    added_cap_list.append([round(added_cap[t,y].X) 
                                           for t in range(self.techs)])
                    prod_cap_list.append([round(prod_cap[t,y].X) 
                                           for t in range(self.techs)])
                    inst_cap_list.append([round(inst_cap[t,y].X) 
                                           for t in range(self.techs)])
                    prod_elec_list.append([round(prod_elec[t,y].X/1000) 
                                           for t in range(self.techs)])
                    e_cost_list.append([round(e_capex[y].X), 
                                        round(e_fx_opex[y].X), 
                                        round(e_var_opex[y].X),
                                        round(e_fuel_opex[y].X)])
                    gas_list.append([round(gas_needed[y].X), 
                                     round(prod_gas[y].X)])
                    
                    up_list.append([round(prod_gas[y].X),
                                    round(up_capex[y].X),
                                    round(up_dep_capex[y].X),
                                    round(up_opex[y].X),
                                    round(royalty[y].X),
                                    round(cost_gas[y].X),
                                    round(s_profit_gas[y].X),
                                    round(c_profit_gas[y].X),
                                    round(taxable_amount[y].X),
                                    round(tax[y].X),
                                    round(up_comp_rev[y].X)])
                    
                    self.added_cap_pv_2030 = sum(added_cap[2,y].X
                                                 for y in range(8))
                    #Emissions in kilo Tonnes CO2
                    emissions_list.append([round(e_emissions[t,y].X/1000) 
                                           for t in range(self.techs)] + 
                                          [round(up_emissions[y].X/1000)])
                    
                disp_year = 7 # 2030   
                for h in range(self.hours):
                    disp_cap_list.append([round(disp_cap[t,disp_year,1,h].X)
                                          for t in range(self.techs)])
                
                self.added_cap_df = pd.DataFrame(added_cap_list, 
                                            columns=['CCGT', 'OCGT', 'PV', 
                                                     'Wind', 'HFO', 'DO', 
                                                     'Hydro', 'Unsat'
                                                     ])
                self.prod_cap_df = pd.DataFrame(prod_cap_list, 
                                            columns=['CCGT', 'OCGT', 'PV',
                                                     'Wind', 'HFO', 'DO', 
                                                     'Hydro', 'Unsat'
                                                     ])
                self.inst_cap_df = pd.DataFrame(inst_cap_list, 
                                            columns=['CCGT', 'OCGT', 'PV',
                                                     'Wind', 'HFO', 'DO', 
                                                     'Hydro', 'Unsat'
                                                     ])
                self.prod_elec_df = pd.DataFrame(prod_elec_list, 
                                            columns=['CCGT', 'OCGT', 'PV',
                                                     'Wind', 'HFO', 'DO', 
                                                     'Hydro', 'Unsat'
                                                     ]) #GWh
                self.disp_cap_df = pd.DataFrame(disp_cap_list, 
                                            columns=['CCGT', 'OCGT', 'PV',
                                                     'Wind', 'HFO', 'DO', 
                                                     'Hydro', 'Unsat'
                                                     ]) #MW
                self.disp_cap_df['Total'] = self.disp_cap_df.sum(axis=1)
                
                self.e_cost_df = pd.DataFrame(e_cost_list,
                                              columns=['capex', 'fx opex',
                                                       'var opex', 
                                                       'fuel opex'])
                self.e_cost_df['Total'] = self.e_cost_df.sum(axis=1)
                
                self.gas_df = pd.DataFrame(gas_list, 
                                           columns=['gas needed', 'gas prod'])
                
                self.up_df = pd.DataFrame(up_list,
                                          columns=['prod gas', 'capex', 
                                                   'dep capex', 'opex',
                                                   'royalty', 'cost gas',
                                                   's profit gas', 
                                                   'c profit gas',
                                                   'taxable amount','tax',
                                                   'c cashflow'])
                self.up_df['s cashflow'] = (self.up_df['royalty'] +
                                            self.up_df['s profit gas'] +
                                            self.up_df['tax'])
                
                self.emissions_df = pd.DataFrame(emissions_list,
                                                 columns=['CCGT', 'OCGT', 'PV',
                                                           'Wind','HFO', 'DO', 
                                                           'Hydro', 'Unsat',
                                                           'upstream'])
                self.emissions_df['Total'] = self.emissions_df.sum(axis=1)
                
                
                    
                #self.up_df.to_excel('up_df.xlsx')
                
                print(self.prod_elec_df)
                print(f'prod_gas_all: {round(prod_gas_all.X)}')
                print(round(u_npv.getValue()))
            
            print(f"""Objective: {round(m.getObjective().getValue())}, 
                  prod_gas: {round(prod_gas_all.X)}""")
            #print(self.disp_cap_df)
            
            #print(added_cap[2,2].X)
            
            #for v in prod_cap.values():
            #    print('{}: {}'.format(v.varName, v.X))
        
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError:
            print('Encountered an attribute error')
            
    def iterate(self, _gp_step=0.5, _spg_step=0.05,_gp_start=None,
                _spg_start=None,_gp_max=None, _emission_limit=None,
                _re_limit=None, _out_file='out_best.xlsx', _save_state=False,
                _load_state=False):
        if(_gp_start is None):
            gp_start = round(self.up_u_capex + self.up_u_opex) + _gp_step
        else:
            gp_start = _gp_start
        if(_gp_max is None):
            gp_max = round(math.ceil(max(self.import_gp))) + _gp_step
        else:
            gp_max = _gp_max + _gp_step
        if(_spg_start is None):
            spg_start = 0
        else:
            spg_start = _spg_start
        
        if(_emission_limit is not None):
            self.max_yearly_emissions = _emission_limit * THOUSAND
        else:
            self.max_yearly_emissions = None
            
        self.min_re_target = _re_limit
        best_spg = spg_start
        
        state_file = self.output_dir + 'SS_' + _out_file
        
        if _load_state and os.path.exists(state_file):
            print("!!loading state data from: " + state_file)
            states_data = pd.read_excel(state_file, sheet_name=None)
            
            gp_start = states_data['states']['curr_gp'][0]
            best_obj = int(states_data['states']['best_obj'][0])
            if best_obj == -1:
                best_obj = None
            best_gp = states_data['states']['best_gp'][0]
            best_spg = states_data['states']['best_spg'][0]
            time_begin = states_data['states']['time_begin'][0]
            pre_upstream = bool(states_data['states']['pre_upstream'][0])
            temp_iter_results_df = states_data['iter'].iloc[:,1:]
            temp_iter_results_list = temp_iter_results_df.values.tolist()
            iter_results_dict = {tuple(sublist[:2]): sublist 
                                 for sublist in temp_iter_results_list}
            
            iter_best_results_df = states_data['iter_best'].iloc[:,1:]
            iter_best_results_list = iter_best_results_df.values.tolist()
            
        else:
            best_obj = None
            best_gp = gp_start
            iter_results_dict = {}
            iter_best_results_list = []
            time_begin = timeit.default_timer()
            pre_upstream = True
        
        for gp in np.arange(gp_start, gp_max, _gp_step):
            
            if _save_state:
                if(best_obj is None):
                    states = [[gp, -1, best_gp, best_spg, time_begin, 
                               pre_upstream]]
                
                else:
                    states = [[gp, best_obj, best_gp, best_spg, time_begin, 
                              pre_upstream]]
                states_df = pd.DataFrame(states, 
                                         columns=['curr_gp', 
                                                  'best_obj', 'best_gp',
                                                  'best_spg', 'time_begin',
                                                  'pre_upstream'])
                iter_results_df = pd.DataFrame(list(iter_results_dict.values()),
                                            columns=['gp', 
                                                     'spg', 'Obj',
                                                     'u_npv', 'e_costs_npv',
                                                     'Prod Gas', 
                                                     'Total Prod Gas',
                                                     'Total Gas Needed'])
                iter_best_results_df = pd.DataFrame(iter_best_results_list,
                                            columns=['gp', 
                                                     'spg', 'Obj',
                                                     'u_npv', 'e_costs_npv',
                                                     'Prod Gas',
                                                     'Total Prod Gas',
                                                     'Total Gas Needed'])
                state_file = self.output_dir + 'SS_' + _out_file
                with pd.ExcelWriter(state_file) as f:
                    states_df.to_excel(f,sheet_name='states')
                    iter_results_df.to_excel(f,sheet_name='iter')
                    iter_best_results_df.to_excel(f,sheet_name='iter_best')
                
                
            prev_prod_gas = 0
            
            best_obj_gp = None
            best_obj_gp_spg = 0
            for spg in np.arange(spg_start, 1, _spg_step):
                self.solve(_up_gp=gp, _state_profit_gas=spg)
                if(self.prod_gas_all > EPSILON):
                    pre_upstream = False
                iter_results_dict[(gp,spg)] = [gp, spg, self.obj,-self.u_npv,
                                          -self.e_npv, self.prod_gas_all, 
                                          self.total_prod_gas,
                                          self.total_gas_needed]
                
                print(f"results for price: {gp} and spg: {spg}")
                if(best_obj is None or self.obj <= best_obj):
                    best_obj = self.obj
                    best_gp = gp
                    best_spg = spg
                    
                #Saving the best results for the current gas price
                if(best_obj_gp is None or self.obj <= best_obj_gp):
                    best_obj_gp = self.obj
                    best_obj_gp_spg = spg
                
                if(self.prod_gas_all < EPSILON and pre_upstream):
                    print(f'breaking for price: {gp} at spg: {spg}')
                    #Removed the fast track assumption below
                    #spg_start = max(spg - _spg_step, 0)
                    break
                
                if(self.prod_gas_all < EPSILON and prev_prod_gas > EPSILON):
                    print(f'breaking for price: {gp} at spg: {spg}')
                    #Removed the fast track assumption below
                    #spg_start = max(spg - _spg_step, 0)
                    break
                
                prev_prod_gas = self.prod_gas_all
                
            # Saving the "optimal" results
            iter_best_results_list.append(iter_results_dict[(gp,
                                                             best_obj_gp_spg)])
            

        
        self.iter_results_df = pd.DataFrame(list(iter_results_dict.values()),
                                            columns=['Gas Price', 
                                                     'State Profit Gas', 'Obj',
                                                     'u_npv', 'e_costs_npv',
                                                     'Prod Gas', 
                                                     'Total Prod Gas',
                                                     'Total Gas Needed'])
        self.iter_best_results_df = pd.DataFrame(iter_best_results_list,
                                            columns=['Gas Price', 
                                                     'State Profit Gas', 'Obj',
                                                     'u_npv', 'e_costs_npv',
                                                     'Prod Gas',
                                                     'Total Prod Gas',
                                                     'Total Gas Needed'])
        print(f'best gp: {best_gp}, best spg: {best_spg}')
        print('Getting best results')
        self.solve(_up_gp=best_gp, 
                   _state_profit_gas=best_spg, _store_results=True)
        
        self.run_time = timeit.default_timer() - time_begin
        
        summary_list = []
        summary_list.append([self.obj, -self.e_npv, -self.u_npv, best_gp,
                             self.prod_gas_all, best_spg, self.run_time])
        self.summary_df = pd.DataFrame(summary_list,
                                                 columns=['Obj', 'e_costs_NPV',
                                                          'u_NPV', 'Gas Price',
                                                          'Plateau Gas',
                                                          'State Profit Gas',
                                                           'Run Time'])
                                
        
        ###########################################################
        high_summary_list = []
        high_summary_list.append([self.budget[0],
                                  self.i_s,
                                  self.up_u_capex \
                                      /self.data['Up']['UnitCapex'][0],
                                  self.import_gp[0]/self.data['GP']['Mean'].to_numpy()[0],
                                  self.min_re_target,
                                  self.obj,
                                  self.prod_gas_all, 
                                  self.pv_emissions,
                                  self.added_cap_pv_2030])
        
        self.high_summary_df = pd.DataFrame(high_summary_list, 
                                    columns=['Budget', 
                                             'Discount Rate', 
                                             'Upstream Cost Factor',
                                             'NG Imp Price Factor', 
                                             'RE Target',
                                             'Obj',
                                             'Plateau Gas',
                                             'Emissions_PV',
                                             'PV_Added_2030'
                                             ])
        
        
        #######################################################
        
        summary_out = self.output_dir + _out_file
        with pd.ExcelWriter(summary_out) as f:
            self.high_summary_df.to_excel(f,sheet_name='summary')
            self.summary_df.to_excel(f,sheet_name='overview')
            self.added_cap_df.to_excel(f,sheet_name='added_cap')
            self.prod_cap_df.to_excel(f,sheet_name='prod_cap')
            self.inst_cap_df.to_excel(f,sheet_name='inst_cap')
            self.prod_elec_df.to_excel(f,sheet_name='prod_elec')
            self.disp_cap_df.to_excel(f,sheet_name='disp_cap')
            self.e_cost_df.to_excel(f,sheet_name='cost')
            self.gas_df.to_excel(f,sheet_name='gas_demand')
            self.up_df.to_excel(f,sheet_name='up')
            self.emissions_df.to_excel(f,sheet_name='emissions')
            self.iter_best_results_df.to_excel(f,sheet_name='opt_by_gp_price')
            self.iter_results_df.to_excel(f,sheet_name='all_iterations')
            
