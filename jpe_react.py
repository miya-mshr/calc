# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:25:29 2022
#%% 
@author: jet00107
"""
#%%
import chemicals
from chemicals import search_chemical, heat_capacity, Poling, Poling_integral, critical, viscosity, thermal_conductivity, phase_change, reaction
import numpy as np
import pandas as pd
import copy
from scipy import interpolate
import math
import matplotlib.pyplot as plt
#import weakref

#%%　AC比熱
def cp_ac(temp1,temp2=0):
    # temp2　から temp1 間の　ACの平均比熱を求める。温度の単位は℃、比熱の単位は kJ/kgK
    #　temp2　のデフォルトは0℃。　
    # calc_ac_hinetu(200) は0℃から200℃までの平均比熱
    # calc_ac_hinetu(200,100) は100℃から200℃までの平均比熱
    A = 0.1522
    B = 0.0007392
    C = -0.0000007272
    D = 0.0000000002966
    cp1 = (A * temp1 + 1/2 * B * temp1 ** 2 + 1/3 * C * temp1 ** 3 + 1/4 * D * temp1 ** 4)
    cp2 = (A * temp2 + 1/2 * B * temp2 ** 2 + 1/3 * C * temp2 ** 3 + 1/4 * D * temp2 ** 4)
    cp_ac = (cp1 - cp2) / (temp1 - temp2) * 4.1868
    return cp_ac
 
#%%　ガス平均比熱
def Cp_gas_mean(chem,T1,T2=0):
    # T2 ～　T1　間のガスの平均比熱を算出します
    # chem に物質名,例えば 'N2' を、T1,T1には温度（℃）を入力してください。T2は省略可能。
    # 'n2' など小文字でもOK
    # 例　　Cp_gas_mean('N2',30,0)
    #　単位は J/molK, kJ/kgK, kJ/m3NK をリストで出力します。
    # kJ/m3NK のみ欲しい場合は、Cp_gas_mean('N2',30,0)[2] としてください。
    try:
        Cp_df = chemicals.heat_capacity.Cp_data_Poling
        chem = search_chemical(chem)
        coeffs = Cp_df.loc[chem.CASs,'a0':'a4']
        Cp_gas_mean_mol = (Poling_integral(T1+273.15, *coeffs)-Poling_integral(T2+273.15, *coeffs))/(T1-T2)
        Cp_gas_mean_kg = (Poling_integral(T1+273.15, *coeffs)-Poling_integral(T2+273.15, *coeffs))/(T1-T2)/chem.MW
        Cp_gas_mean_m3 = (Poling_integral(T1+273.15, *coeffs)-Poling_integral(T2+273.15, *coeffs))/(T1-T2)/22.4
        return Cp_gas_mean_mol, Cp_gas_mean_kg, Cp_gas_mean_m3
    except ValueError as e:
        print(e)
    except KeyError as k:
        print(chem, k)

# ガス比熱
def Cp_gas(chem,T):
    try:
        Cp_df = chemicals.heat_capacity.Cp_data_Poling
        chem = search_chemical(chem)
        coeffs = Cp_df.loc[chem.CASs,'a0':'a4']
        Cp_gas_mol = Poling(T+273.15, *coeffs)
        Cp_gas_kg = Poling(T+273.15, *coeffs)/chem.MW
        Cp_gas_m3 = Poling(T+273.15, *coeffs)/22.4
        return Cp_gas_mol, Cp_gas_kg, Cp_gas_m3
    
    except ValueError as e:
        print(e)
    except KeyError as k:
        print(chem, k)
    
    return Cp_gas

#
def strm_Cp_gas(**strm): 
    strm['Sosei_Cp_mol'] = {}
    for k, v in strm['Sosei_wet'].items():
        strm['Sosei_Cp_mol'][k] = Cp_gas(k,strm['T'])[0]
    return strm

#
def strm_Cp_gas_mix(**strm):
    ys = pd.DataFrame(strm['Sosei_wet'].values(),index=strm['Sosei_wet'].keys(),columns=['y'])
    MWs = pd.DataFrame(strm['Sosei_MW'].values(),index=strm['Sosei_MW'].keys(),columns=['MW'])
    Cps = pd.DataFrame(strm['Sosei_Cp_mol'].values(),index=strm['Sosei_Cp_mol'].keys(),columns=['Cp'])

    strm['Cp_mix_mol'] = sum(list(ys.y * Cps.Cp))
    strm['MW_mix'] = sum(list(ys.y * MWs.MW))
    
    return strm
    
def strm_entropy_gas_mix(**strm):
    strm['Entropy_mix'] = (strm['Cp_mix_mol'] * np.log(strm['T']+273.15) - 8.3145 * np.log(strm['P']+101.325))*strm['vol_flow_normal']/22.4
    
    return strm


#%% ガス熱伝導率 w/[mk]
def k_gas(chem, T):
    Cp_df = chemicals.heat_capacity.Cp_data_Poling
    MW = search_chemical(chem).MW
    coeffs = Cp_df.loc[search_chemical(chem).CASs,'a0':'a4']
    R = 8.3143
    
    Cp = heat_capacity.Poling(T+273.15, *coeffs)
    Cv = Cp - R
    mu = mu_gas(chem,T)
    
    k_gas = thermal_conductivity.Eucken(MW, Cv, mu)
        
    return k_gas

def k_gas_modified(chem, T):
    Cp_df = chemicals.heat_capacity.Cp_data_Poling
    MW = search_chemical(chem).MW
    coeffs = Cp_df.loc[search_chemical(chem).CASs,'a0':'a4']
    R = 8.3143
    
    Cp = heat_capacity.Poling(T+273.15, *coeffs)
    Cv = Cp - R
    mu = mu_gas(chem,T)
    
    k_gas = chemicals.thermal_conductivity.Eucken_modified(MW, Cv, mu)
        
    return k_gas


def strm_k_gas_modified(**strm):
    strm['Sosei_k_gas'] = {}
    for k, v in strm['Sosei_wet'].items():
        strm['Sosei_k_gas'][k] = k_gas_modified(k,strm['T'])    
    
    return strm


def strm_k_gas_modified_mix(**strm):
    ys = pd.DataFrame(strm['Sosei_wet'].values(),index=strm['Sosei_wet'].keys(),columns=['y'])
    mus = pd.DataFrame(strm['Sosei_mu'].values(),index=strm['Sosei_mu'].keys(),columns=['mu'])
    MWs = pd.DataFrame(strm['Sosei_MW'].values(),index=strm['Sosei_MW'].keys(),columns=['MW'])
    ks = pd.DataFrame(strm['Sosei_k_gas'].values(),index=strm['Sosei_k_gas'].keys(),columns=['k'])
    Tbs = pd.DataFrame(strm['Sosei_Tb'].values(),index=strm['Sosei_Tb'].keys(),columns=['Tb'])
    T = strm['T']
    
    df = pd.merge(ys, mus, left_index=True,right_index=True,how = 'outer')
    df = pd.merge(df, MWs, left_index=True,right_index=True,how = 'outer')
    df = pd.merge(df, ks, left_index=True,right_index=True,how = 'outer')
    df = pd.merge(df, Tbs, left_index=True,right_index=True,how = 'outer')
    
    strm['k_gas_mix'] = thermal_conductivity.Lindsay_Bromley(T, df.y, df.k, df.mu, df.Tb, df.MW)
    
    return strm
    



#%% ガス粘性係数 (Pa.s)
def mu_gas(chem,T):
    Tc = critical.Tc(search_chemical(chem).CASs)
    Pc = critical.Pc(search_chemical(chem).CASs)
    MW = search_chemical(chem).MW
    
    mu_gas = viscosity.Yoon_Thodos(T+273.15, Tc, Pc, MW)
    # viscosity.Yoon_Thodos(T, Tc, Pc, MW))
    return mu_gas

#%% ストリームクラスの粘性係数
# ガス単体
def strm_mu_gas(**strm):
    strm['Sosei_mu'] = {}
    for k, v in strm['Sosei_wet'].items():
        strm['Sosei_mu'][k] = mu_gas(k,strm['T'])
    
    return strm

# 混合
def strm_mu_gas_mix(**strm):
    ys = pd.DataFrame(strm['Sosei_wet'].values(),index=strm['Sosei_wet'].keys(),columns=['y'])
    mus = pd.DataFrame(strm['Sosei_mu'].values(),index=strm['Sosei_mu'].keys(),columns=['mu'])
    MWs = pd.DataFrame(strm['Sosei_MW'].values(),index=strm['Sosei_MW'].keys(),columns=['MW'])
    
    df = pd.merge(ys, mus, left_index=True,right_index=True,how = 'outer')
    df = pd.merge(df, MWs, left_index=True,right_index=True,how = 'outer')
    
    strm['mu_mix'] = viscosity.Wilke(df.y, df.mu, df.MW)
    return strm
        
        
#%% ストリームクラスにモル重量を追加
def strm_add_MW(**strm):
    strm['Sosei_MW'] = {}
    for k, v in strm['Sosei_wet'].items():
        strm['Sosei_MW'][k] = search_chemical(k).MW
    
    return strm    
    
#%% ストリームクラスに沸点を追加
def strm_add_Tb(**strm):
    strm['Sosei_Tb'] = {}
    for k, v in strm['Sosei_wet'].items():
        strm['Sosei_Tb'][k] = phase_change.Tb(search_chemical(k).CASs)
    
    return strm    


     
#%% ストリームクラスに mass_flow, vol_flow, rho などを追加
def strm_add_spec(**strm):
    strm['Sosei_mass_flow'] = {}
    strm['mass_flow'] = 0
    for k, v in strm['Sosei_vol_flow'].items():
        strm['Sosei_mass_flow'][k] = v / 22.4 * search_chemical(k).MW
        strm['mass_flow'] += strm['Sosei_mass_flow'][k]
    
    strm['vol_flow_normal'] = 0
    for v in strm['Sosei_vol_flow'].values():
        strm['vol_flow_normal'] += v
    
    
    strm['vol_flow_actual'] = strm['vol_flow_normal'] * (strm['T'] + 273.15)/ 273.15 * 101.325 / (101.325 + strm['P'])
    
    strm['rho_normal'] = strm['mass_flow'] / strm['vol_flow_normal']
    strm['rho_actual'] = strm['mass_flow'] / strm['vol_flow_actual']
    
    return strm    
    

     
  
#%%　ガスエンタルピ
def h_gas(chem,T1,T2=0):
    # T2 ～　T1　間のガスの平均比熱を算出します
    # chem に物質名,例えば 'N2' を、T1,T1には温度（℃）を入力してください。T2は省略可能。
    # 'n2' など小文字でもOK
    # 例　  h_gas('N2',30,0)
    #　単位は J/molK, kJ/kgK, kJ/m3NK をリストで出力します。
    # kJ/m3NK のみ欲しい場合は、Cp_gas('N2',30,0)[2] としてください。
    try:
        chem = search_chemical(chem)
        Cp_df = chemicals.heat_capacity.Cp_data_Poling
    
        coeffs = Cp_df.loc[chem.CASs,'a0':'a4']
        
        h_gas_mol = (Poling_integral(T1+273.15, *coeffs)-Poling_integral(T2+273.15, *coeffs))
        h_gas_kg = (Poling_integral(T1+273.15, *coeffs)-Poling_integral(T2+273.15, *coeffs))/chem.MW
        h_gas_m3 = (Poling_integral(T1+273.15, *coeffs)-Poling_integral(T2+273.15, *coeffs))/22.4
        
        return h_gas_mol, h_gas_kg, h_gas_m3
    
    except ValueError as e:
        print(e)
        
    except KeyError as k:
        print(chem, k)
 
#%% ストリームのエンタルピを算出する       
# strm01 = {'T':126, 'P': 0, 'Flow':1800000, 'Sosei_dry' : {'H2O' : 0.08, 'N2' : 0.807, 'O2' : 0.043, 'CO2' : 0.15}}
# 引数は辞書型
# 温度は℃、圧力は無関係、流量はm3N/h、組成はH2O以外はDryベース
#　単位があった方がいいのか。
 
def strm_h_gas(**strm):
    try:
        strm['Sosei_h'] = {}
        #x = []
        strm['h'] = 0
        for k, v in strm['Sosei_vol_flow'].items():
#            if k == 'H2O':
            strm['Sosei_h'][k] = h_gas(k, strm['T'])[2] * v / 1000
                #x.append(h_gas(k, strm['T'])[2] * v)
#            else:
#               strm['Sosei_h'][k] = h_gas(k, strm['T'])[2] * v * (1-strm['Sosei_dry']['H2O'])
                #x.append(h_gas(k, strm['T'])[2] * v * (1-strm['Sosei_dry']['H2O']))   
            strm['h'] += strm['Sosei_h'][k]    
    
    
    except TypeError:
        pass
 
    #h_x = sum(x) * strm['Flow'] / 1000
 
    #return h_x
    return strm
 

#%% ﾄﾞﾗｲﾍﾞｰｽの組成からウェットベースを追加
 
def strm_dry_to_wet(**strm):
    strm['Sosei_wet'] = {}
    for k, v in strm['Sosei_dry'].items():
        if k != 'H2O':
            strm['Sosei_wet'][k] = strm['Sosei_dry'][k] * (1-strm['Sosei_dry']['H2O'])
        else:
            strm['Sosei_wet'][k] = strm['Sosei_dry'][k]
    return strm

#%% ウェットﾍﾞｰｽの組成からドライベースを追加
 
def strm_wet_to_dry(**strm):
    strm['Sosei_dry'] = {}
    for k, v in strm['Sosei_wet'].items():
        if k != 'H2O':
            strm['Sosei_dry'][k] = strm['Sosei_wet'][k] / (1-strm['Sosei_wet']['H2O'])
        else:
            strm['Sosei_dry'][k] = strm['Sosei_wet'][k]
    return strm
 
#%% 組成から各成分の流量を計算
def strm_dry_to_vol_flow(**strm):
    strm['Sosei_vol_flow'] = {}
    for k, v in strm['Sosei_dry'].items():
        if k != 'H2O':
            strm['Sosei_vol_flow'][k] = strm['Sosei_dry'][k] * (1-strm['Sosei_dry']['H2O']) * strm['Flow']
        else:
            strm['Sosei_vol_flow'][k] = strm['Sosei_dry'][k] * strm['Flow']
    return strm

#%% 各成分の流量から組成を求める
def strm_flow_to_wet(**strm):
    strm['Sosei_wet'] = {}
    F = 0
    for v in strm['Sosei_vol_flow'].values():
        F += v
    for k, v in strm['Sosei_vol_flow'].items():
        strm['Sosei_wet'][k] = v / F
    return strm

def set_NH3_flow(mol_ratio, **strm):
    initial_ammonia_gas = {'T': 20., 'P': 10.0, 'Flow':1000,'Sosei_dry':{'H2O':0,'NH3':100/100}}
    strm_flue_gas = GasStream(**strm)
    initial_ammonia_gas['Flow'] = strm_flue_gas.strm['Sosei_vol_flow']['SO2'] * mol_ratio
    strm_ammonia_gas = GasStream(**initial_ammonia_gas)
    return strm_ammonia_gas
 
def set_NH3_dilution_air(**strm):
    initial_air = {'T': 20., 'P': 10.0, 'Flow':1000,'Sosei_dry':{'H2O':2/100,'N2':79/100,'O2':21/100}}
    initial_air['Flow'] = strm['Sosei_vol_flow']['NH3'] * 29
    strm_ammonia_dilution_air = GasStream(**initial_air)
    return strm_ammonia_dilution_air

def set_No2_out_gas_spec(T,P,**Sosei):
    No2_out_gas_spec = {'T':T,'P':P,'Sosei_vol_flow':Sosei}
    del(No2_out_gas_spec['Sosei_vol_flow']['HCl'])
    del(No2_out_gas_spec['Sosei_vol_flow']['HF'])
    No2_out_gas_spec['Sosei_vol_flow']['H2O'] = 0
    
    flow = 0
    for val in No2_out_gas_spec['Sosei_vol_flow'].values():
        flow += val
    
    P_sat = MoistAir().f1d_p_sat(T)
    No2_out_gas_spec['Sosei_vol_flow']['H2O'] = flow * P_sat /(101.325 + P - P_sat)
    
    return No2_out_gas_spec

def set_air_spec(P_atm,T_atm,ϕ_atm,flow):
    MC_seal_air_spec = {'T':T_atm,'P':0,'Sosei_vol_flow':{'H2O':0,'N2':0,'O2':0}}
    MC_seal_air_flow = flow
    Ps_atm = MoistAir().f1d_p_sat(T_atm) #大気飽和蒸気圧 kPa
    Pw_atm = Ps_atm * ϕ_atm # 大気蒸気圧 kPa
    x_atm = Pw_atm/(P_atm-Pw_atm) # 大気絶対湿度 m3N/m3N'
    MC_seal_air_spec['Sosei_vol_flow']['H2O'] = MC_seal_air_flow * x_atm
    MC_seal_air_spec['Sosei_vol_flow']['N2'] = MC_seal_air_flow * 0.79
    MC_seal_air_spec['Sosei_vol_flow']['O2'] = MC_seal_air_flow * 0.21
    return MC_seal_air_spec

class Hex_shell_and_tube:
    def __init__(self, number, name):
        self.number = number
        self.name = name
        self.type = 'counter_flow' #'parallel_flow'
        self.tube_side = {'fluid':'air','T_in':300,'T_out':100}
        self.shell_side =  {'fluid':'air','T_in':50,'T_out':200}
    
    def calc_LMTD(self):
        if self.type == 'counter_flow':
            Th = self.tube_side['T_in'] - self.shell_side['T_out']
            Tc = self.tube_side['T_out'] - self.shell_side['T_in']
        else:
            Th = self.tube_side['T_in'] - self.shell_side['T_in']
            Tc = self.tube_side['T_out'] - self.shell_side['T_out']
        try:
            self.LMTD = (Th - Tc) / math.log(Th / Tc)
        except ValueError as v:
            print(v,'LMTDを計算できません。温度を確認してください。')

class MoistAir:
    def __init__(self):
        temperature = np.array([-20,-15,-10,-5,0,\
                                2,4,6,8,10,\
                                12,14,16,18,20,\
                                22,24,26,28,30,\
                                32,34,36,38,40,\
                                45,50,55,60,\
                                65,70,75,80])

        p_sat = np.array([0.125,0.191,0.287,0.421,0.611,\
                          0.705,0.813,0.935,1.072,1.227,\
                          1.402,1.597,1.814,2.062,2.337,\
                          2.642,2.850,3.361,3.780,4.244,\
                          4.756,5.319,5.942,6.626,7.378,\
                          9.586,12.34,15.75,19.93,\
                          25.01,31.17,38.56,47.37])

        x_sat = np.array([0.00077,0.00118,0.00176,0.00260,0.00377,\
                          0.00436,0.00503,0.00579,0.00665,0.00763,\
                          0.00872,0.00996,0.01136,0.01293,0.01469,\
                          0.01666,0.01887,0.02134,0.02410,0.02718,\
                          0.0306,0.0345,0.0387,0.0435,0.0488,\
                          0.0645,0.0862,0.1144,0.1523,\
                          0.2039,0.2763,0.3820,0.5460])

        self.f1d_p_sat = interpolate.interp1d(temperature, p_sat, kind = 'cubic')
        self.f1d_x_sat = interpolate.interp1d(temperature, x_sat, kind = 'cubic')
    
    def plot_figure(self,T):
        plt.rcParams['font.family']='IPAexGothic'
        t_new = np.linspace(-20,80, 200)
        p_sat_f1d = self.f1d_p_sat(t_new)
        x_sat_f1d = self.f1d_x_sat(t_new)
        
        fig = plt.figure(figsize =(10,5))
        fig.suptitle('飽和湿り空気の性質', fontsize = 16)
        
        ax1 = fig.add_subplot(121)
        ax1.plot(t_new, p_sat_f1d)
        ax1.plot([T,T],[0,self.f1d_p_sat(T)],'r')
        ax1.plot([-20,T],[self.f1d_p_sat(T),self.f1d_p_sat(T)],'r')
        ax1.set_xlabel('温度 ℃')
        ax1.set_ylabel('飽和圧力 kPa')
        plt.grid()
        
        ax2 = fig.add_subplot(122)
        ax2.plot(t_new, x_sat_f1d)
        ax2.plot([T,T],[0,self.f1d_x_sat(T)],'r')
        ax2.plot([-20,T],[self.f1d_x_sat(T),self.f1d_x_sat(T)],'r')
        ax2.set_xlabel('温度 ℃')
        ax2.set_ylabel(r"絶対湿度 kg/kg'")
        plt.grid()
        
        plt.show()



#%% 気体ストリームクラス

class GasStream:
    #gas_stream_class_list = []
    #references = []
    def __init__(self,**strm):
        self.strm_num = ''
        self.strm_name = ''
        self.strm = strm
        self.check_strm()
        self.calc_strm()
        #print(__class__.__name__, 'クラス生成')
        #GasStream.gas_stream_class_list.append(self)
        #GasStream.references.append(weakref.ref(self))

    def check_strm(self):
        if 'Sosei_dry' in self.strm:
            total = 0
            for k,v in self.strm['Sosei_dry'].items():
                if k == 'H2O':
                    pass
                else:
                    total += v
            if total != 1:
                self.strm['Sosei_dry']['N2'] += 1 - total
                #print(total)
            else:
                pass
        
        elif 'Sosei_wet' in self.strm:
            total = 0
            for k,v in self.strm['Sosei_wet'].items():
                total += v
            if total != 1:
                self.strm['Sosei_wet']['N2'] += 1 - total
                #print(total)
            else:
                pass
            
        elif 'Sosei_vol_flow' in self.strm:
            self.strm['Flow'] = 0
            for v in self.strm['Sosei_vol_flow'].values():
                self.strm['Flow'] += v        
        return
            


    def calc_strm(self):
        if 'Sosei_dry' in self.strm:
            self.strm = strm_dry_to_wet(**self.strm)
            self.strm = strm_dry_to_vol_flow(**self.strm)
        elif 'Sosei_wet' in self.strm:
            self.strm = strm_wet_to_dry(**self.strm)
            self.strm = strm_dry_to_vol_flow(**self.strm)
        elif 'Sosei_vol_flow' in self.strm:
            self.strm = strm_flow_to_wet(**self.strm)
            self.strm = strm_wet_to_dry(**self.strm)
        else:
            print('no Sosei Data')
        
        self.strm = strm_add_spec(**self.strm)
        self.strm = strm_add_MW(**self.strm)
        self.strm = strm_h_gas(**self.strm)
        self.strm = strm_Cp_gas(**self.strm)
        self.strm = strm_Cp_gas_mix(**self.strm)
        self.strm = strm_entropy_gas_mix(**self.strm)
        self.strm = strm_mu_gas(**self.strm)
        self.strm = strm_add_Tb(**self.strm)
        self.strm = strm_mu_gas_mix(**self.strm)
        self.strm = strm_k_gas_modified(**self.strm)
        self.strm = strm_k_gas_modified_mix(**self.strm)
        self.strm['nu'] = self.strm['mu_mix']/self.strm['rho_actual']
        self.strm['alpha'] = self.strm['k_gas_mix']/self.strm['rho_actual']/(self.strm['Cp_mix_mol']/self.strm['MW_mix']*1000)
        self.strm['Pr'] = self.strm['nu']/self.strm['alpha']
        self.strm['Cv_mix_mol'] = self.strm['Cp_mix_mol'] - 8.3145
        self.strm['kappa'] = self.strm['Cp_mix_mol'] / self.strm['Cv_mix_mol']
  
 
#%% ストリームミキサークラス

class Stream_Mixer:
    def __init__(self,strm1,strm2):
        self.strm1 = copy.copy(strm1.strm)
        self.strm2 = copy.copy(strm2.strm)
        self.strm = self.calc_mixing().strm
    
    def calc_mixing(self):
        strm_spec = self.strm1
        strm_mix = GasStream(**strm_spec)
        for ka, va in self.strm1['Sosei_vol_flow'].items():
            for kb, vb in self.strm2['Sosei_vol_flow'].items():
                if ka == kb:
                    strm_mix.strm['Sosei_vol_flow'][ka] = va + vb
                else:
                    pass
        for kb, vb in self.strm2['Sosei_vol_flow'].items():
            if kb in self.strm1['Sosei_vol_flow'].keys():
                pass
            else:
                strm_mix.strm['Sosei_vol_flow'][kb] = vb
     
        strm_mix.strm['Sosei_wet']  = strm_flow_to_wet(**strm_mix.strm)['Sosei_wet']   
        strm_mix.strm['Sosei_dry']  = strm_wet_to_dry(**strm_mix.strm)['Sosei_dry']
        strm_mix.strm['Flow'] = self.strm1['Flow'] + self.strm2['Flow']
     
        T_list = list(np.linspace(-273.15,800,2000))
        res_list = []
     
        for i in T_list:
            strm_mix.strm['T'] = i
            h_temp = strm_h_gas(**strm_mix.strm)['h']
            res_list.append((self.strm1['h'] + self.strm2['h'] - h_temp)**2)
     
        strm_mix.strm['T'] = T_list[res_list.index(min(res_list))]
        strm_mix.strm['P'] = min(self.strm1['P'], self.strm2['P'])
        strm_mix.calc_strm()
        #print(self.strm1.__class__.__name__,'+' ,self.strm2.__class__.__name__)
     
        return strm_mix
    
# ## 脱硫塔クラス
class ReACT_Adsorber:
    def __init__(self, number, name):
        self.number = number
        self.name = name
        self.inlet_strm = {}
        self.outlet_strm = {}
        self.η_and_out = {'η_SO2'  : 98/100,\
                          'η_SO3'  :100/100,\
                          'η_NOx'  : 50/100,\
                          'η_Dust' : 70/100,\
                          'η_HCl'  : 50/100,\
                          'η_HF'   : 50/100,\
                          'out_NH3': 3e-6}
    
    def sulfuric_acid_production(self,SO2,SO3):
        SO2_to_H2SO4 = {'SO2':1,'O2':1/2,'H2O':1,'H2SO4':-1}
        SO3_to_H2SO4 = {'SO3':1,'H2O':1,'H2SO4':-1}
        H2SO4_SO2 = - SO2_to_H2SO4['H2SO4']/SO2_to_H2SO4['SO2'] * SO2
        H2SO4_SO3 = - SO3_to_H2SO4['H2SO4']/SO3_to_H2SO4['SO3'] * SO3
        H2SO4 = H2SO4_SO2 + H2SO4_SO3
        O2 = - SO2_to_H2SO4['O2'] * SO2
        H2O = - (SO2_to_H2SO4['H2O'] * SO2 + SO3_to_H2SO4['H2O'] * SO3)
        return H2SO4, O2, H2O
        
    def ammonium_sulfate_bisulfate_production(self,H2SO4,NH3,T):
        Ts = np.linspace(100,200,50)
        NH5SO4  = np.array([10**( -81530/4.574/(273+T)+5.25*math.log10(273+Temperature)+10.5) for Temperature in Ts])
        N2H8SO4 = np.array([10**(-107510/4.574/(273+T)+7.00*math.log10(273+Temperature)+13.8) for Temperature in Ts])
        NH5SO4_rate  = (N2H8SO4/NH5SO4/(4.5e-6))/(1+(N2H8SO4/NH5SO4/(4.5e-6)))
        N2H8SO4_rate = 1-NH5SO4_rate
        f1d_NH5SO4  = interpolate.interp1d(Ts,  NH5SO4_rate, kind = 'cubic')
        f1d_N2H8SO4 = interpolate.interp1d(Ts, N2H8SO4_rate, kind = 'cubic')
        NH5SO4  = min(NH3 * f1d_NH5SO4(T) /(f1d_NH5SO4(T) + 2*f1d_N2H8SO4(T)), f1d_NH5SO4(T) * H2SO4)
        N2H8SO4 = min(NH3 * f1d_N2H8SO4(T)/(f1d_NH5SO4(T) + 2*f1d_N2H8SO4(T)),f1d_N2H8SO4(T) * H2SO4)
        return NH5SO4, N2H8SO4
    
    def set_adsorber_size(self,**adsorber_size):
        self.adsorber_size = adsorber_size
        self.adsorber_size['AC_Volume'] = self.adsorber_size['Height']*\
                                          self.adsorber_size['Width']*\
                                          self.adsorber_size['Depth']*\
                                          self.adsorber_size['Num_of_Cartridges']
        self.adsorber_size['AC_Area']   = self.adsorber_size['Height']*\
                                          self.adsorber_size['Width']*\
                                          self.adsorber_size['Num_of_Cartridges']

    def set_adsorber_inlet_condition(self,**strm):
        self.inlet_strm = strm
        
        
    def calc_DeSOx_Efficiency(self,SO2L_in=6,RT=170,SO2_index=0.9):
        #"""READ Adsorption Table""" 
        table=np.loadtxt('table.csv',delimiter=',',skiprows=1,dtype=float)

        #"""Parameter"""
        ACth     = self.adsorber_size['Depth'] * 1000       #2000"""AC Bed Depth [mm]"""
        so2in    = self.inlet_strm['Sosei_dry']['SO2'] * 1000000  #500.0"""Flue Gas SO2 [ppm-dry]"""
        h2oin    = self.inlet_strm['Sosei_wet']['H2O'] * 100      #10.0"""Flue Gas H2O [%-wet]"""
        o2in     = self.inlet_strm['Sosei_dry']['O2'] * 100       #5.0"""Flue Gas O2 [%-dry]"""
        Tempin   = self.inlet_strm['T']                 #150.0"""Flue Gas Temperature [degC]"""
        lvin     = self.inlet_strm['Flow']/3600/self.adsorber_size['AC_Area']   #0.20"""Linear Velocity [mN/s]"""
        so2Lin   = SO2L_in/(0.2*0.36*0.93/22.4*64/0.65) 
                                             #2.0/(0.2*0.36*0.93/22.4*64/0.65)"""SO2 Adsorption [mg/g-aC-->ppm/cm]""" 
        RTime    = RT                #80"""Retention Time [h]"""
        SO2index = SO2_index         #1.0"""DeSOx index  [-]"""
        ACmax    = int(ACth/10)              #"""AC Bed Depth [cm]"""
        
        self.SV = self.inlet_strm['Flow']/self.adsorber_size['AC_Volume']
        self.RT = RT
        self.rho_AC = 0.65
        self.AC_mass_flow_rate = self.adsorber_size['AC_Volume'] * self.rho_AC / self.RT
        self.AC_vol_flow_rate = self.adsorber_size['AC_Volume'] / self.RT 

        #""" """
        h2o=np.zeros(RTime+1)
        o2=np.zeros(RTime+1)
        Temp=np.zeros(RTime+1)
        lv=np.zeros(RTime+1)
        acad=np.zeros((RTime+2, ACmax+1))
        so2=np.zeros((RTime+2, ACmax+1))

        table = SO2index * table

        #"""H2O=7% Base"""
        Bh2o = 7.0    

        #"""inisialize """
        for i in range(0,RTime+1):
            j = 0
            h2o[i] = h2oin
            so2[i, j] = so2in
            o2[i] = o2in
            Temp[i] = Tempin
            lv[i] = lvin

        for j in range(0,ACmax):
            acad[0, j] = so2Lin


        #"""Calculation"""
        for i  in range(0,RTime+1):
            for j in range(0,ACmax):        
                """SO2換算　H2O＝7%ベース"""
                so2D = so2[i, j]
                so2W = so2D * (100 - h2o[i]) / 100
                so2B = so2W / ((100 - Bh2o) / 100)

                """O2換算　H2O＝7%ベース"""
                o2D = o2[i]
                o2W = o2D * (100 - h2o[i]) / 100
                o2B = o2W / ((100 - Bh2o) / 100)

                """残存SO2量"""
                so2L = acad[i, j]

                """行列番号取得"""


                """SO2濃度650ppm超の処理"""
                if so2B >= 630:
                    so2row = int(so2L / 20) 
                    so2col = int(630 / 20) 

                    r = table[so2row, int(630/20)]


                else:
                    so2row = int(so2L / 20) 
                    so2col = int(so2B / 20) 

                    r0 = table[so2row, so2col]
                    r1 = table[so2row, so2col + 1]
                    r2 = table[so2row + 1, so2col]
                    r = (r1-r0)*(so2B-so2col*20)/20+(r2-r0)*(so2L-so2row*20)/20+r0

                """O2依存性"""
                if o2B < 3:
                    o2prm = 0.94805
                elif o2B < 5:
                    o2prm = 0.025974 * o2B + 0.87013
                elif o2B < 7:
                    o2prm = 0.038961 * o2B + 0.8052
                else:
                    o2prm = 1.07792

                """水分依存性"""
                if h2o[i] < 7:
                    h2oprm = 1
                elif h2o[i] < 9:
                    h2oprm = 0.038961 * h2o[i] + 0.72727
                elif h2o[i] < 11:
                    h2oprm = 0.032467 * h2o[i] + 0.78571
                else:
                    h2oprm = 1.14286

                """温度依存性"""
                if Temp[i] < 120:
                    Tprm = 1.07792
                elif Temp[i] < 140:
                    Tprm = -0.0038961 * Temp[i] + 1.54545
                elif Temp[i] < 160:
                    Tprm = -0.0012987 * Temp[i] + 1.18182
                elif Temp[i] < 190:
                    Tprm = -0.0004329 * Temp[i] + 1.04329
                else:
                    Tprm = 0.96104

                """O2、水分、温度補正後脱硫速度計算"""
                SO2Rate = o2prm * h2oprm * Tprm * r / 5 * 0.635
                #SO2Rate = o2prm * h2oprm * Tprm * r          
                """出口SO2濃度算出"""
                so2bb = so2B - SO2Rate * 0.2 / lv[i]

                """出口SO2濃度＜０の処理"""
                if so2bb < 0:
                    so2bb = 0

                """SO2濃度をDRYベースに換算"""
                so2db = so2bb * ((100 - Bh2o) / 100) / ((100 - h2o[i]) / 100)

                """SO2濃度を次のACへ渡す（層幅方向）"""
                so2[i, j + 1] = so2db

                """SO2吸着量を次のACへ渡す（層高さ方向）"""
                acad[i + 1, j] = acad[i, j] + SO2Rate


        #"""脱硫塔出口SO2濃度総和算出"""
        so2sum = so2[:RTime+1,ACmax]
        Sum = np.sum(so2sum)

        #"""積分平均脱硫効率算出 '小数点第３位切り捨て"""
        EffSO2 = (int((so2in - Sum/RTime) / so2in * 1000)) / 10
        #print("DeSOx Efficiency = " + str(EffSO2)+" %")
        self.η_and_out['η_SO2'] = EffSO2/100
        self.SO2_adsorption_distribution = acad *(0.2*0.36*0.93/22.4*64/0.65) 
        self.SO2_ppm_distribution = acad
        return self.η_and_out['η_SO2']



    def set_η_and_out(self,**η_and_out):
        self.η_and_out = η_and_out
           
    def calc_adsorber_reaction(self, **strm):
        #self.inlet_strm = strm
        # 硫酸生成
        SO2 = strm['Sosei_vol_flow']['SO2'] * self.η_and_out['η_SO2']
        SO3 = strm['Sosei_vol_flow']['SO3'] * self.η_and_out['η_SO3']
        NOx = strm['Sosei_vol_flow']['Nitric Oxide'] * self.η_and_out['η_NOx']
        HCl = strm['Sosei_vol_flow']['HCl'] * self.η_and_out['η_HCl']
        HF = strm['Sosei_vol_flow']['HF'] * self.η_and_out['η_HF']
        Dust = strm['Dust'] * self.η_and_out['η_Dust']
        
        H2SO4, O2, H2O = self.sulfuric_acid_production(SO2,SO3)
        # 硫安生成
        NH3 = strm['Sosei_vol_flow']['NH3']
        T = strm['T']
        NH5SO4, N2H8SO4 = self.ammonium_sulfate_bisulfate_production(H2SO4,NH3,T)
        # NH3と反応する硫酸
        H2SO4_NH3 = NH5SO4 + N2H8SO4
        # ACに吸着する硫酸
        H2SO4_AC = H2SO4 - H2SO4_NH3
        # 排ガス中水分濃度
        H2O_content = strm['Sosei_wet']['H2O']
        # 吸着硫酸濃度
        H2SO4_noudo = ((0.1969*T+44.332)*(H2O_content)**-0.055)/100
        # 吸着硫酸量 kg
        H2SO4_AC_kg = H2SO4_AC/22.4*98
        # 吸着水分量 kg
        H2O_AC_kg = H2SO4_AC_kg/H2SO4_noudo - H2SO4_AC_kg
        # 吸着水分量
        H2O_AC = H2O_AC_kg/18*22.4
        # 吸着物質
        self.AC_adsorption = {'H2SO4':H2SO4_AC,'H2O':H2O_AC,'NH4HSO4':NH5SO4,'(NH4)2SO4':N2H8SO4,'HCl':HCl,'HF':HF}
        
        self.outlet_strm['Sosei_vol_flow'] = {}
        self.outlet_strm['Sosei_vol_flow']['H2O'] = strm['Sosei_vol_flow']['H2O'] - H2O
        self.outlet_strm['Sosei_vol_flow']['N2'] = strm['Sosei_vol_flow']['N2']
        self.outlet_strm['Sosei_vol_flow']['O2'] = strm['Sosei_vol_flow']['O2'] - O2
        self.outlet_strm['Sosei_vol_flow']['CO2'] = strm['Sosei_vol_flow']['CO2']
        self.outlet_strm['Sosei_vol_flow']['SO2'] = strm['Sosei_vol_flow']['SO2'] - SO2
        self.outlet_strm['Sosei_vol_flow']['SO3'] = strm['Sosei_vol_flow']['SO3'] - SO3
        self.outlet_strm['Sosei_vol_flow']['Nitric Oxide'] = strm['Sosei_vol_flow']['Nitric Oxide'] - NOx
        self.outlet_strm['Sosei_vol_flow']['HCl'] = strm['Sosei_vol_flow']['HCl'] - HCl
        self.outlet_strm['Sosei_vol_flow']['HF'] = strm['Sosei_vol_flow']['HF'] - HF
        self.outlet_strm['Sosei_vol_flow']['NH3'] = self.η_and_out['out_NH3'] * (self.outlet_strm['Sosei_vol_flow']['N2'] +\
                                                                                 self.outlet_strm['Sosei_vol_flow']['O2'] +\
                                                                                 self.outlet_strm['Sosei_vol_flow']['CO2']+\
                                                                                 self.outlet_strm['Sosei_vol_flow']['SO2']+\
                                                                                 self.outlet_strm['Sosei_vol_flow']['SO3']+\
                                                                                 self.outlet_strm['Sosei_vol_flow']['Nitric Oxide']+\
                                                                                 self.outlet_strm['Sosei_vol_flow']['HCl']+\
                                                                                 self.outlet_strm['Sosei_vol_flow']['HF'])/ ( 1- self.η_and_out['out_NH3'])
        self.outlet_strm['Flow'] =  self.outlet_strm['Sosei_vol_flow']['H2O']+\
                                    self.outlet_strm['Sosei_vol_flow']['N2'] +\
                                    self.outlet_strm['Sosei_vol_flow']['O2'] +\
                                    self.outlet_strm['Sosei_vol_flow']['CO2']+\
                                    self.outlet_strm['Sosei_vol_flow']['SO2']+\
                                    self.outlet_strm['Sosei_vol_flow']['SO3']+\
                                    self.outlet_strm['Sosei_vol_flow']['Nitric Oxide']+\
                                    self.outlet_strm['Sosei_vol_flow']['HCl']+\
                                    self.outlet_strm['Sosei_vol_flow']['HF']
        self.outlet_strm['Dust'] =  strm['Dust'] - Dust
        self.outlet_strm['T'] =  strm['T']
        self.outlet_strm['P'] =  strm['P']    

# 再生塔クラス        
class ReACT_Regenerator:
    def __init__(self, number, name, num):
        self.number = number
        self.name = name
        self.num_of_regenerators = num
    
    def calc_regenerator_reaction(self,rho_AC,AC_vol_flow_rate,**adsorption):
        self.mechanical_loss_rate = 1/100
        self.rho_AC = rho_AC
        self.AC_vol_flow_rate = AC_vol_flow_rate
        self.AC_mass_flow_rate = rho_AC * AC_vol_flow_rate
        self.desorption_gas = self.calc_desorption(**adsorption)
        self.carrier_N2 = self.calc_carrier_N2(self.AC_vol_flow_rate,self.num_of_regenerators)
        self.seal_N2 = self.calc_seal_N2(self.num_of_regenerators)
        self.SRG = self.calc_SRG(self.carrier_N2,self.seal_N2,self.AC_mass_flow_rate,**self.desorption_gas)
        self.chemical_loss = self.calc_chemical_loss(self.SRG)
        self.mechanical_loss = self.calc_mecanical_loss(self.AC_mass_flow_rate,self.mechanical_loss_rate)
        self.chemical_loss_rate = self.chemical_loss/(self.AC_mass_flow_rate*1000)
        self.AC_loss = self.mechanical_loss + self.chemical_loss
        self.AC_loss_rate = self.AC_loss/(self.AC_mass_flow_rate*1000)
        self.strm_SRG = GasStream(**self.SRG)
    

    def calc_desorption(self,**adsorption):
        SO3 = adsorption['H2SO4'] + adsorption['NH4HSO4'] + adsorption['(NH4)2SO4']
        H2O = adsorption['H2SO4'] + adsorption['NH4HSO4'] + adsorption['(NH4)2SO4'] + adsorption['H2O']
        NH3 = adsorption['NH4HSO4'] + 2 * adsorption['(NH4)2SO4']
        HCl = adsorption['HCl']
        HF  = adsorption['HF']
        desorption_gas = {'SO3':SO3,'H2O':H2O,'NH3':NH3,'HCl':HCl,'HF':HF}
        return desorption_gas
    
    def calc_carrier_N2(self,AC_vol_flow_rate,num_of_regenerators):
        N2_per_AC_ratio = 2.2 # N2/AC体積流量比率
        carrier_N2 = math.ceil(AC_vol_flow_rate * N2_per_AC_ratio/num_of_regenerators)*num_of_regenerators
        return carrier_N2
    
    def calc_seal_N2(self,num_of_regenerators):
        pressure_gage = 1
        emergency_line = 4
        top_rotary_valve_carrier = 8
        bottom_rotary_valve_carrier = 12
        bottom_rotary_valve_leak = 12
        seal_N2 = (pressure_gage + emergency_line + top_rotary_valve_carrier + bottom_rotary_valve_carrier + bottom_rotary_valve_leak)*num_of_regenerators
        return seal_N2
    
    def calc_SRG(self,carrier_N2,seal_N2,AC_mass_flow_rate,**desorption_gas):
        CO2_generation = 4.26 #g/kg-AC
        SO2 = desorption_gas['SO3']
        H2O = desorption_gas['NH3'] * 1.5 + desorption_gas['H2O']
        N2  = desorption_gas['NH3'] * 0.5 + carrier_N2 + seal_N2
        CO2 = (desorption_gas['SO3'] - desorption_gas['NH3'] * 1.5) * 0.5 + AC_mass_flow_rate * CO2_generation / 44 * 22.4
        HCl = desorption_gas['HCl']
        HF  = desorption_gas['HF']
        SRG = {'T':200,'P':0,'Sosei_vol_flow':{'SO2':SO2,'H2O':H2O,'N2':N2,'CO2':CO2,'HCl':HCl,'HF':HF}}
        return SRG
    
    def calc_chemical_loss(self, SRG):
        return SRG['Sosei_vol_flow']['CO2']/22.4*12
    
    def calc_mecanical_loss(self,AC_mass_flow_rate,mechanical_loss_rate):
        return AC_mass_flow_rate * mechanical_loss_rate * 1000        
        
#%% ヒーター・クーラー
def strm_heater_cooler_deltaT_to_Q(strm1, deltaT):
    strm_spec = copy.copy(strm1.strm)
    strm_spec['T'] += deltaT 
    strm3 = GasStream(**strm_spec)
    Q = strm3.strm['h'] - strm1.strm['h']
    
    return strm3, Q


def strm_heater_cooler_T_to_Q(strm1, T):
    strm_spec = copy.copy(strm1.strm)
    strm_spec['T'] = T 
    strm3 = GasStream(**strm_spec)
    Q = strm3.strm['h'] - strm1.strm['h']
    
    return strm3, Q

# Q MJ/h (kW*3.6)
def strm_heater_cooler_Q_to_Tout(strm1, Q):
    strm_spec = copy.copy(strm1.strm)
    strm3 = GasStream(**strm_spec)
    strm3.strm['h'] += Q
    
    #T_list = list(np.linspace(strm3.strm['T'],strm3.strm['T']*(1+Q/strm3.strm['h']),50))
    T_list = list(np.linspace(-273.15,800,2000))
    res_list = []
 
    for i in T_list:
        strm3.strm['T'] = i
        h_temp = strm_h_gas(**strm3.strm)['h']
        res_list.append((strm3.strm['h'] - h_temp)**2)
 
    strm3.strm['T'] = T_list[res_list.index(min(res_list))]
    strm3.calc_strm()
    
    Tout = strm3.strm['T']
    
    return strm3, Tout
    
#%% ファン　クラス
class Fan:
    def __init__(self, k, v, strm, ita=0.7):
        self.strm_in = copy.copy(strm.strm)
        self.strm_out = copy.copy(strm.strm)
        self.condition = k
        self.value = v
        self.ita = ita
        self.check_condition()
    
    def check_condition(self):
        if self.condition == 'P_out':
            self.strm_out['P'] = self.value
#            self.power_ad = self.calc_adiabatic_shaft_power()
#            self.power = self.calc_actual_shaft_power()
#            self.strm_out['T_ad'] = self.calc_adiabatic_outlet_temperature()
#            self.strm_out['T'] = self.calc_actual_outlet_temperature()
            
            self.power_ad = calc_Fan_Lt_Tout(self.strm_out['P'],self.ita,**self.strm_in)[0]
            self.power = calc_Fan_Lt_Tout(self.strm_out['P'],self.ita,**self.strm_in)[1]
            self.strm_out['T_ad'] = calc_Fan_Lt_Tout(self.strm_out['P'],self.ita,**self.strm_in)[2]
            self.strm_out['T'] = calc_Fan_Lt_Tout(self.strm_out['P'],self.ita,**self.strm_in)[3]          
            
        
        elif self.condition == 'P_in':
            self.strm_in['P'] = self.value
            self.power_ad = self.calc_adiabatic_shaft_power()
            self.power = self.calc_actual_shaft_power()
            self.strm_out['T_ad'] = self.calc_adiabatic_outlet_temperature()
            self.strm_out['T'] = self.calc_actual_outlet_temperature()
        
        elif self.condition == 'delta_P':
            if self.value > 0:
                self.strm_out['P'] = self.strm_in['P'] + self.value
            else:
                self.strm_in['P'] = self.strm_out['P'] + self.value
            self.power_ad = self.calc_adiabatic_shaft_power()
            self.power = self.calc_actual_shaft_power()
            self.strm_out['T_ad'] = self.calc_adiabatic_outlet_temperature()
            self.strm_out['T'] = self.calc_actual_outlet_temperature()
        
        elif self.condition == 'Power':
            self.Power = {'Power':self.value}
            
        else:
            print('P_in, P_out, delta_P, Power')
    
    def calc_adiabatic_shaft_power(self):
        Lt_ad = self.strm_in['kappa']/(self.strm_in['kappa']-1)*(self.strm_in['P']+101.325)*self.strm_in['vol_flow_actual']/3600*(((self.strm_out['P']+101.325)/(self.strm_in['P']+101.325))**((self.strm_in['kappa']-1)/self.strm_in['kappa'])-1)
        return Lt_ad    
        
    def calc_adiabatic_outlet_temperature(self):
        T_out_ad = (self.strm_in['T']+273.15)*((self.strm_in['P']+101.325)/(self.strm_out['P']+101.325))**((1-self.strm_in['kappa'])/self.strm_in['kappa']) - 273.15
        return T_out_ad

    def calc_actual_outlet_temperature(self):
        T_out = (self.strm_out['T_ad'] - self.strm_in['T'])/self.ita + self.strm_out['T']
        return T_out
    
    def calc_actual_shaft_power(self):
        Lt = self.power_ad / self.ita
        return Lt
        
         
 
print('JPE_ReACT')

#%%
def calc_Fan_Lt_Tout(P_out,ita,**strm):
    Lt_ad = strm['kappa']/(strm['kappa']-1)*(strm['P']+101.325)*strm['vol_flow_actual']/3600*(((P_out+101.325)/(strm['P']+101.325))**((strm['kappa']-1)/strm['kappa'])-1)
    Lt = Lt_ad / ita
    T_out_ad = (strm['T']+273.15)*((strm['P']+101.325)/(P_out+101.325))**((1-strm['kappa'])/strm['kappa']) - 273.15
    T_out = (T_out_ad - strm['T'])/ita + strm['T']
    
    return Lt_ad,  Lt, T_out_ad, T_out


#%% ファン　クラス
class Fan_:
    def __init__(self, number, name):
        self.number = number
        self.name = name
    
    def set_design_spec(self, **spec):
        self.design_spec= spec
        self.strm_design_in = {}
        self.strm_design_in['Flow'] = self.design_spec['吸込流量_m3/min'] * 60 * 273.15 / (self.design_spec['吸込温度_℃']+273.15) * (101.325 + self.design_spec['吸込圧力_kPa']) /101.325
        self.strm_design_in['vol_flow_actual'] = self.design_spec['吸込流量_m3/min'] * 60
        self.strm_design_in['Sosei_dry'] = self.design_spec['流体性状_dry']
        self.strm_design_in['T'] = self.design_spec['吸込温度_℃']
        self.strm_design_in['P'] = self.design_spec['吸込圧力_kPa']
        self.strm_design_in = GasStream(**self.strm_design_in).strm
        calc_fan = calc_Fan_Lt_Tout(self.design_spec['吐出圧力_kPa'], self.design_spec['効率_%']/100,  **self.strm_design_in)
        
        
        self.strm_design_out = {}
        self.strm_design_out['Flow'] =  self.strm_design_in['Flow']
        self.strm_design_out['vol_flow_actual'] = self.design_spec['吸込流量_m3/min'] * 60
        self.strm_design_out['Sosei_dry'] = self.design_spec['流体性状_dry']
        self.strm_design_out['T'] = calc_fan[3]
        self.strm_design_out['P'] = self.design_spec['吐出圧力_kPa']        
        self.strm_design_out = GasStream(**self.strm_design_out).strm
        
        self.design_spec['軸動力_kW'] = round(calc_fan[1],1)
        self.design_spec['吐出温度_℃'] = round(calc_fan[3],1)
        self.design_spec['昇温_℃'] = round(calc_fan[3] - self.design_spec['吸込温度_℃'],1)
        
        self.strm_design_in_min_temp = copy.copy(self.strm_design_in)
        self.strm_design_in_min_temp['T'] = -5.0
        self.strm_design_in_min_temp = GasStream(**self.strm_design_in_min_temp).strm
        pressure_min_temp = (self.design_spec['吐出圧力_kPa']-self.design_spec['吸込圧力_kPa']) * (273.15 + self.design_spec['吸込温度_℃']) / (273.15 + self.strm_design_in_min_temp['T']) + self.design_spec['吸込圧力_kPa']
        self.strm_design_in_min_temp['vol_flow_actual'] = self.strm_design_in['vol_flow_actual']
        print(pressure_min_temp)
        calc_fan_min_temp = calc_Fan_Lt_Tout(pressure_min_temp, self.design_spec['効率_%']/100,  **self.strm_design_in_min_temp)
        self.design_spec['軸動力(最低温度)_kW'] = round(calc_fan_min_temp[1],1)


    
    def check_condition(self):
        if self.condition == 'P_out':
            self.strm_out['P'] = self.value
#            self.power_ad = self.calc_adiabatic_shaft_power()
#            self.power = self.calc_actual_shaft_power()
#            self.strm_out['T_ad'] = self.calc_adiabatic_outlet_temperature()
#            self.strm_out['T'] = self.calc_actual_outlet_temperature()
            
            self.power_ad = calc_Fan_Lt_Tout(self.strm_out['P'],self.ita,**self.strm_in)[0]
            self.power = calc_Fan_Lt_Tout(self.strm_out['P'],self.ita,**self.strm_in)[1]
            self.strm_out['T_ad'] = calc_Fan_Lt_Tout(self.strm_out['P'],self.ita,**self.strm_in)[2]
            self.strm_out['T'] = calc_Fan_Lt_Tout(self.strm_out['P'],self.ita,**self.strm_in)[3]          
            
        
        elif self.condition == 'P_in':
            self.strm_in['P'] = self.value
            self.power_ad = self.calc_adiabatic_shaft_power()
            self.power = self.calc_actual_shaft_power()
            self.strm_out['T_ad'] = self.calc_adiabatic_outlet_temperature()
            self.strm_out['T'] = self.calc_actual_outlet_temperature()
        
        elif self.condition == 'delta_P':
            if self.value > 0:
                self.strm_out['P'] = self.strm_in['P'] + self.value
            else:
                self.strm_in['P'] = self.strm_out['P'] + self.value
            self.power_ad = self.calc_adiabatic_shaft_power()
            self.power = self.calc_actual_shaft_power()
            self.strm_out['T_ad'] = self.calc_adiabatic_outlet_temperature()
            self.strm_out['T'] = self.calc_actual_outlet_temperature()
        
        elif self.condition == 'Power':
            self.Power = {'Power':self.value}
            
        else:
            print('P_in, P_out, delta_P, Power')
    
    def calc_adiabatic_shaft_power(self):
        Lt_ad = self.strm_in['kappa']/(self.strm_in['kappa']-1)*(self.strm_in['P']+101.325)*self.strm_in['vol_flow_actual']/3600*(((self.strm_out['P']+101.325)/(self.strm_in['P']+101.325))**((self.strm_in['kappa']-1)/self.strm_in['kappa'])-1)
        return Lt_ad    
        
    def calc_adiabatic_outlet_temperature(self):
        T_out_ad = (self.strm_in['T']+273.15)*((self.strm_in['P']+101.325)/(self.strm_out['P']+101.325))**((1-self.strm_in['kappa'])/self.strm_in['kappa']) - 273.15
        return T_out_ad

    def calc_actual_outlet_temperature(self):
        T_out = (self.strm_out['T_ad'] - self.strm_in['T'])/self.ita + self.strm_out['T']
        return T_out
    
    def calc_actual_shaft_power(self):
        Lt = self.power_ad / self.ita
        return Lt
    
    def pq_curve(self):
        inlet_flow = np.array(self.design_spec['PQ_Q'])
        delta_p = np.array(self.design_spec['PQ_P'])
        f1d_delta_p = interpolate.interp1d(inlet_flow, delta_p, kind = 'cubic')
        inlet_flow_new = np.linspace(0,max(self.design_spec['PQ_Q']), 100)
        delta_p_f1d = f1d_delta_p(inlet_flow_new)
        return inlet_flow_new, delta_p_f1d
    

def calc_desox_efficiency(ACth,so2in,h2oin,o2in,Tempin,lvin,so2Lin,RTinme,SO2index):
    #"""READ Adsorption Table""" 
    table=np.loadtxt('table.csv',delimiter=',',skiprows=1,dtype=float)

    #"""Parameter"""
    ACth = ACth * 1000             #2000"""AC Bed Depth [mm]"""
    so2in = so2in * 1000000           #500.0"""Flue Gas SO2 [ppm-dry]"""
    h2oin = h2oin * 100            #10.0"""Flue Gas H2O [%-wet]"""
    o2in = o2in * 100              #5.0"""Flue Gas O2 [%-dry]"""
    #Tempin = 133            #150.0"""Flue Gas Temperature [degC]"""
    #lvin = 0.109             #0.20"""Linear Velocity [mN/s]"""
    so2Lin = so2Lin/(0.2*0.36*0.93/22.4*64/0.65) 
    #2.0/(0.2*0.36*0.93/22.4*64/0.65)"""SO2 Adsorption [mg/g-aC-->ppm/cm]""" 
    #RTime = 170              #80"""Retention Time [h]"""
    #SO2index = 0.9          #1.0"""DeSOx index  [-]"""
    ACmax = int(ACth/10)    #"""AC Bed Depth [cm]"""

    #""" """
    h2o=np.zeros(RTime+1)
    o2=np.zeros(RTime+1)
    Temp=np.zeros(RTime+1)
    lv=np.zeros(RTime+1)
    acad=np.zeros((RTime+2, ACmax+1))
    so2=np.zeros((RTime+2, ACmax+1))

    table = SO2index * table

    #"""H2O=7% Base"""
    Bh2o = 7.0    

    #"""inisialize """
    for i in range(0,RTime+1):
        j = 0
        h2o[i] = h2oin
        so2[i, j] = so2in
        o2[i] = o2in
        Temp[i] = Tempin
        lv[i] = lvin

    for j in range(0,ACmax):
        acad[0, j] = so2Lin


    #"""Calculation"""
    for i  in range(0,RTime+1):
        for j in range(0,ACmax):        
            """SO2換算　H2O＝7%ベース"""
            so2D = so2[i, j]
            so2W = so2D * (100 - h2o[i]) / 100
            so2B = so2W / ((100 - Bh2o) / 100)

            """O2換算　H2O＝7%ベース"""
            o2D = o2[i]
            o2W = o2D * (100 - h2o[i]) / 100
            o2B = o2W / ((100 - Bh2o) / 100)

            """残存SO2量"""
            so2L = acad[i, j]

            """行列番号取得"""


            """SO2濃度650ppm超の処理"""
            if so2B >= 630:
                so2row = int(so2L / 20) 
                so2col = int(630 / 20) 

                r = table[so2row, int(630/20)]


            else:
                so2row = int(so2L / 20) 
                so2col = int(so2B / 20) 

                r0 = table[so2row, so2col]
                r1 = table[so2row, so2col + 1]
                r2 = table[so2row + 1, so2col]
                r = (r1-r0)*(so2B-so2col*20)/20+(r2-r0)*(so2L-so2row*20)/20+r0

            """O2依存性"""
            if o2B < 3:
                o2prm = 0.94805
            elif o2B < 5:
                o2prm = 0.025974 * o2B + 0.87013
            elif o2B < 7:
                o2prm = 0.038961 * o2B + 0.8052
            else:
                o2prm = 1.07792

            """水分依存性"""
            if h2o[i] < 7:
                h2oprm = 1
            elif h2o[i] < 9:
                h2oprm = 0.038961 * h2o[i] + 0.72727
            elif h2o[i] < 11:
                h2oprm = 0.032467 * h2o[i] + 0.78571
            else:
                h2oprm = 1.14286

            """温度依存性"""
            if Temp[i] < 120:
                Tprm = 1.07792
            elif Temp[i] < 140:
                Tprm = -0.0038961 * Temp[i] + 1.54545
            elif Temp[i] < 160:
                Tprm = -0.0012987 * Temp[i] + 1.18182
            elif Temp[i] < 190:
                Tprm = -0.0004329 * Temp[i] + 1.04329
            else:
                Tprm = 0.96104

            """O2、水分、温度補正後脱硫速度計算"""
            SO2Rate = o2prm * h2oprm * Tprm * r / 5 * 0.635
            #SO2Rate = o2prm * h2oprm * Tprm * r          
            """出口SO2濃度算出"""
            so2bb = so2B - SO2Rate * 0.2 / lv[i]

            """出口SO2濃度＜０の処理"""
            if so2bb < 0:
                so2bb = 0

            """SO2濃度をDRYベースに換算"""
            so2db = so2bb * ((100 - Bh2o) / 100) / ((100 - h2o[i]) / 100)

            """SO2濃度を次のACへ渡す（層幅方向）"""
            so2[i, j + 1] = so2db

            """SO2吸着量を次のACへ渡す（層高さ方向）"""
            acad[i + 1, j] = acad[i, j] + SO2Rate


    #"""脱硫塔出口SO2濃度総和算出"""
    so2sum = so2[:RTime+1,ACmax]
    Sum = np.sum(so2sum)


    #"""積分平均脱硫効率算出 '小数点第３位切り捨て"""
    EffSO2 = (int((so2in - Sum/RTime) / so2in * 1000)) / 10
    print("DeSOx Efficiency = " + str(EffSO2)+" %")
    return EffSO2,so2,acad


if __name__ == '__main__':
    print('prpertyがimportされました')
    
    spec_list = {'B0420':{'機器名称':'熱風循環ファン',
                          '吸込流量_m3/min':1160.,
                          '吸込温度_℃':284.,
                          '吸込圧力_kPa':-6.,
                          '吐出圧力_kPa':9.,
                          '効率_%':69.1,
                          '電動機出力_kW':450.,
                          '流体名':'空気',
                          '流体性状_dry':{'H2O':2/100,'N2':79/100,'O2':21/100},
                          'PQ_Q':[0, 420, 780, 1160, 1460],
                          'PQ_P':[16.0, 17.5, 17.5, 15.0, 12.5]},
                 'B0110':{'機器名称':'アンモニア希釈ファン',
                          '吸込流量_m3/min':360.,
                          '吸込温度_℃':35.,
                          '吸込圧力_kPa':-0.58,
                          '吐出圧力_kPa':11.42,
                          '効率_%':71.1,
                          '電動機出力_kW':132.,
                          '流体名':'空気',
                          '流体性状_dry':{'H2O':2/100,'N2':79/100,'O2':21/100}},
                 'B0540':{'機器名称':'風選ファン',
                          '吸込流量_m3/min':108.,
                          '吸込温度_℃':86.,
                          '吸込圧力_kPa':-4.42,
                          '吐出圧力_kPa':2.58,
                          '効率_%':60.8,
                          '電動機出力_kW':22.,
                          '流体名':'空気',
                          '流体性状_dry':{'H2O':2/100,'N2':79/100,'O2':21/100}},
                 'B0520':{'機器名称':'防じんファン',
                          '吸込流量_m3/min':179.,
                          '吸込温度_℃':50.,
                          '吸込圧力_kPa':-5.6,
                          '吐出圧力_kPa':0.8,
                          '効率_%':69.1,
                          '電動機出力_kW':37.,
                          '流体名':'空気',
                          '流体性状_dry':{'H2O':2/100,'N2':79/100,'O2':21/100}}
                 }
    
    #fan_list = {'B0420' :'熱風循環ファン','B0110' :'アンモニア希釈ファン', 'B0540':'風選ファン', 'B0520':'防じんファン'}
    fan_class_list = []
    for k, v in spec_list.items():
        fan_class_list.append(Fan_(k, v['機器名称']))
        
    for i in range(0,len(ｓpec_list)):
        fan_class_list[i].set_design_spec(**spec_list[fan_class_list[i].number])
    
    for i in range(0,len(fan_class_list)):
        print(fan_class_list[i].number)
        print(fan_class_list[i].design_spec)



