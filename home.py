import streamlit as st
import pandas as pd
import jpe_react as jpe
import re

strm = []
cols = st.columns(3)
for i in range(0,3):
    cols[i].subheader('Stream' + str(i))
    flow = cols[i].number_input('ガス量'+ str(i)+'：m3N/h',value=1000000)
    temp = cols[i].number_input('ガス温度'+ str(i)+'：℃',value=150.0)
    pres = cols[i].number_input('圧力'+ str(i)+'：kPa',value=2.5)
    sosei = cols[i].data_editor(pd.DataFrame({'成分'+str(i):['H2O','N2','O2'],
                                   '組成':['10.0%','70.5%','540ppm']}),
                       hide_index=True,
                       num_rows="dynamic")

    strm_dict = {}
    strm_dict['Flow'] = flow
    strm_dict['T']    = temp
    strm_dict['P']    = pres
    strm_dict['Sosei_dry'] = {}
    try:
        for i in range(sosei.shape[0]):
            if '%' in sosei.iloc[i,1]:
                strm_dict['Sosei_dry'][sosei.iloc[i,0]] =  float(re.sub(r'%', "", sosei.iloc[i,1]))/100
            elif 'ppm' in sosei.iloc[i,1]:
                strm_dict['Sosei_dry'][sosei.iloc[i,0]] =  float(re.sub(r'\D', "", sosei.iloc[i,1]))/1000000
            else:
                pass
    except:
        print('error')
        
    strm.append(jpe.GasStream(**strm_dict))

for i in range(0,3):
    cols[i].write(strm[i].strm['Sosei_vol_flow'])



if st.button('Mix0&1'):
    strm_mix = jpe.Stream_Mixer(strm[0],strm[1])
    strm_mix

if st.button('Mix1&2'):
    strm_mix = jpe.Stream_Mixer(strm[1],strm[2])
    strm_mix

if st.button('Mix0&2'):
    strm_mix = jpe.Stream_Mixer(strm[0],strm[2])
    strm_mix
