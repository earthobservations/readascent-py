import numpy as np
from datetime import datetime
#from django.utils import timezone
import pytz
from io import BytesIO, open
from ftplib import FTP
from netCDF4 import Dataset
import gzip
import json
from scipy.interpolate import interp1d
#from api.models import Station, Radiosonde, UpdateRecord
import warnings
#warnings.filterwarnings("ignore")



def winds_to_UV(windSpeeds, windDirection):
    u = [];  v = []
    for i,wdir in enumerate(windDirection):
        rad = 4.0*np.arctan(1)/180.
        u.append(-windSpeeds[i] * np.sin(rad*wdir))
        v.append(-windSpeeds[i] * np.cos(rad*wdir))
    return np.array(u), np.array(v)


def basic_qc(Ps, T, Td, U, V):
    # remove the weird entries that give TOA pressure at the start of the array
    Ps = np.round(Ps[np.where(Ps>100)], 2)
    T = np.round(T[np.where(Ps>100)], 2)
    Td = np.round(Td[np.where(Ps>100)], 2)
    U = np.round(U[np.where(Ps>100)], 2)
    V = np.round(V[np.where(Ps>100)], 2)

    U[np.isnan(U)] = -9999
    V[np.isnan(V)] = -9999
    Td[np.isnan(Td)] = -9999
    T[np.isnan(T)] = -9999
    Ps[np.isnan(Ps)] = -9999

    if T.size != 0:
        if T[0]<200 or T[0]>330 or np.isnan(T).all():
            Ps = np.array([]); T = np.array([]); Td = np.array([])
            U = np.array([]); V = np.array([])

    if not isinstance(Ps, list):
        Ps = Ps.tolist()
    if not isinstance(T, list):
        T = T.tolist()
    if not isinstance(Td, list):
        Td = Td.tolist()
    if not isinstance(U, list):
        U = U.tolist()
    if not isinstance(V, list):
        V = V.tolist()

    return Ps, T, Td, U, V


def RemNaN_and_Interp(raob):
    P_allstns = []; T_allstns = []; Td_allstns = []; times_allstns = []
    U_allstns = []; V_allstns = [];  wmo_ids_allstns = []

    for i,stn in enumerate(raob['Psig']):
        Ps = raob['Psig'][i]
        Ts = raob['Tsig'][i]
        Tds = raob['Tdsig'][i]
        Tm = raob['Tman'][i]
        Tdm = raob['Tdman'][i]
        Pm = raob['Pman'][i]
        Ws = raob['Wspeed'][i]
        Wd = raob['Wdir'][i]

        if len(Pm)>10 and len(Ps)>10:
            u, v = winds_to_UV(Ws, Wd)

            PmTm = zip(Pm, Tm)
            PsTs = zip(Ps, Ts)
            PmTdm = zip(Pm, Tdm)
            PsTds = zip(Ps, Tds)

            PT=[]; PTd = []
            for pmtm in PmTm:
                PT.append(pmtm)
            for psts in PsTs:
                PT.append(psts)
            for pmtdm in PmTdm:
                PTd.append(pmtdm)
            for pstds in PsTds:
                PTd.append(pstds)


            PT = [x for x in PT if all(i == i for i in x)]
            PTd = [x for x in PTd if all(i == i for i in x)]

            PT = sorted(PT, key=lambda x: x[0])
            PT = PT[::-1]
            PTd = sorted(PTd, key=lambda x: x[0])
            PTd = PTd[::-1]

            if len(PT)!=0 and len(PTd)>10:
                P, T = zip(*PT)
                Ptd, Td = zip(*PTd)
                P = np.array(P)
                P = P.astype(int)
                T = np.array(T)
                Td = np.array(Td)

                f = interp1d(Ptd, Td, kind='linear', fill_value="extrapolate")
                Td = f(P)
                f = interp1d(Pm, u, kind='linear', fill_value="extrapolate")
                U = f(P)
                f = interp1d(Pm, v, kind='linear', fill_value="extrapolate")
                V = f(P)

                U = U*1.94384
                V = V*1.94384

                Pqc, Tqc, Tdqc, Uqc, Vqc = basic_qc(P, T, Td, U, V)

                if len(Pqc)!=0:
                    P_allstns.append(Pqc)
                    T_allstns.append(Tqc)
                    Td_allstns.append(Tdqc)
                    U_allstns.append(Uqc)
                    V_allstns.append(Vqc)
                    wmo_ids_allstns.append(raob['wmo_ids'][i])
                    times_allstns.append(raob['times'][i])

    return P_allstns, T_allstns, Td_allstns, U_allstns, V_allstns, wmo_ids_allstns, times_allstns


def commit_sonde(raob, station):
    P, T, Td, U, V, wmo_ids, times = RemNaN_and_Interp(raob)

    if not station in wmo_ids:
        print("--- station not found:", station)
        return False

    for i,stn in enumerate(wmo_ids):
        if stn != station:
            continue
        print(i,stn)

        sonde_validtime = times[i]
        temperatureK = T[i]
        dewpointK = Td[i]
        pressurehPA = P[i]
        u_windMS = U[i]
        v_windMS = V[i]
        ascent = []
        #print(sonde_validtime, len(temperatureK),len(dewpointK),len(pressurehPA),len(u_windMS),len(v_windMS))
        for i in range(len(temperatureK)):
            pt = {
            "pressure" : pressurehPA[i],
            "temperature": temperatureK[i],
            "dewpoint":dewpointK[i],
            "wind_u": u_windMS[i],
            "wind_v":  v_windMS[i]
            }
            ascent.append(pt)
            #print(pressurehPA[i],temperatureK[i], dewpointK[i],  u_windMS[i], v_windMS[i])
        print(json.dumps(ascent, indent=4))
        
def extract_madis_data(file, station):

    with gzip.open(file, 'rb') as f:
        nc = Dataset('inmemory.nc', memory=f.read())
        Tman = nc.variables['tpMan'][:].filled(fill_value=np.nan)
        DPDman = nc.variables['tdMan'][:].filled(fill_value=np.nan)
        wmo_ids = nc.variables['wmoStaNum'][:].filled(fill_value=np.nan)
        DPDsig = nc.variables['tdSigT'][:].filled(fill_value=np.nan)
        Tsig = nc.variables['tpSigT'][:].filled(fill_value=np.nan)
        synTimes = nc.variables['synTime'][:].filled(fill_value=np.nan)
        raob = {
            "Tsig": nc.variables['tpSigT'][:].filled(fill_value=np.nan),
            "Tdsig": Tsig - DPDsig,
            "Tman": Tman,
            "Psig": nc.variables['prSigT'][:].filled(fill_value=np.nan),
            "Pman": nc.variables['prMan'][:].filled(fill_value=np.nan),
            "Tdman": Tman - DPDman,
            "Wspeed": nc.variables['wsMan'][:].filled(fill_value=np.nan),
            "Wdir": nc.variables['wdMan'][:].filled(fill_value=np.nan),
            "times": [datetime.utcfromtimestamp(tim).replace(tzinfo=pytz.utc) for tim in synTimes],
            "wmo_ids": [str(id).zfill(5) for id in wmo_ids]
        }
        commit_sonde(raob, station)




#def run():
    # UpdateRecord.delete_expired(10)
raob = extract_madis_data("madis/20210130_1100.gz", '42182')
