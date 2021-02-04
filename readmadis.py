import geojson
import numpy as np
from datetime import datetime, timezone, timedelta
from math import cos, pi, isnan
#from django.utils import timezone
import pytz
from io import BytesIO, open
from ftplib import FTP
from netCDF4 import Dataset
import gzip
from scipy.interpolate import interp1d
#from api.models import Station, Radiosonde, UpdateRecord
import warnings
warnings.filterwarnings("ignore")
# blatantly stolen from https://github.com/tjlang/SkewT
from thermodynamics import  barometric_equation_inv
import sys
import os
import json

ascent_rate = 5 #m/s = 300m/min


def winds_to_UV(windSpeeds, windDirection):
    u = []
    v = []
    for i, wdir in enumerate(windDirection):
        rad = 4.0 * np.arctan(1) / 180.
        u.append(-windSpeeds[i] * np.sin(rad * wdir))
        v.append(-windSpeeds[i] * np.cos(rad * wdir))
    return np.array(u), np.array(v)


def basic_qc(Ps, T, Td, U, V):
    # remove the weird entries that give TOA pressure at the start of the array
    Ps = np.round(Ps[np.where(Ps > 100)], 2)
    T = np.round(T[np.where(Ps > 100)], 2)
    Td = np.round(Td[np.where(Ps > 100)], 2)
    U = np.round(U[np.where(Ps > 100)], 2)
    V = np.round(V[np.where(Ps > 100)], 2)

    U[np.isnan(U)] = -9999
    V[np.isnan(V)] = -9999
    Td[np.isnan(Td)] = -9999
    T[np.isnan(T)] = -9999
    Ps[np.isnan(Ps)] = -9999

    if T.size != 0:
        if T[0] < 200 or T[0] > 330 or np.isnan(T).all():
            Ps = np.array([])
            T = np.array([])
            Td = np.array([])
            U = np.array([])
            V = np.array([])

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
    P_allstns = []
    T_allstns = []
    Td_allstns = []
    times_allstns = []
    U_allstns = []
    V_allstns = []
    wmo_ids_allstns = []
    relTime_allstns = []
    sondTyp_allstns = []
    staLat_allstns = []
    staLon_allstns = []
    staElev_allstns = []

    for i, stn in enumerate(raob['Psig']):
        Ps = raob['Psig'][i]
        Ts = raob['Tsig'][i]
        Tds = raob['Tdsig'][i]
        Tm = raob['Tman'][i]
        Tdm = raob['Tdman'][i]
        Pm = raob['Pman'][i]
        Ws = raob['Wspeed'][i]
        Wd = raob['Wdir'][i]

        if len(Pm) > 10 and len(Ps) > 10:
            u, v = winds_to_UV(Ws, Wd)

            PmTm = zip(Pm, Tm)
            PsTs = zip(Ps, Ts)
            PmTdm = zip(Pm, Tdm)
            PsTds = zip(Ps, Tds)

            PT = []
            PTd = []
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

            if len(PT) != 0 and len(PTd) > 10:
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

                U = U * 1.94384
                V = V * 1.94384

                Pqc, Tqc, Tdqc, Uqc, Vqc = basic_qc(P, T, Td, U, V)

                if len(Pqc) != 0:
                    P_allstns.append(Pqc)
                    T_allstns.append(Tqc)
                    Td_allstns.append(Tdqc)
                    U_allstns.append(Uqc)
                    V_allstns.append(Vqc)
                    wmo_ids_allstns.append(raob['wmo_ids'][i])
                    times_allstns.append(raob['times'][i])
                    relTime_allstns.append(raob['relTime'][i])
                    sondTyp_allstns.append(raob['sondTyp'][i])
                    staLat_allstns.append(raob['staLat'][i])
                    staLon_allstns.append(raob['staLon'][i])
                    staElev_allstns.append(raob['staElev'][i])

    return (relTime_allstns, sondTyp_allstns, staLat_allstns, staLon_allstns, staElev_allstns,
            P_allstns, T_allstns, Td_allstns, U_allstns, V_allstns, wmo_ids_allstns, times_allstns)


class Radiosonde(object):
    pass

# very simplistic
def height2time(h0, height):
    hdiff = height - h0
    return hdiff / ascent_rate


mperdeg = 111320.0

def latlonPlusDisplacement(lat=0, lon=0, u=0, v=0):
    dLat =  v / mperdeg
    dLon =  u / (cos((lat + dLat/2) / 180 * pi) * mperdeg)
    return lat + dLat, lon + dLon


def commit_sonde(raob, stations):
    relTime, sondTyp, staLat, staLon, staElev, P, T, Td, U, V, wmo_ids, times = RemNaN_and_Interp(
        raob)
    #print(staElev)
    for i, stn in enumerate(wmo_ids):

        if stn in stations:
            station = stations[stn]
#            print(f"---- station {stn} found: {station}")
            if isnan(staLat[i]):
                staLat[i] = station['lat']
            if isnan(staLon[i]):
                staLon[i] = station['lon']
            if isnan(staElev[i]):
                staElev[i] = station['elevation']
        else:
#            print(f"station {stn} not found")
            station = None

        if isnan(staLat[i]) or isnan(staLon[i]) or isnan(staElev[i]):
            continue

        #print(i, stn)
        takeoff =  datetime.utcfromtimestamp(relTime[i]).replace(tzinfo=pytz.utc)
        syntime = times[i]
        properties = {
            "station_id":  stn,
            "id_type":  "madis",
            "sonde_type": int(sondTyp[i]),
            # # "syn_date": h['typicalDate'],
            # # "syn_time": h['typicalTime'],
            "syn_timestamp": syntime.timestamp(),
            "firstSeen": float(relTime[i]),
            "lat": float(staLat[i]),
            "lon": float(staLon[i]),
            "elevation": float(staElev[i]),
        }
        fc = geojson.FeatureCollection([])
        fc.properties = properties

        lat_t = staLat[i]
        lon_t = staLon[i]
        firstSeen = fc.properties['firstSeen']
        previous_elevation = fc.properties['elevation'] - 100  # args.hstep

        t0 = T[i][0]
        p0 = P[i][0]
        h0 = staElev[i]

        #print(f"station h0={h0:.1f}m p0={p0} t0={t0}")
        prevSecsIntoFlight  = 0
        for n in range(0, len(P[i])):
            pn = P[i][n]

            # gross haque to determine rough time of sample
            height = round(barometric_equation_inv(h0, t0, p0, pn),1)
            secsIntoFlight = height2time(h0, height)
            delta = timedelta(seconds=secsIntoFlight)
            sampleTime = takeoff + delta

            properties = {
                "time": sampleTime.timestamp(),
    #            "gpheight": gpheight,
                "temp": round(T[i][n],2),
                "dewpoint": round(Td[i][n], 2),
                "pressure": P[i][n],
            }
            u = U[i][n]
            v = V[i][n]
            du = dv = 0
            if u > -9999.0 and v > -9999.0:
                properties["wind_u"] = u
                properties["wind_v"] = v
                dt = secsIntoFlight - prevSecsIntoFlight
                du = u * dt
                dv = v * dt
                lat_t,lon_t = latlonPlusDisplacement(lat=lat_t, lon=lon_t, u=du, v=dv)
                prevSecsIntoFlight = secsIntoFlight

            print(f"level={n} s={secsIntoFlight:.0f} {height:.1f}m p={pn} lon_t={lon_t} lat_t={lat_t} u={u} v={v} du={du:.1f} dv={dv:.1f} ", file=sys.stderr)

            f = geojson.Feature(geometry=geojson.Point((float(lat_t), float(lon_t), height)),
                                properties=properties)
            fc.features.append(f)
        fc.properties['lastSeen'] = sampleTime.timestamp()
        #print(fc)
        print(geojson.dumps(fc, indent=4,ignore_nan=True))




        sys.exit(0)

        # for n in range(len(P[i])-1):
        #     print(i, n, stn,  T[n], P[n])
        #     #print(n, P[n])times[i],
        continue
        t = datetime.utcfromtimestamp(relTime[i]).replace(tzinfo=pytz.utc)
        print(i, stn, times[i], t, len(raob['Psig'][i]), len(T[i]), len(P[i]))
        radiosonde = Radiosonde()
        radiosonde.sonde_validtime = times[i]
        radiosonde.temperatureK = T[i]
        radiosonde.dewpointK = Td[i]
        radiosonde.pressurehPA = P[i]
        radiosonde.u_windMS = U[i]
        radiosonde.v_windMS = V[i]
        # radiosonde.save()
        # print(radiosonde)


def extract_madis_data(file, stations):
    # print("Reading {}...".format(file))
    # print("\n\n############################\n")
    # flo = BytesIO()
    # ftp.retrbinary('RETR {}'.format(file), flo.write)
    # flo.seek(0)


    if stations and os.path.exists(stations):
        with open(stations , 'rb') as f:
            s = f.read().decode()
            stationdict = json.loads(s)
            #print(f'read stations from {stations}')
    else:
        print("no stations file")
        stationdict = dict()

    with gzip.open(file, 'rb') as f:
        nc = Dataset('inmemory.nc', memory=f.read())

        relTime = nc.variables['relTime'][:].filled(fill_value=np.nan)
        sondTyp = nc.variables['sondTyp'][:].filled(fill_value=np.nan)
        staLat = nc.variables['staLat'][:].filled(fill_value=np.nan)
        staLon = nc.variables['staLon'][:].filled(fill_value=np.nan)
        staElev = nc.variables['staElev'][:].filled(fill_value=np.nan)

        Tman = nc.variables['tpMan'][:].filled(fill_value=np.nan)
        DPDman = nc.variables['tdMan'][:].filled(fill_value=np.nan)
        wmo_ids = nc.variables['wmoStaNum'][:].filled(fill_value=np.nan)

        DPDsig = nc.variables['tdSigT'][:].filled(fill_value=np.nan)
        Tsig = nc.variables['tpSigT'][:].filled(fill_value=np.nan)
        synTimes = nc.variables['synTime'][:].filled(fill_value=np.nan)
        Psig = nc.variables['prSigT'][:].filled(fill_value=np.nan)
        Pman = nc.variables['prMan'][:].filled(fill_value=np.nan)
        #print(len(wmo_ids), wmo_ids, synTimes)

        Wspeed = nc.variables['wsMan'][:].filled(fill_value=np.nan)
        Wdir = nc.variables['wdMan'][:].filled(fill_value=np.nan)
        raob = {
            "relTime": relTime,
            "sondTyp": sondTyp,
            "staLat": staLat,
            "staLon": staLon,
            "staElev": staElev,

            "Tsig": Tsig,
            "Tdsig": Tsig - DPDsig,
            "Tman": Tman,
            "Psig": Psig,
            "Pman": Pman,
            "Tdman": Tman - DPDman,
            "Wspeed": Wspeed,
            "Wdir": Wdir,
            "times": [datetime.utcfromtimestamp(tim).replace(tzinfo=pytz.utc) for tim in synTimes],
            "wmo_ids": [str(id).zfill(5) for id in wmo_ids]
        }
        # print(raob)
        commit_sonde(raob, stationdict)


def read_madis():
    # Access the MADIS server and login
    ftp = FTP('madis-data.ncep.noaa.gov')
    print(ftp.login())
    print(ftp.cwd('point/raob/netcdf/'))

    # Iterate through the files, find what's been modified since the last call and extract the new data
    for file in ftp.nlst():
        file_timestamp = datetime.strptime(ftp.voidcmd("MDTM {}".format(file))[
                                           4:].strip(), '%Y%m%d%H%M%S').replace(microsecond=0).replace(tzinfo=pytz.utc)
        #record = UpdateRecord.objects.filter(filename=file).first()
        record = None
        now = datetime.now(tz=timezone.utc)

        if record:
            if (file_timestamp != record.updatetime and (now - file_timestamp).total_seconds() < 173000):
                print("{} will be updated. Old mod time was {}. New mod time is {}".format(
                    file, record.updatetime, file_timestamp))
                extract_madis_data(ftp, file)
                # breakpoint()
                record.updatetime = file_timestamp
                record.save()
        else:
            if (now - file_timestamp).total_seconds() < 173000:
                print("{} has not been downloaded before. Downloading..".format(file))
                extract_madis_data(ftp, file)
                #new_record = UpdateRecord(updatetime=file_timestamp, filename=file)
                print("Contents of {} recorded with timestamp {}".format(
                    file, file_timestamp))
                # new_record.save()


# def run():
if __name__ == "__main__":
    extract_madis_data('20210204_1100.gz', 'station_list.json')
    # extract_madis_data('madis/20210131_0600.gz')

    # UpdateRecord.delete_expired(10)
    #raob = read_madis()
