#  This program was automatically generated with bufr_dump -Dpython
#  Using ecCodes version: 2.19.1
#
# bufr_dump -Dpython A_IUSD02LOWM210300_C_EDZW_20210121040000_59339751.bin >dump.py
# and hand edited..

import traceback
import sys
from eccodes import *
import pprint


def bufr_decode(input_file):
    f = open(input_file, 'rb')
    # Message number 1
    ibufr = codes_bufr_new_from_file(f)
    codes_set(ibufr, 'unpack', 1)

    header = dict()
    arraykeys = ['delayedDescriptorReplicationFactor',
                 'extendedDelayedDescriptorReplicationFactor',
                 'unexpandedDescriptors']

    for k in arraykeys:
        header[k] = codes_get_array(ibufr, k)

    scalarkeys = [
        'edition',
        'masterTableNumber',
        'bufrHeaderCentre',
        'bufrHeaderSubCentre',
        'updateSequenceNumber',
        'dataCategory',
        'internationalDataSubCategory',
        'dataSubCategory',
        'masterTablesVersionNumber',
        'localTablesVersionNumber',
        'typicalYear',
        'typicalMonth',
        'typicalDay',
        'typicalHour',
        'typicalMinute',
        'typicalSecond',
        'numberOfSubsets',
        'observedData',
        'compressedData',
        'radiosondeSerialNumber',
        'radiosondeAscensionNumber',
        'radiosondeReleaseNumber',
        'observerIdentification',
        'radiosondeCompleteness',
        'radiosondeConfiguration',
        'correctionAlgorithmsForHumidityMeasurements',
        'radiosondeGroundReceivingSystem',
        'radiosondeOperatingFrequency',
        'balloonManufacturer',
        'weightOfBalloon',
        'amountOfGasUsedInBalloon',
        'pressureSensorType',
        'temperatureSensorType',
        'humiditySensorType',
        'geopotentialHeightCalculation',
        'softwareVersionNumber',
        'reasonForTermination',
        'blockNumber',
        'stationNumber',
        'radiosondeType',
        'solarAndInfraredRadiationCorrection',
        'trackingTechniqueOrStatusOfSystem',
        'measuringEquipmentType',
        'timeSignificance',
        'year',
        'month',
        'day',
        'hour',
        'minute',
        'second',
        'latitude',
        'longitude',
        'heightOfStationGroundAboveMeanSeaLevel',
        'heightOfBarometerAboveMeanSeaLevel',
        'height',
        'text']

    for k in scalarkeys:
        header[k] = codes_get(ibufr, k)

    keys = ['timePeriod', 'extendedVerticalSoundingSignificance',
            'pressure', 'nonCoordinateGeopotentialHeight',
            'latitudeDisplacement', 'longitudeDisplacement',
            'airTemperature', 'dewpointTemperature',
            'windDirection', 'windSpeed']

    samples = []
    try:
        i = 1
        while True:
            sample = dict()
            for k in keys:
                name = f"#{i}#{k}"
                sample[k] = codes_get(ibufr, name)
            samples.append(sample)
            i += 1
    except Exception as e:
        print(f"loop done {e}, samples = {len(samples)}", file=sys.stderr)
        pass

    header['samples'] = samples
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(header)
    codes_release(ibufr)
    f.close()


def main():
    if len(sys.argv) < 2:
        print('Usage: ', sys.argv[0], ' BUFR_file', file=sys.stderr)
        sys.exit(1)

    try:
        bufr_decode(sys.argv[1])
    except CodesInternalError as err:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
