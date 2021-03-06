# -*- coding:utf-8 -*-
'''
This is a class that converts json file to csv file.
You can load this class easily with an input file name and output file name.
This class is made for chainer log parser but it'll be modified to an another type of file converter.

'''
import csv
import json
import sys

class ConvertFormat(object):
    def __init__(self,inputfile,outputfile,output_ext='json'):
        self.inputfile = inputfile
        self.outputfile = outputfile
        self.output_ext = output_ext

    def json2csv(self):
        with open(self.inputfile, 'r') as fr, open(self.outputfile, 'w') as fw:
            dataWriter = csv.writer(fw)
            jsonData = json.load(fr)
            keys = ["elapsed_time", "epoch", "iteration","main/loss","validation/main/loss", "main/accuracy", "validation/main/accuracy"]
            rows = [[row[key] for key in keys] for row in jsonData]
            dataWriter.writerow(keys)
            dataWriter.writerows(rows)
if __name__ == '__main__':
    inputfile = sys.argv[1]
    output = 'result.csv'
    conv = ConvertFormat(inputfile, output)
    conv.json2csv()

