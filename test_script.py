import glob
import os

sets = glob.glob("C:/Users/Nik/Desktop/Testfiles/Benchmark/*.mps")
#os.system('python C:/Users/Nik/PycharmProjects/PrimalHeuristics/SaP_v11.py C:/Users/Nik/Desktop/Testfiles/Benchmark/air03.mps')

for element in sets:
    os.system('python C:/Users/Nik/PycharmProjects/PrimalHeuristics/SaP_v11.py ' + element)
