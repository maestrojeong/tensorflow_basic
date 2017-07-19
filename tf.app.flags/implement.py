import os

for i in range(3):
    os.system("python practice.py --var {}".format(i))

'''
0.0
{'__parsed': True, '__flags': {'var1': 0.0}}
{'var1': 0.0}
1.0
{'__parsed': True, '__flags': {'var1': 1.0}}
{'var1': 1.0}
2.0
{'__parsed': True, '__flags': {'var1': 2.0}}
{'var1': 2.0}
'''

