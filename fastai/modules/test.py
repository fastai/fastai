class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



import os
rows, columns = os.popen('stty size', 'r').read().split()
text = "Warning: No active frommets remain. Continue?"
l = int((int(columns) - len(text))/2)

print (bcolors.OKBLUE + '-'*l + text + '-'*l + bcolors.ENDC)