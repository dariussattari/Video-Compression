#write full reference genome into individual chromosome files.
import re

skip_sequence = True
pattern = r"NC_0000(?:0[1-9]|1[0-9]|2[0-4])"

ch = 0

with open('GCF_000001405.26_GRCh38_genomic.txt', 'r') as f:
    for line in f:
        if '>' in line:
            try:
                s.close()
            except (NameError, AttributeError):
                pass

            if not re.search(pattern, line):
                skip_sequence = True
            else:
                skip_sequence = False
                ch += 1
                s = open(f'ch{ch}.txt', 'w')

                print(f'Writing to ch{ch}.txt')

            continue
        
        if skip_sequence:
            continue

        line = line.replace('N', '')

        if line:
            s.write(line.strip().upper())

#safely close the file
try:
    s.close()
except (NameError, AttributeError):
    pass



print("Program ran successfully")
