#writes full genome to a text file
skip_genome = False
chromosome_22_write = False

chromosome_22 = open('ch22.txt', 'w')
full_genome = open('human_ref_genome_without_ch22.txt', 'w')

with open('GCF_000001405.26_GRCh38_genomic.txt', 'r') as f:
    for line in f:
        if '>' in line:
            chromosome_22_write = False #chromosome 22 switch
            if 'alt' in line or 'unlo' in line or 'unplace' in line:
                skip_genome = True
            else:
                if 'NC_000022.11' in line:
                    chromosome_22_write = True #toggle to write to ch22.txt
                skip_genome = False
            continue
        
        if skip_genome:
            continue

        if 'N' in line:
            continue

        if chromosome_22_write:
            chromosome_22.write(line.upper())
        else:
            full_genome.write(line.upper())


full_genome.close()
chromosome_22.close()

print("Program ran successfully")
