# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once

# BENCHMARKS
# python ../yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m3
# python ../yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=m4
# python ../yaml_to_sh.py sweeps/ENGD.yaml sweeps/ENGD.sh --qos=m4
# python ../yaml_to_sh.py sweeps/SPRING.yaml sweeps/SPRING.sh --qos=m4 --array=50

# WOODBURY ENGD
python ../yaml_to_sh.py sweeps/ENGD_woodbury_exact.yaml sweeps/ENGD_woodbury_exact.sh --qos=m4 --array=50
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_naive_10.yaml sweeps/ENGD_woodbury_naive_10.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_naive_50.yaml sweeps/ENGD_woodbury_naive_50.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_nystrom_10.yaml sweeps/ENGD_woodbury_nystrom_10.sh --qos=m4 --array=50
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_nystrom_50.yaml sweeps/ENGD_woodbury_nystrom_50.sh --qos=m3 --array=50



# SPRING

python ../yaml_to_sh.py sweeps/SPRING_exact.yaml sweeps/SPRING_exact.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_naive_10.yaml sweeps/SPRING_naive_10.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_naive_50.yaml sweeps/SPRING_naive_50.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_nystrom_10.yaml sweeps/SPRING_nystrom_10.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_nystrom_50.yaml sweeps/SPRING_nystrom_50.sh --qos=m3 --array=50