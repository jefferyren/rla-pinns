# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once

# BENCHMARKS
python ../yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m
# python ../yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=m2
# python ../yaml_to_sh.py sweeps/ENGD.yaml sweeps/ENGD.sh --qos=m3
# python ../yaml_to_sh.py sweeps/HessianFree.yaml sweeps/HessianFree.sh --qos=m4

# WOODBURY ENGD
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_exact.yaml sweeps/ENGD_woodbury_exact.sh --qos=m3 --array=50
# python ../yaml_to_sh.py sweeps/ENGD_woodbury_nystrom.yaml sweeps/ENGD_woodbury_nystrom.sh --qos=m2 --array=50

# SPRING
# python ../yaml_to_sh.py sweeps/SPRING_exact.yaml sweeps/SPRING_exact.sh --qos=m4 --array=50
# python ../yaml_to_sh.py sweeps/SPRING_nystrom.yaml sweeps/SPRING_nystrom.sh --qos=m4 --array=50