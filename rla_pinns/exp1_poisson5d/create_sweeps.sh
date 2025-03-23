# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once
# python ../yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m3
# python ../yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=m4
python ../yaml_to_sh.py sweeps/ENGD.yaml sweeps/ENGD.sh --qos=m4
# python ../yaml_to_sh.py sweeps/RNGD.yaml sweeps/RNGD.sh --qos=m3 --array=81
