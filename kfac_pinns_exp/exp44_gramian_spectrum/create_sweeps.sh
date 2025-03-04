# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once
# python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/poisson.yaml sweeps/poisson.sh --qos=m3 --array_max_active=5 --array=1
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/heat.yaml sweeps/heat.sh --qos=m3 --array_max_active=5 --array=1