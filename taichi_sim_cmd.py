import os
import subprocess
import argparse
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",type=str,default="tacchi_obj/obj_100")
    parser.add_argument("--output_dir",type=str,default="results_tr/posmap")
    parser.add_argument("--unsuccess_file",type=str,default="results_tr/unsuccess.json")
    parser.add_argument("--num",type=int,default=200)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    file_len = args.num
    print("total files from {}:{}\n".format(args.dataset_dir, file_len))
    unsuccess = []
    for i in range(file_len):
        process = subprocess.Popen(['python3','taichi_sim.py','--dataset_dir',args.dataset_dir,'--output_dir',args.output_dir,'--index',str(i)])
        process.wait()  # must be serialized
        if process.returncode != 0:
            unsuccess.append(i)
    json.dump(unsuccess, open(args.unsuccess_file, 'w'))