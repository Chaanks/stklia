# ResNet test & extraction with VoxCeleb

In this exemple, we test a trained ResNet on VoxCeleb-2 Dev with VoxCeleb-1 and we extract VoxCeleb-1 x-vectors.

## Command

Set Kaldi path, you can use the script in the `recipes/utils` folder. Remember to change the KALDI_ROOT variable using your path.
```sh
source ../utils/kaldi_path.sh
```
> **Note:**  
> As a first test to check the installation, open a bash shell, type "copy-feats" or > >> "copy-vector" and make sure no errors appear.

To run the test :
```bash
python run.py --mode test --cfg exemples/voxceleb/cfg/voxceleb.cfg --checkpoint 5800
```

To extract the x-vectors of VoxCeleb1 :
```sh
python extract.py --modeldir exemples/voxceleb/model_dir --checkpoint 5800 --data eval
``` 
