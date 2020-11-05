# ResNet test with Fabiol

In this exemple, we test a trained ResNet with the French dataset Fabiol.

## Command

To run the test :
```bash
python run.py --mode test --cfg exemples/fabiol/cfg/test_fabiol.cfg --checkpoint 98200
```

To extract the x-vectors of fabiol :
```sh
python extract.py --modeldir exemples/fabiol/model_dir --checkpoint 98200 --data test
``` 

