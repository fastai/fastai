This is a custom setup.py that will build torchvision-nightly (doesn't depend torch, since otherwise dependencies are all broken), we now depend on torchvision-nightly in fastai

   git clone https://github.com/pytorch/vision.git

   cp setup.py vision/
   cd vision

   rm -rf dist/*
   python setup.py sdist
   python setup.py bdist_wheel
   ls -l dist

   echo "Uploading dist/* to pypi"
   twine upload dist/*
   # twine upload --repository testpypi dist/*
