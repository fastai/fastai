# to quickstart: 
sphinx-quickstart


# create rst files, one rst for each module
# -f: force overwirte
# -e: seperate (or you'll get one giant rst file. ) 
# -o: outputdir
sphinx-apidoc -e -f -o source/ ../fastai

# to create html: 
make html
