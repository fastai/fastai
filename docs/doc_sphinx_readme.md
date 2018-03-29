## Sphinx Documentation Compilation guide

This documentation helps uer compile their own documentations for fast.ai library. Two methods are included: 
1. easy way: run .sh script
2. less easy way: use 3 to 4 sphinx commands to get the same result.

[Sphinx](http://www.sphinx-doc.org/en/stable/tutorial.html) is an easy tool used for documentation compilation, including python. 
For us, we're using Sphinx to generate .rst (reStructuredText) files, which inturn we make into html files. 
The end result are python documentation webpages the are easily deployable. 

### Easy Way: 
1. install sphinx
`apt-get install python-sphinx`

2. go to sphinx_docs/, 
make shell script excecutable `chmod +x make_html_doc.sh`

3. execute `./make_html_doc.sh`


### Less Easy Way: 

(Step 3 and 4 can be done in any order. )

1. install sphinx
```
apt-get install python-sphinx
```

2. to quickstart, go to sphinx_docs/ directory and:  
```
sphinx-quickstart
```
the yes and nos should be pretty straightforward. 

3. create rst files in sphinx_docs/source directory 
```
sphinx-apidoc -e -f -o source/ ../fastai
```
this create rst files, one rst for each module
  * -f: force overwirte existing documentations
  * -e: seperate (or you'll get one giant rst file, which turns into one giant html file ) 
  * -o: outputdir

4. go to sphinx_docs/source/conf.py, and add following lines: 
```python
    import os
    import sys
    sys.path.insert(0, os.path.abspath('../..'))
```
    
5. to create html under sphinx_docs/build/html directory: 
```
make html
```

now you should see deployable html documentation. 


### Conclusion: 

Which ever way you built your documentation, 
just open sphinx_docs/build/html/index.html and enjoy!
