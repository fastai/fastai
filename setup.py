from pkg_resources import parse_version
from configparser import ConfigParser
import setuptools,re,sys
assert parse_version(setuptools.__version__)>=parse_version('36.2')

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

cfg_keys = 'version description keywords author author_email'.split()
expected = cfg_keys + "lib_name user branch license status min_python audience language".split()
for o in expected: assert o in cfg, "missing expected setting: {}".format(o)
setup_cfg = {o:cfg[o] for o in cfg_keys}

if len(sys.argv)>1 and sys.argv[1]=='version':
    print(setup_cfg['version'])
    exit()

licenses = {
    'apache2': ('Apache Software License 2.0','OSI Approved :: Apache Software License'),
}
statuses = [ '1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
    '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive' ]
py_versions = '2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 3.10'.split()
min_python = cfg['min_python']
lic = licenses[cfg['license']]

requirements = ['pip', 'packaging']
if cfg.get('requirements'): requirements += cfg.get('requirements','').split()
if cfg.get('pip_requirements'): requirements += cfg.get('pip_requirements','').split()
dev_requirements = (cfg.get('dev_requirements') or '').split()

long_description = open('README.md', encoding='utf8').read()
# ![png](docs/images/output_13_0.png)
for ext in ['png', 'svg']:
    long_description = re.sub(r'!\['+ext+'\]\((.*)\)', '!['+ext+']('+'https://raw.githubusercontent.com/{}/{}'.format(cfg['user'],cfg['lib_name'])+'/'+cfg['branch']+'/\\1)', long_description)
    long_description = re.sub(r'src=\"(.*)\.'+ext+'\"', 'src=\"https://raw.githubusercontent.com/{}/{}'.format(cfg['user'],cfg['lib_name'])+'/'+cfg['branch']+'/\\1.'+ext+'\"', long_description)

setuptools.setup(
    name = 'fastai',
    license = lic[0],
    classifiers = [
        'Development Status :: ' + statuses[int(cfg['status'])],
        'Intended Audience :: ' + cfg['audience'].title(),
        'License :: ' + lic[1],
        'Natural Language :: ' + cfg['language'].title(),
    ] + ['Programming Language :: Python :: '+o for o in py_versions[py_versions.index(min_python):]],
    url = cfg['git_url'],
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = requirements,
    extras_require={ 'dev': dev_requirements },
    python_requires  = '>=' + cfg['min_python'],
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    entry_points = {
        'console_scripts': cfg.get('console_scripts','').split(),
        'nbdev': [f'{cfg.get("lib_path")}={cfg.get("lib_path")}._modidx:d']
    },
    **setup_cfg)

