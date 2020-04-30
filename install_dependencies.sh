#sudo pip3 uninstall numpy
#sudo pip3 uninstall scipy
#sudo pip3 uninstall gdal

sudo apt-get install python3.6-dev libmysqlclient-dev -y
sudo apt-get install python3-numpy -y
sudo pip3 install numpy scipy

add-apt-repository ppa:ubuntugis/ubuntugis-unstable -y
apt-get update
apt-get install python-numpy gdal-bin libgdal-dev python3-rtree -y
sudo aptitude install libgdal26 libgdal-dev -y

sudo apt-get install build-essential python-all-dev -y

cd ./temp
wget http://download.osgeo.org/gdal/3.0.4/gdal-3.0.4.tar.gz
tar xvfz gdal-3.0.4.tar.gz

python setup.py build_ext --include-dirs=/usr/include/gdal/
python setup.py install

./configure --with-python
./configure

sudo make
sudo make install
sudo ldconfig

sudo apt-get install python3.6-dev libmysqlclient-dev -y
sudo pip install --upgrade setuptools --user python
sudo pip install --upgrade setuptools
sudo apt-get install libpq-dev python-dev -y
sudo pip3 install python-language-server
sudo pip3 install --upgrade pip setuptools wheel
sudo pip3 install rtree


sudo pip3 install ez_setup
#import ez_setup
sudo python3 import ez_setup, easy_install gdal
pip3 install --no-cache-dir solaris
sudo apt install gdal

sudo apt install solaris -y

sudo pip3 install torch --upgrade
sudo pip3 install torchvision --upgrade
#sudo pip3 install tensorflow-gpu==1.14

apt-get install -qq curl g++ make -y
curl -L http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz | tar xz

cd ./spatialindex-src-1.8.5
./configure
make
make install
ldconfig

sudo pip3 install python-language-server
sudo pip3 install geopandas
sudo pip3 install rtree

sudo pip3 install gdal
sudo apt-get install libgeos-dev
sudo pip3 install solaris

ldconfig