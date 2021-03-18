git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
pip install -e .
git clone https://github.com/andersjo/pyrouge.git rouge
pyrouge_set_rouge_path pyrouge/rouge/tools/ROUGE-1.5.5/
#########################
conda install -c bioconda perl-xml-libxml
#########################
cd pyrouge/rouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db # only if exist
cd WordNet-2.0-Exceptions
rm WordNet-2.0.exc.db # only if exist

./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db