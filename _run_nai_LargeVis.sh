#prepare file for largevis -> nai.txt
python largevisInput.py
#largevis
#for dim in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
#for dim in {3..12}
for dim in 6 8 10 14 20 27 44 70 100
do
  echo #largevis
  echo "m = $dim"
  python2.7 ../../LargeVis_run.py -input nai.txt -output nai_dim$dim.txt -outdim $dim
  #make files from large vis
  python3.5 transformer.py $dim       #nai_dim$dim.txt
  #merge csv files:
  paste -d , train_id.csv numerai_training_data.csv trainLargevis_$dim.csv > numerai_training_data_$dim.csv
  paste -d , numerai_tournament_data.csv testLargevis_$dim.csv > numerai_tournament_data_$dim.csv
done




